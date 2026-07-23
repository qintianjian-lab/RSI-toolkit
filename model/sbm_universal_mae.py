from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sbm_universal import SBM_Universal_Framework as SBMUniversalV1
from model.sbm_universal_v2 import SBM_Universal_Framework as SBMUniversalV2


@dataclass(frozen=True)
class UniversalMAEOutput:
    recon: torch.Tensor  # [B,1,L]
    fmap: torch.Tensor   # [B,C,L']


def _resolve_backbone_class(encoder_model: str):
    key = str(encoder_model or "sbm_universal").strip().lower()
    if key in {"sbm_universal", "sbm_universal_v1", "v1"}:
        return SBMUniversalV1
    if key in {"sbm_universal_v2", "v2"}:
        return SBMUniversalV2
    raise ValueError("encoder_model must be 'sbm_universal' or 'sbm_universal_v2'")


class SBMUniversalEncoder(nn.Module):
    """
    Backbone-only encoder used for masked-reconstruction pre-training.
    """

    def __init__(
        self,
        in_channel: int,
        spectrum_size: int,
        *,
        encoder_model: str = "sbm_universal",
        block_type: str | dict = "mspc",
        stem_type: str | dict = "mspc",
        preband_split: bool = True,
        stage_depths: int | tuple[int, int, int, int] | list[int] = (2, 2, 2, 1),
        band_stage_depths: dict | None = None,
        base_channels: int = 48,
        drop_path: float = 0.1,
        mspc_stem_kwargs: dict[str, Any] | None = None,
        mspc_block_kwargs: dict[str, Any] | None = None,
        k_low: int = 65,
        k_mid: int = 17,
        global_mixer: str = "none",
        global_mixer_depth: int = 1,
        global_attn_heads: int = 4,
        global_pos_encoding: str = "none",
        global_learned_pos_max_len: int = 512,
        global_mixer_insert_after: str = "stage4",
    ):
        super().__init__()
        backbone_cls = _resolve_backbone_class(encoder_model)
        backbone_kwargs: dict[str, Any] = {
            "in_channel": in_channel,
            "out_channel": 2,  # dummy (head unused in pretraining)
            "spectrum_size": spectrum_size,
            "block_type": block_type,
            "stem_type": stem_type,
            "preband_split": preband_split,
            "stage_depths": stage_depths,
            "band_stage_depths": band_stage_depths,
            "base_channels": base_channels,
            "drop_path": drop_path,
            "mspc_stem_kwargs": mspc_stem_kwargs,
            "mspc_block_kwargs": mspc_block_kwargs,
            "k_low": k_low,
            "k_mid": k_mid,
        }
        if backbone_cls is SBMUniversalV2:
            backbone_kwargs.update(
                {
                    "global_mixer": global_mixer,
                    "global_mixer_depth": global_mixer_depth,
                    "global_attn_heads": global_attn_heads,
                    "global_pos_encoding": global_pos_encoding,
                    "global_learned_pos_max_len": global_learned_pos_max_len,
                    "global_mixer_insert_after": global_mixer_insert_after,
                }
            )
        self.backbone = backbone_cls(**backbone_kwargs)
        self.out_channels = int(self.backbone.stage4.out_ch)

    def forward_fmap(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        if m.preband_split:
            low, mid, high = m.pre_split(x)
            low = m.stem_low(low)
            mid = m.stem_mid(mid)
            high = m.stem_high(high)
            x = m.stem_fuse(low, mid, high)
        else:
            x = m.stem(x)
        x = m.stage1(x)
        x = m.stage2(x)
        x = m.stage3(x)
        if getattr(m, "global_mixer_insert_after", None) == "stage3":
            x = m.global_mixer(x)
        x = m.stage4(x)
        if getattr(m, "global_mixer_insert_after", None) == "stage4":
            x = m.global_mixer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_fmap(x)


class WeakProgressiveDecoder1D(nn.Module):
    """
    Deliberately weak decoder:
    - operate mostly on low-resolution feature map
    - only one optional full-resolution refine conv
    """

    def __init__(
        self,
        in_channels: int,
        spectrum_size: int,
        hidden_channels: int = 128,
        fullres_refine: bool = True,
    ):
        super().__init__()
        self.spectrum_size = int(spectrum_size)
        h = int(max(in_channels, hidden_channels))
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, h, kernel_size=1, bias=False),
            nn.BatchNorm1d(h),
            nn.GELU(),
        )
        # fixed 3 light upsample steps (most work remains low-res)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(h, h // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(h // 2),
            nn.GELU(),
            nn.ConvTranspose1d(h // 2, h // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(h // 4),
            nn.GELU(),
            nn.ConvTranspose1d(h // 4, h // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(h // 8),
            nn.GELU(),
        )
        c = max(16, h // 8)
        self.refine = (
            nn.Sequential(
                nn.Conv1d(c, c, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(c),
                nn.GELU(),
            )
            if fullres_refine
            else nn.Identity()
        )
        self.out = nn.Conv1d(c, 1, kernel_size=1, bias=True)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        x = self.proj(fmap)
        x = self.up(x)
        if int(x.shape[-1]) != self.spectrum_size:
            x = F.interpolate(x, size=self.spectrum_size, mode="linear", align_corners=False)
        x = self.refine(x)
        x = self.out(x)
        return x


class SBMUniversalMAE(nn.Module):
    def __init__(
        self,
        in_channel: int,
        spectrum_size: int,
        *,
        encoder_model: str = "sbm_universal",
        block_type: str | dict = "mspc",
        stem_type: str | dict = "mspc",
        preband_split: bool = True,
        stage_depths: int | tuple[int, int, int, int] | list[int] = (2, 2, 2, 1),
        band_stage_depths: dict | None = None,
        base_channels: int = 48,
        drop_path: float = 0.1,
        mspc_stem_kwargs: dict[str, Any] | None = None,
        mspc_block_kwargs: dict[str, Any] | None = None,
        k_low: int = 65,
        k_mid: int = 17,
        global_mixer: str = "none",
        global_mixer_depth: int = 1,
        global_attn_heads: int = 4,
        global_pos_encoding: str = "none",
        global_learned_pos_max_len: int = 512,
        global_mixer_insert_after: str = "stage4",
        decoder_hidden: int = 128,
        decoder_fullres_refine: bool = True,
    ):
        super().__init__()
        self.encoder = SBMUniversalEncoder(
            in_channel=in_channel,
            spectrum_size=spectrum_size,
            encoder_model=encoder_model,
            block_type=block_type,
            stem_type=stem_type,
            preband_split=preband_split,
            stage_depths=stage_depths,
            band_stage_depths=band_stage_depths,
            base_channels=base_channels,
            drop_path=drop_path,
            mspc_stem_kwargs=mspc_stem_kwargs,
            mspc_block_kwargs=mspc_block_kwargs,
            k_low=k_low,
            k_mid=k_mid,
            global_mixer=global_mixer,
            global_mixer_depth=global_mixer_depth,
            global_attn_heads=global_attn_heads,
            global_pos_encoding=global_pos_encoding,
            global_learned_pos_max_len=global_learned_pos_max_len,
            global_mixer_insert_after=global_mixer_insert_after,
        )
        self.decoder = WeakProgressiveDecoder1D(
            in_channels=self.encoder.out_channels,
            spectrum_size=spectrum_size,
            hidden_channels=decoder_hidden,
            fullres_refine=bool(decoder_fullres_refine),
        )

    def forward(self, x: torch.Tensor) -> UniversalMAEOutput:
        fmap = self.encoder.forward_fmap(x)
        recon = self.decoder(fmap)
        return UniversalMAEOutput(recon=recon, fmap=fmap)
