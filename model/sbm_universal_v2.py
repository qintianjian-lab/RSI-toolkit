from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

# =========================================================================
#  1. Basic Components
# =========================================================================

class StochasticDepth(nn.Module):
    """Stochastic depth (DropPath)."""
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class LayerNormChannel(nn.Module):
    """LayerNorm for [B, C, L]"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


def _normalize_stage_depths(depths, n_stages: int = 4):
    """
    Normalize stage depths to a list of length n_stages.
    Accepts: int -> broadcast, list/tuple -> validate length.
    """
    if isinstance(depths, int):
        if depths < 0:
            raise ValueError("depths must be >= 0")
        return [depths] * n_stages
    if isinstance(depths, (list, tuple)):
        if len(depths) != n_stages:
            raise ValueError(f"depths must have length {n_stages}, got {len(depths)}")
        depths = [int(d) for d in depths]
        if any(d < 0 for d in depths):
            raise ValueError("depths must be >= 0")
        return depths
    raise TypeError("depths must be int or list/tuple of ints")


def _normalize_per_band_stage_depths(depths, n_stages: int = 4):
    """
    Normalize per-band stage depths.
    - None -> returns None
    - dict: keys in {'low','mid','high'}; values int or list/tuple (len n_stages)
    """
    if depths is None:
        return None
    if not isinstance(depths, dict):
        raise TypeError("band_depths must be a dict or None")
    out = {}
    for k in ["low", "mid", "high"]:
        if k in depths and depths[k] is not None:
            out[k] = _normalize_stage_depths(depths[k], n_stages=n_stages)
    return out if out else None

class ECA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        k = int(math.ceil(math.log(max(channels, 2), 2) / 2 + 0.5))
        if k % 2 == 0: k += 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(1, 2)
        return x * y


class SinCosPositionalEncoding1D(nn.Module):
    """Parameter-free sin/cos position encoding for [B,L,C]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C]
        _, L, C = x.shape
        device = x.device
        dtype = x.dtype
        pe = torch.zeros((L, C), device=device, dtype=dtype)
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # [L,1]
        div = torch.exp(torch.arange(0, C, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(C, 1)))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)  # [1,L,C]


class LearnedPositionalEncoding1D(nn.Module):
    """Learned absolute position embedding for [B,L,C] with interpolation when L changes."""
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.dim = int(dim)
        self.max_len = int(max_len)
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, self.dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C]
        L = x.shape[1]
        if L == self.max_len:
            return x + self.pos
        pe = F.interpolate(self.pos.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
        return x + pe


class TransformerMixer1D(nn.Module):
    """
    Light global mixer (Transformer block) applied on [B,C,L] by attending over L.
    Kept small and optional for global sequence mixing.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
        pos_encoding: str = "none",  # 'none' | 'sincos' | 'learned'
        learned_pos_max_len: int = 512,
    ):
        super().__init__()
        c = int(channels)
        h = int(num_heads)
        assert c % h == 0, f"channels ({c}) must be divisible by num_heads ({h})"
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=h, dropout=float(attn_dropout), batch_first=True)
        self.proj_drop = nn.Dropout(float(proj_dropout))
        self.drop = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

        hidden = int(round(c * float(mlp_ratio)))
        self.norm2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(float(proj_dropout)),
            nn.Linear(hidden, c, bias=True),
            nn.Dropout(float(proj_dropout)),
        )

        pe = str(pos_encoding or "none").lower()
        if pe in ["none", "off", "no"]:
            self.pos = None
        elif pe in ["sincos", "sin", "sinusoidal"]:
            self.pos = SinCosPositionalEncoding1D(c)
        elif pe in ["learned", "abs", "absolute"]:
            self.pos = LearnedPositionalEncoding1D(c, max_len=int(learned_pos_max_len))
        else:
            raise ValueError("pos_encoding must be 'none'|'sincos'|'learned'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L] -> [B,L,C]
        xl = x.transpose(1, 2)
        if self.pos is not None:
            xl = self.pos(xl)
        y = self.norm1(xl)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = self.proj_drop(y)
        xl = xl + self.drop(y)
        y2 = self.mlp(self.norm2(xl))
        xl = xl + self.drop(y2)
        return xl.transpose(1, 2)


def _resolve_norm_layer(norm_layer):
    if isinstance(norm_layer, str):
        key = norm_layer.upper()
        if key == "BN":
            return nn.BatchNorm1d
        if key == "LN":
            # Keep compatibility with mspc_net kwargs; LN for [B,C,L] in this file uses LayerNormChannel.
            return LayerNormChannel
    if isinstance(norm_layer, type):
        return norm_layer
    return nn.BatchNorm1d


def _resolve_act_layer(act_layer):
    if isinstance(act_layer, str):
        key = act_layer.upper()
        if key == "RELU":
            return nn.ReLU
        if key == "GELU":
            return nn.GELU
        if key in ("NONE", "IDENTITY"):
            return nn.Identity
    if isinstance(act_layer, type):
        return act_layer
    return nn.ReLU

# =========================================================================
#  2. Core Block Wrappers
#  All blocks preserve shape: input [B,C,L] -> output [B,C,L].
# =========================================================================

# [1] SBM-native MSPC-style block with ECA/ELA attention.
class SBMNativeBlock(nn.Module):
    def __init__(self, channels: int, drop_path=0.0):
        super().__init__()
        # ChannelSplitPartial Logic
        c = channels // 4
        self.k1 = nn.Conv1d(c, c, 5, padding=2, groups=c, bias=False)
        self.k2 = nn.Conv1d(c, c, 45, padding=22, groups=c, bias=False)
        self.k3 = nn.Conv1d(c, c, 405, padding=202, groups=c, bias=False)
        self.mix = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.attn = ECA(channels) # default attention module
        self.drop = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        c = x.size(1) // 4
        x1 = x[:, :c, :]
        x2 = x[:, c:2*c, :]
        x3 = x[:, 2*c:3*c, :]
        x4 = x[:, 3*c:, :]
        y = torch.cat([self.k1(x1), self.k2(x2), self.k3(x3), x4], dim=1)
        y = self.mix(y)
        y = self.bn(y)
        y = self.attn(y)
        return F.gelu(self.drop(y) + shortcut)

# [2] MSPC-aligned block.
class PartialConv1DAligned(nn.Module):
    """
    Align with model/mspc_net.py PartialConv1D:
    - split channels to 3 conv branches + rest(identity)
    - each conv branch uses channel//n_div
    """
    def __init__(
        self,
        channels: int,
        n_div: int = 16,
        pc_conv_size: int = 5,
        pc_conv_size_scale: float = 9.0,
    ):
        super().__init__()
        n_div = max(1, int(n_div))
        branch_ch = channels // n_div
        self.channel_div1 = branch_ch
        self.channel_div2 = branch_ch
        self.channel_div3 = branch_ch
        self.channel_conv = self.channel_div1 + self.channel_div2 + self.channel_div3
        if self.channel_conv > channels:
            # Guard against invalid n_div for narrow channels.
            self.channel_div1 = 0
            self.channel_div2 = 0
            self.channel_div3 = 0
            self.channel_conv = 0

        k1 = max(1, int(pc_conv_size))
        k2 = max(1, int(pc_conv_size * pc_conv_size_scale))
        k3 = max(1, int(pc_conv_size * pc_conv_size_scale * pc_conv_size_scale))

        self.conv1 = nn.Conv1d(self.channel_div1, self.channel_div1, k1, stride=1, padding="same", bias=False) \
            if self.channel_div1 > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(self.channel_div2, self.channel_div2, k2, stride=1, padding="same", bias=False) \
            if self.channel_div2 > 0 else nn.Identity()
        self.conv3 = nn.Conv1d(self.channel_div3, self.channel_div3, k3, stride=1, padding="same", bias=False) \
            if self.channel_div3 > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.channel_div1
        c2 = self.channel_div2
        c3 = self.channel_div3
        x_div1 = x[:, :c1, :]
        x_div2 = x[:, c1:c1 + c2, :]
        x_div3 = x[:, c1 + c2:c1 + c2 + c3, :]
        x_rest = x[:, c1 + c2 + c3:, :]

        y1 = self.conv1(x_div1)
        y2 = self.conv2(x_div2)
        y3 = self.conv3(x_div3)
        return torch.cat([y1, y2, y3, x_rest], dim=1)


class PureMSPCBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        drop_path=0.0,
        n_div: int = 16,
        mp_ratio: float = 2.0,
        pc_conv_size: int = 5,
        pc_conv_size_scale: float = 9.0,
        norm_layer: str | type = "BN",
        act_layer: str | type = "RELU",
        attention: str = "ECA_F",
    ):
        super().__init__()
        norm_cls = _resolve_norm_layer(norm_layer)
        act_cls = _resolve_act_layer(act_layer)

        hidden = max(1, int(channels * float(mp_ratio)))
        if act_cls == nn.ReLU:
            act = act_cls(inplace=True)
        else:
            act = act_cls()

        self.spatial_mixing = PartialConv1DAligned(
            channels=channels,
            n_div=n_div,
            pc_conv_size=pc_conv_size,
            pc_conv_size_scale=pc_conv_size_scale,
        )
        self.mlp = nn.Sequential(
            ECA(channels) if attention == "ECA_F" else nn.Identity(),
            nn.Conv1d(channels, hidden, 1, bias=False),
            norm_cls(hidden),
            act,
            nn.Conv1d(hidden, channels, 1, bias=False),
            ECA(channels) if attention == "ECA_B" else nn.Identity(),
        )
        self.drop = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        mixed = self.spatial_mixing(x)
        return shortcut + self.drop(self.mlp(mixed))

# =========================================================================
#  3. Configurable Block/Stem Processors and Framework Components
# =========================================================================

BLOCK_REGISTRY = {
    'sbm': SBMNativeBlock,
    'mspc': PureMSPCBlock,
}

class UniversalBandProcessor(nn.Module):
    """
    Configurable band processor selected by block_type.
    """
    def __init__(self, channels: int, block_type: str, drop_path: float = 0.0, mspc_block_kwargs: dict | None = None):
        super().__init__()
        if block_type not in BLOCK_REGISTRY:
            raise ValueError(f"Unknown block type: {block_type}. Available: {list(BLOCK_REGISTRY.keys())}")
        block_kwargs = dict(mspc_block_kwargs or {}) if block_type == "mspc" else {}

        # Instantiate the selected block.
        self.block = BLOCK_REGISTRY[block_type](channels, drop_path, **block_kwargs)

    def forward(self, x):
        return self.block(x)


class MSPCStem1D(nn.Module):
    """
    MSPC PatchEmbed-aligned stem.
    Align with model/mspc_net.py PatchEmbed:
      Conv1d(kernel=patch_conv_size, stride=patch_conv_stride, bias=False) + norm
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        patch_conv_size: int = 3,
        patch_conv_stride: int = 1,
        norm_layer: str | type = "BN",
        act_layer: str | type = "NONE",
    ):
        super().__init__()
        norm_cls = _resolve_norm_layer(norm_layer)
        act_cls = _resolve_act_layer(act_layer)
        layers = [
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=max(1, int(patch_conv_size)),
                stride=max(1, int(patch_conv_stride)),
                bias=False,
            ),
            norm_cls(out_ch) if norm_cls is not None else nn.Identity(),
        ]
        if act_cls != nn.Identity:
            layers.append(act_cls(inplace=True) if act_cls == nn.ReLU else act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MSPCLegacyStem1D(nn.Module):
    """Legacy MSPC stem (pre-alignment version)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


STEM_REGISTRY = {
    "mspc": MSPCStem1D,
    "mspc_legacy": MSPCLegacyStem1D,
}


class UniversalStemProcessor(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stem_type: str, mspc_stem_kwargs: dict | None = None):
        super().__init__()
        if stem_type not in STEM_REGISTRY:
            raise ValueError(f"Unknown stem type: {stem_type}. Available: {list(STEM_REGISTRY.keys())}")
        stem_kwargs = dict(mspc_stem_kwargs or {}) if stem_type == "mspc" else {}
        self.stem = STEM_REGISTRY[stem_type](in_ch, out_ch, **stem_kwargs)

    def forward(self, x):
        return self.stem(x)


class BandSplit(nn.Module):
    """Three-band spectral split."""
    def __init__(self, channels: int, k_low: int = 65, k_mid: int = 17):
        super().__init__()
        self.low = nn.Conv1d(channels, channels, k_low, padding=k_low // 2, groups=channels, bias=False)
        self.mid = nn.Conv1d(channels, channels, k_mid, padding=k_mid // 2, groups=channels, bias=False)
        nn.init.constant_(self.low.weight, 1.0 / k_low)
        nn.init.constant_(self.mid.weight, 1.0 / k_mid)

    def forward(self, x):
        low = self.low(x)
        mid = self.mid(x) - low
        high = x - self.mid(x)
        return low, mid, high

class CrossBandAttention(nn.Module):
    """Cross-band attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(3, 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, low, mid, high):
        s = torch.stack([low.mean(2), mid.mean(2), high.mean(2)], dim=-1)
        w = self.softmax(self.fc(s))
        return w[..., 0:1] * low + w[..., 1:2] * mid + w[..., 2:3] * high

class BlurPool1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        k = (k / k.sum()).view(1, 1, -1)
        self.register_buffer('weight', k.repeat(channels, 1, 1))
        self.channels = channels
    def forward(self, x):
        return F.conv1d(x, self.weight, stride=1, padding=2, groups=self.channels)

class Stage(nn.Module):
    """
    Configurable stage.
    params:
        block_config: str or dict
            str: use the same selected block for all bands, e.g. 'mspc'
            dict: use per-band selected blocks, e.g. {'low': 'mspc', 'mid': 'sbm', 'high': 'mspc'}
    """
    def __init__(
        self,
        channels,
        downsample,
        block_config,
        band_depths: dict[str, int] | None = None,
        drop_path=0.0,
        mspc_block_kwargs: dict | None = None,
    ):
        super().__init__()
        self.split = BandSplit(channels)
        
        # Resolve block configuration.
        if isinstance(block_config, str):
            cfg = {'low': block_config, 'mid': block_config, 'high': block_config}
        else:
            cfg = block_config
            
        # Default depth is one block per band; per-band depths are supported.
        band_depths = band_depths or {}
        d_low = int(band_depths.get("low", 1))
        d_mid = int(band_depths.get("mid", 1))
        d_high = int(band_depths.get("high", 1))
        if d_low < 0 or d_mid < 0 or d_high < 0:
            raise ValueError("band_depths for low/mid/high must be >= 0")

        def _make_stack(d: int, t: str):
            if d == 0:
                return nn.Identity()
            return nn.Sequential(
                *[UniversalBandProcessor(channels, t, drop_path, mspc_block_kwargs=mspc_block_kwargs) for _ in range(d)]
            )

        self.proc_low = _make_stack(d_low, cfg.get('low', 'sbm'))
        self.proc_mid = _make_stack(d_mid, cfg.get('mid', 'sbm'))
        self.proc_high = _make_stack(d_high, cfg.get('high', 'sbm'))
        
        self.cross = CrossBandAttention(channels)
        
        # Sandglass MLP
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.drop = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        
        # Downsample
        self.down = nn.Sequential(
            BlurPool1d(channels),
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(channels, channels * 2, 1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.GELU()
        ) if downsample else nn.Identity()
        self.out_ch = channels * 2 if downsample else channels

    def forward(self, x):
        low, mid, high = self.split(x)
        
        low = self.proc_low(low)
        mid = self.proc_mid(mid)
        high = self.proc_high(high)
        
        y = self.cross(low, mid, high)
        y = F.gelu(self.drop(self.mlp(y)) + y) # stage-level residual
        
        return self.down(y)

# =========================================================================
#  4. Main Framework
# =========================================================================

class SBM_Universal_Framework(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spectrum_size: int,
        # Core block configuration: a string or per-band dictionary.
        block_type: str | dict = "mspc",
        # Stem configuration: a string or per-band dictionary.
        stem_type: str | dict = "mspc_legacy",
        # If true, split bands before stem processing.
        preband_split: bool = True,
        # Number of blocks per stage, e.g. (4,4,2,4).
        stage_depths: int | tuple[int, int, int, int] | list[int] = (2, 2, 2, 1),
        # Optional per-band stage depths, e.g. {'low':(4,4,2,4), 'high':(2,2,1,2)}.
        band_stage_depths: dict | None = None,
        base_channels: int = 48,
        drop_path: float = 0.1,
        # MSPC stem kwargs, used when stem_type contains 'mspc'.
        mspc_stem_kwargs: dict | None = None,
        # MSPC block kwargs, used when block_type contains 'mspc'.
        mspc_block_kwargs: dict | None = None,
        # Band-split kernel sizes.
        k_low: int = 65,
        k_mid: int = 17,
        # Optional lightweight global mixer:
        # - 'none': disabled
        # - 'transformer': insert 1..N MHSA layers after a later stage
        global_mixer: str = "none",
        global_mixer_depth: int = 1,
        global_attn_heads: int = 4,
        global_pos_encoding: str = "none",   # 'none'|'sincos'|'learned'
        global_learned_pos_max_len: int = 512,
        # Global mixer insertion point:
        # - 'stage4': apply after stage4
        # - 'stage3': apply after stage3 and before stage4
        global_mixer_insert_after: str = "stage4",
    ):
        super().__init__()

        stage_depths_n = _normalize_stage_depths(stage_depths, n_stages=4)
        band_stage_depths_n = _normalize_per_band_stage_depths(band_stage_depths, n_stages=4)

        # 1) Optional pre-band split + per-band stem.
        self.preband_split = bool(preband_split)
        if self.preband_split:
            self.pre_split = BandSplit(in_channel, k_low=k_low, k_mid=k_mid)

            if isinstance(stem_type, str):
                stem_cfg = {"low": stem_type, "mid": stem_type, "high": stem_type}
            else:
                stem_cfg = stem_type
                stem_cfg = {
                    "low": stem_cfg.get("low", "mspc_legacy"),
                    "mid": stem_cfg.get("mid", "mspc_legacy"),
                    "high": stem_cfg.get("high", "mspc_legacy"),
                }

            self.stem_low = UniversalStemProcessor(
                in_channel, base_channels, stem_cfg["low"], mspc_stem_kwargs=mspc_stem_kwargs
            )
            self.stem_mid = UniversalStemProcessor(
                in_channel, base_channels, stem_cfg["mid"], mspc_stem_kwargs=mspc_stem_kwargs
            )
            self.stem_high = UniversalStemProcessor(
                in_channel, base_channels, stem_cfg["high"], mspc_stem_kwargs=mspc_stem_kwargs
            )
            self.stem_fuse = CrossBandAttention(base_channels)
        else:
            # 1) Single-path stem for non-preband mode.
            if isinstance(stem_type, str):
                self.stem = UniversalStemProcessor(
                    in_channel, base_channels, stem_type, mspc_stem_kwargs=mspc_stem_kwargs
                )
            else:
                # Per-band dictionaries only apply when preband_split=True.
                self.stem = UniversalStemProcessor(in_channel, base_channels, "mspc_legacy")
        
        # 2. Stages.
        c = base_channels
        def _band_depths_for_stage(si: int):
            if band_stage_depths_n is None:
                d = stage_depths_n[si]
                return {"low": d, "mid": d, "high": d}
            # Prefer per-band depth; otherwise use shared stage_depths.
            d_default = stage_depths_n[si]
            return {
                "low": (band_stage_depths_n.get("low", stage_depths_n)[si] if band_stage_depths_n.get("low") else d_default),
                "mid": (band_stage_depths_n.get("mid", stage_depths_n)[si] if band_stage_depths_n.get("mid") else d_default),
                "high": (band_stage_depths_n.get("high", stage_depths_n)[si] if band_stage_depths_n.get("high") else d_default),
            }

        self.stage1 = Stage(
            c, True, block_type, band_depths=_band_depths_for_stage(0), drop_path=drop_path, mspc_block_kwargs=mspc_block_kwargs
        ); c *= 2
        self.stage2 = Stage(
            c, True, block_type, band_depths=_band_depths_for_stage(1), drop_path=drop_path, mspc_block_kwargs=mspc_block_kwargs
        ); c *= 2
        self.stage3 = Stage(
            c, True, block_type, band_depths=_band_depths_for_stage(2), drop_path=drop_path, mspc_block_kwargs=mspc_block_kwargs
        ); c *= 2
        self.stage4 = Stage(
            c, False, block_type, band_depths=_band_depths_for_stage(3), drop_path=drop_path, mspc_block_kwargs=mspc_block_kwargs
        )

        gm = str(global_mixer or "none").lower()
        insert_after = str(global_mixer_insert_after or "stage4").lower()
        if insert_after not in ["stage3", "stage4"]:
            raise ValueError("global_mixer_insert_after must be 'stage3' or 'stage4'")
        self.global_mixer_insert_after = insert_after
        if gm in ["none", "off", "no"]:
            self.global_mixer = nn.Identity()
        elif gm in ["transformer", "attn", "mhsa"]:
            depth = int(global_mixer_depth)
            assert depth >= 1, "global_mixer_depth must be >= 1"
            blocks: List[nn.Module] = []
            for _ in range(depth):
                blocks.append(
                    TransformerMixer1D(
                        channels=self.stage4.out_ch,
                        num_heads=int(global_attn_heads),
                        mlp_ratio=2.0,
                        attn_dropout=0.0,
                        proj_dropout=0.0,
                        drop_path=float(drop_path) * 0.2,  # keep it light vs backbone
                        pos_encoding=global_pos_encoding,
                        learned_pos_max_len=int(global_learned_pos_max_len),
                    )
                )
            self.global_mixer = nn.Sequential(*blocks)
        else:
            raise ValueError("global_mixer must be 'none' or 'transformer'")
        
        # 3. Prediction head.
        self.attnpool = nn.Conv1d(self.stage4.out_ch, 1, 3, padding=1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(self.stage4.out_ch, self.stage4.out_ch // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.stage4.out_ch // 2, out_channel)
        )

    def forward_features(self, x):
        if self.preband_split:
            low, mid, high = self.pre_split(x)
            low = self.stem_low(low)
            mid = self.stem_mid(mid)
            high = self.stem_high(high)
            x = self.stem_fuse(low, mid, high)
        else:
            x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        if self.global_mixer_insert_after == "stage3":
            x = self.global_mixer(x)
        x = self.stage4(x)
        if self.global_mixer_insert_after == "stage4":
            x = self.global_mixer(x)
        
        # Attn Pool Logic
        score = torch.softmax(self.attnpool(x), dim=-1)
        return (x * score).sum(dim=-1)

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat)

# =========================================================================
#  Smoke test
# =========================================================================
if __name__ == "__main__":
    x = torch.randn(2, 1, 3000)

    print("--- MSPC-based SBM v2 smoke test ---")
    model_mspc = SBM_Universal_Framework(
        1,
        4,
        3000,
        global_mixer="transformer",
        global_pos_encoding="sincos",
    )
    print(f"MSPC Output: {model_mspc(x).shape}") # Should be [2, 4]

    print("\nModel smoke test passed.")
