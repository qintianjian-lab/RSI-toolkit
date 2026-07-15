from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.sbm_universal_mae import SBMUniversalMAE  # noqa: E402


def _load_py_config(path: str) -> dict[str, Any]:
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"config file not found: {path}")
    mod_name = f"_cfg_{abs(hash(p))}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load config module from: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    cfg = mod.get_config() if hasattr(mod, "get_config") and callable(getattr(mod, "get_config")) else getattr(mod, "CFG", None)
    if not isinstance(cfg, dict):
        raise ValueError("config must define dict `CFG` or function `get_config()`")
    return cfg


class CachedSpectrumDataset(Dataset):
    """
    Read preprocessed arrays from index csv:
    required columns: basename, npy_path
    """

    def __init__(self, index_csv: str, spectrum_size: int = 4000):
        self.df = pd.read_csv(index_csv)
        if "npy_path" not in self.df.columns:
            raise ValueError(f"index csv must contain 'npy_path': {index_csv}")
        self.spectrum_size = int(spectrum_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = str(row["npy_path"]).strip()
        x = np.load(p).astype(np.float32, copy=False).reshape(-1)
        L = x.shape[0]
        if L != self.spectrum_size:
            if L > self.spectrum_size:
                s = (L - self.spectrum_size) // 2
                x = x[s:s + self.spectrum_size]
            else:
                pad = self.spectrum_size - L
                left = pad // 2
                x = np.pad(x, (left, pad - left), mode="constant", constant_values=0.0)
        return torch.from_numpy(x).float().unsqueeze(0)


class LitUniversalMAE(pl.LightningModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": cfg})
        data = cfg["data"]
        model = cfg["model"]
        loss = cfg["loss"]
        mask = cfg["mask"]
        phys = cfg.get("physics_mask", {})
        dist = cfg.get("distill", {})

        self.model = SBMUniversalMAE(
            in_channel=1,
            spectrum_size=int(data["spectrum_size"]),
            block_type=model.get("block_type", "mspc"),
            stem_type=model.get("stem_type", "mspc"),
            preband_split=bool(model.get("preband_split", True)),
            stage_depths=model.get("stage_depths", (2, 2, 2, 1)),
            band_stage_depths=model.get("band_stage_depths", None),
            base_channels=int(model.get("base_channels", 48)),
            drop_path=float(model.get("drop_path", 0.1)),
            mspc_stem_kwargs=model.get("mspc_stem_kwargs", None),
            mspc_block_kwargs=model.get("mspc_block_kwargs", None),
            k_low=int(model.get("k_low", 65)),
            k_mid=int(model.get("k_mid", 17)),
            decoder_hidden=int(model.get("decoder_hidden", 128)),
            decoder_fullres_refine=bool(model.get("decoder_fullres_refine", True)),
        )
        self.lr = float(cfg["train"]["lr"])
        self.recon_loss = str(loss.get("recon_loss", "smoothl1"))
        self.recon_weight = float(loss.get("recon_weight", 1.0))
        self.grad_weight = float(loss.get("grad_weight", 0.0))
        self.grad_mask_only = bool(loss.get("grad_mask_only", True))
        self.mask_ratio = float(mask.get("mask_ratio", 0.6))
        self.mask_block = int(mask.get("mask_block", 30))
        self.mask_fill = str(mask.get("mask_fill", "mean"))
        self.mask_loss_only = bool(mask.get("mask_loss_only", True))
        self.mask_physics = bool(phys.get("enabled", False))
        self.mask_physics_prob = float(phys.get("prob", 0.5))
        self.mask_physics_only = bool(phys.get("only", False))
        self.mask_lines = [float(s.strip()) for s in str(phys.get("lines", "4101.7,4340.5,4861.3")).split(",") if s.strip()]
        self.mask_line_width = float(phys.get("line_width", 20.0))
        self.wl_min = phys.get("wl_min", None)
        self.wl_max = phys.get("wl_max", None)
        self.feat_distill = bool(dist.get("enabled", False))
        self.feat_weight = float(dist.get("weight", 1.0))
        self.feat_ema = float(dist.get("ema", 0.99))
        self.feat_loss = str(dist.get("loss", "mse"))
        self.teacher = None
        if self.feat_distill:
            self.teacher = copy.deepcopy(self.model)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def _wl_to_idx(self, wl: float, L: int) -> int:
        if self.wl_min is None or self.wl_max is None:
            return 0
        t = (float(wl) - float(self.wl_min)) / max(1e-12, float(self.wl_max) - float(self.wl_min))
        t = max(0.0, min(1.0, t))
        return int(round(t * (L - 1)))

    def _apply_mask(self, x: torch.Tensor):
        B, _, L = x.shape
        mask = torch.zeros((B, 1, L), device=x.device, dtype=x.dtype)
        if self.mask_physics and self.wl_min is not None and self.wl_max is not None:
            do = torch.rand((B,), device=x.device) < max(0.0, min(1.0, self.mask_physics_prob))
            for i in range(B):
                if not bool(do[i].item()):
                    continue
                for c in self.mask_lines:
                    lo = self._wl_to_idx(c - self.mask_line_width, L)
                    hi = self._wl_to_idx(c + self.mask_line_width, L)
                    a, b = min(lo, hi), max(lo, hi)
                    mask[i, :, a:b + 1] = 1.0
        if self.mask_ratio > 0.0 and not self.mask_physics_only:
            block = max(1, min(self.mask_block, L))
            r = max(0.0, min(1.0, self.mask_ratio))
            b = float(block) / float(L)
            n = 1 if b >= 1.0 else int(np.ceil(np.log(max(1e-12, 1.0 - r)) / np.log(max(1e-12, 1.0 - b))))
            n = max(1, n)
            for i in range(B):
                for _ in range(n):
                    s = int(torch.randint(0, max(1, L - block), (1,), device=x.device).item())
                    mask[i, :, s:s + block] = 1.0
        if self.mask_fill == "zero":
            x_in = x * (1.0 - mask)
        elif self.mask_fill == "noise":
            x_in = x * (1.0 - mask) + torch.randn_like(x) * 0.02 * mask
        else:
            x_in = x * (1.0 - mask) + x.mean(dim=-1, keepdim=True) * mask
        return x_in, mask

    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        per = F.smooth_l1_loss(recon, target, reduction="none") if self.recon_loss == "smoothl1" else F.mse_loss(recon, target, reduction="none")
        if self.mask_loss_only:
            denom = mask.sum()
            if float(denom.detach().item()) < 1.0:
                return per.mean()
            return (per * mask).sum() / denom
        return per.mean()

    def _grad_loss(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        dr = recon[..., 1:] - recon[..., :-1]
        dx = target[..., 1:] - target[..., :-1]
        per = F.mse_loss(dr, dx, reduction="none")
        if self.grad_mask_only:
            m = mask[..., 1:]
            denom = m.sum()
            if float(denom.detach().item()) < 1.0:
                return per.mean()
            return (per * m).sum() / denom
        return per.mean()

    def _feat_distill_loss(self, pred: torch.Tensor, tgt: torch.Tensor):
        if self.feat_loss == "smoothl1":
            return F.smooth_l1_loss(pred, tgt)
        return F.mse_loss(pred, tgt)

    def training_step(self, batch, batch_idx):
        x = batch
        x_in, mask = self._apply_mask(x)
        out = self.model(x_in)
        recon_term = self._recon_loss(out.recon, x, mask)
        loss = self.recon_weight * recon_term
        if self.grad_weight > 0:
            loss = loss + self.grad_weight * self._grad_loss(out.recon, x, mask)
        if self.feat_distill and self.feat_weight > 0:
            with torch.no_grad():
                t_out = self.teacher(x)
            f_term = self._feat_distill_loss(out.fmap, t_out.fmap)
            loss = loss + self.feat_weight * f_term
            self.log("train_feat_loss", f_term, on_step=True, prog_bar=False, batch_size=x.size(0))
        self.log("train_masked_recon_loss", recon_term, on_step=True, prog_bar=True, batch_size=x.size(0))
        self.log("train_total_loss", loss, on_step=True, prog_bar=True, batch_size=x.size(0))
        self.log("train_mask_covered", mask.mean(), on_step=True, prog_bar=False, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_in, mask = self._apply_mask(x)
        out = self.model(x_in)
        recon_term = self._recon_loss(out.recon, x, mask)
        loss = self.recon_weight * recon_term
        if self.grad_weight > 0:
            loss = loss + self.grad_weight * self._grad_loss(out.recon, x, mask)
        self.log("val_masked_recon_loss", recon_term, on_epoch=True, on_step=False, prog_bar=True, batch_size=x.size(0))
        self.log("val_total_loss", loss, on_epoch=True, on_step=False, prog_bar=False, batch_size=x.size(0))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self.feat_distill or self.teacher is None:
            return
        m = max(0.0, min(0.99999, self.feat_ema))
        with torch.no_grad():
            for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters()):
                p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))
            for b_t, b_s in zip(self.teacher.buffers(), self.model.buffers()):
                try:
                    b_t.data.copy_(b_s.data)
                except Exception:
                    pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)


def main():
    parser = argparse.ArgumentParser(description="Pretrain isolated SBM-Universal masked-reconstruction encoder.")
    parser.add_argument("--config", type=str, default=os.path.join("config", "pretrain_sbm_universal_mae_config.py"))
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = _load_py_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    if args.run_name:
        cfg["logging"]["wandb_name"] = args.run_name

    data = cfg["data"]
    tr = cfg["train"]
    lg = cfg["logging"]

    train_index_csv = str(data.get("cached_train_index_csv", "")).strip()
    val_index_csv = str(data.get("cached_val_index_csv", "")).strip()
    if not train_index_csv or not val_index_csv:
        raise ValueError("data.cached_train_index_csv and data.cached_val_index_csv are required")
    train_ds = CachedSpectrumDataset(
        index_csv=train_index_csv,
        spectrum_size=int(data.get("spectrum_size", 4000)),
    )
    val_ds = CachedSpectrumDataset(
        index_csv=val_index_csv,
        spectrum_size=int(data.get("spectrum_size", 4000)),
    )
    print(f"[Data] Cached mode only. train={len(train_ds)} val={len(val_ds)}")

    device_str = str(tr.get("device", "cuda:0"))
    use_cuda = torch.cuda.is_available() and ("cuda" in device_str)
    pin_mem = bool(use_cuda)
    train_loader = DataLoader(train_ds, batch_size=int(tr.get("batch_size", 128)), shuffle=True, num_workers=int(tr.get("num_workers", 4)), pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=int(tr.get("batch_size", 128)), shuffle=False, num_workers=int(tr.get("num_workers", 4)), pin_memory=pin_mem)

    lit = LitUniversalMAE(cfg=cfg)
    run_name = str(lg.get("wandb_name", "")).strip() or "sbm_universal_mae_isolated"
    tb = TensorBoardLogger(save_dir=str(tr.get("log_dir", "./logs_umae")), name=run_name)
    loggers = [tb]
    if bool(lg.get("enable_wandb", False)):
        loggers.append(WandbLogger(project=str(lg.get("wandb_project", "SBM_Universal_MAE")), save_dir=str(tr.get("log_dir", "./logs_umae")), name=run_name))

    ckpt = ModelCheckpoint(
        dirpath=os.path.join(str(tr.get("ckpt_dir", "./checkpoints_umae")), run_name),
        filename="best-{epoch}-{val_masked_recon_loss:.6f}",
        monitor="val_masked_recon_loss",
        mode="min",
        save_top_k=3,
        save_weights_only=False,
    )
    lr_mon = LearningRateMonitor(logging_interval="step")
    devices = [int(device_str.split(":")[-1])] if use_cuda and ":" in device_str else (1 if use_cuda else None)
    precision = str(tr.get("precision", "32-true"))
    if use_cuda and precision.startswith("bf16") and not torch.cuda.is_bf16_supported():
        print(f"[Warn] precision={precision} not supported on this GPU. Falling back to 16-mixed.")
        precision = "16-mixed"

    trainer = pl.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
        devices=devices,
        max_epochs=int(tr.get("epochs", 60)),
        precision=precision,
        logger=loggers,
        callbacks=[ckpt, lr_mon],
        log_every_n_steps=10,
        enable_progress_bar=True,
        val_check_interval=float(tr.get("val_check_interval", 1.0)),
    )
    trainer.fit(lit, train_loader, val_loader)
    print(f"[Info] best_ckpt={ckpt.best_model_path}")


if __name__ == "__main__":
    main()
