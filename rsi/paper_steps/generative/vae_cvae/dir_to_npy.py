#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build train/val npy ONLY from the 'train' split of your classifier directory.
No normalization. Only zero-pad/truncate to fixed length L. Filter to label==1.
It shuffles the positive samples in train and splits internally into
  - out_dir/train.npy
  - out_dir/val.npy
Also writes a reference wavelengths.npy.

Expected layout:
  <root>/train/{spectrum,label}
    - spectrum/*.csv  (two columns: wavelength,flux)
    - label/label.csv (columns at least: basename,label[,z])

Usage:
  python dir_to_npy.py \
    --root data/lamost_folds/fold_1 \
    --length 4000 \
    --out_dir "vae_npy" \
    --label_equals 1 \
    --val_ratio 0.1
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert classifier dir (train split) to VAE npy (no normalization)")
    p.add_argument("--root", type=str, required=True, help="Path to fold directory containing train/")
    p.add_argument("--length", type=int, required=True, help="Target spectrum length L (pad/truncate)")
    p.add_argument("--out_dir", type=str, required=True, help="Output dir for npy files")
    p.add_argument("--label_equals", type=str, default="1", help="Keep rows where label==this value")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Fraction for validation split (from train positives)")
    return p.parse_args()


def load_flux_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # robust read for simple two-column CSV (wavelength,flux)
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    wl = arr[:, 0]
    fx = arr[:, 1]
    return wl, fx


def pad_or_truncate(x: np.ndarray, L: int) -> np.ndarray:
    if x.size >= L:
        return x[:L].astype(np.float32)
    y = np.zeros((L,), dtype=np.float32)
    y[: x.size] = x.astype(np.float32)
    return y


def build_from_train(root: str, L: int, label_equals: str) -> Tuple[np.ndarray, List[str], List[float], np.ndarray]:
    split = "train"
    lab_path = os.path.join(root, split, "label", "label.csv")
    spec_dir = os.path.join(root, split, "spectrum")
    if not os.path.isfile(lab_path):
        raise FileNotFoundError(f"label csv not found: {lab_path}")
    if not os.path.isdir(spec_dir):
        raise FileNotFoundError(f"spectrum dir not found: {spec_dir}")
    df = pd.read_csv(lab_path)
    if "basename" not in df.columns or "label" not in df.columns:
        raise ValueError("label.csv must have columns: basename,label")
    df = df[df["label"].astype(str) == str(label_equals)].copy()
    X_list: List[np.ndarray] = []
    bases: List[str] = []
    zs: List[float] = []
    wl_ref: np.ndarray | None = None
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
        base = str(row["basename"])
        path = os.path.join(spec_dir, f"{base}.csv")
        if not os.path.isfile(path):
            continue
        try:
            wl, fx = load_flux_csv(path)
            fxL = pad_or_truncate(fx, L)
            X_list.append(fxL)
            bases.append(base)
            z = float(row["z"]) if "z" in df.columns else np.nan
            zs.append(z)
            if wl_ref is None:
                wl_ref = pad_or_truncate(wl, L)
        except Exception:
            continue
    if not X_list:
        raise RuntimeError(f"No samples built for split={split}")
    X = np.stack(X_list, axis=0).astype(np.float32)
    if wl_ref is None:
        wl_ref = np.arange(L, dtype=np.float32)
    return X, bases, zs, wl_ref


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # build from train only
    X, bases, zs, wl = build_from_train(args.root, args.length, args.label_equals)
    # shuffle and split into train/val
    rng = np.random.default_rng(42)
    idx = rng.permutation(X.shape[0])
    n_val = max(1, int(round(args.val_ratio * X.shape[0])))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr = X[tr_idx]
    X_va = X[val_idx]
    np.save(os.path.join(args.out_dir, "train.npy"), X_tr)
    np.save(os.path.join(args.out_dir, "val.npy"),   X_va)
    np.save(os.path.join(args.out_dir, "wavelengths.npy"), wl.astype(np.float32))
    print(f"[save] train: {X_tr.shape} -> {os.path.join(args.out_dir,'train.npy')}")
    print(f"[save] val  : {X_va.shape} -> {os.path.join(args.out_dir,'val.npy')}")
    print(f"[save] wavelengths: {wl.shape} -> {os.path.join(args.out_dir,'wavelengths.npy')}")


if __name__ == "__main__":
    main()

