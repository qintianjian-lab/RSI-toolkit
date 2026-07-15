#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build CVAE-ready npy from your classifier directory layout (train split ONLY).
- No normalization.
- Only zero-pad/truncate to fixed length L.
- Export features and conditions and split internally into train/val:
  out_dir/X_train.npy, out_dir/X_val.npy        # (N, L)
  out_dir/C_train.npy, out_dir/C_val.npy        # (N, 1) numeric labels as conditions
  out_dir/wavelengths.npy                       # (L,) picked from train

Expected layout:
  <root>/train/{spectrum,label}
    - spectrum/*.csv  (two columns: wavelength,flux)
    - label/label.csv (at least: basename,label[,z])

Example:
  python dir_to_npy_cvae.py \
    --root data/lamost_folds/fold_1 \
    --length 4000 \
    --out_dir "cvae_npy" \
    --val_ratio 0.1

  # Rebalance: keep ALL positives (label==1), sample negatives to 1:3
  python dir_to_npy_cvae.py \
    --root data/lamost_folds/fold_1 \
    --length 4000 \
    --out_dir "cvae_npy_v2" \
    --val_ratio 0.1 \
    --positive_label 1 --neg_pos_ratio 1.0 --seed 42
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert classifier dir (train split) to CVAE npy (X/C + wavelengths)")
    p.add_argument("--root", type=str, required=True, help="Path to fold directory containing train/")
    p.add_argument("--length", type=int, required=True, help="Target spectrum length L")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for npy files")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio from train")
    # Rebalance options
    p.add_argument("--positive_label", type=str, default=None, help="Value treated as positive (e.g., '1'). If set, enables rebalancing.")
    p.add_argument("--neg_pos_ratio", type=float, default=3.0, help="Target negatives per positive; keeps ALL positives.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling/shuffle")
    return p.parse_args()


def load_flux_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"Expected at least two columns (wavelength,flux) in {path}")
    wl = arr[:, 0]
    fx = arr[:, 1]
    return wl, fx


def pad_or_truncate(x: np.ndarray, L: int) -> np.ndarray:
    if x.size >= L:
        return x[:L].astype(np.float32)
    y = np.zeros((L,), dtype=np.float32)
    y[: x.size] = x.astype(np.float32)
    return y


def build_from_train(root: str, L: int, positive_label: str | None, neg_pos_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    split = "train"
    lab_csv = os.path.join(root, split, "label", "label.csv")
    spec_dir = os.path.join(root, split, "spectrum")
    if not os.path.isfile(lab_csv):
        raise FileNotFoundError(f"label csv not found: {lab_csv}")
    if not os.path.isdir(spec_dir):
        raise FileNotFoundError(f"spectrum dir not found: {spec_dir}")
    df = pd.read_csv(lab_csv)
    if "basename" not in df.columns or "label" not in df.columns:
        raise ValueError("label.csv must contain columns: basename,label")

    # Optional rebalance
    if positive_label is not None:
        pos_mask = df["label"].astype(str) == str(positive_label)
        pos_df = df[pos_mask].copy()
        neg_df = df[~pos_mask].copy()
        n_pos = len(pos_df)
        n_neg_avail = len(neg_df)
        n_neg_target = int(round(max(0.0, float(neg_pos_ratio)) * n_pos))
        n_neg_target = min(n_neg_target, n_neg_avail)
        if n_neg_target < n_neg_avail:
            rng = np.random.default_rng(seed)
            sampled_idx = rng.choice(neg_df.index.values, size=n_neg_target, replace=False)
            neg_df = neg_df.loc[sampled_idx]
        df = pd.concat([pos_df, neg_df], axis=0).reset_index(drop=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"[rebalance] positives={n_pos}, negatives_selected={len(neg_df)}, ratio ~ 1:{(len(neg_df)/max(1,n_pos)):.2f}")

    X_list: List[np.ndarray] = []
    C_list: List[np.ndarray] = []
    bases: List[str] = []
    wl_ref: np.ndarray | None = None
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        base = str(row["basename"])
        path = os.path.join(spec_dir, f"{base}.csv")
        if not os.path.isfile(path):
            continue
        try:
            wl, fx = load_flux_csv(path)
            fxL = pad_or_truncate(fx, L)
            lbl = float(row["label"])
            X_list.append(fxL)
            C_list.append(np.array([lbl], dtype=np.float32))
            bases.append(base)
            if wl_ref is None:
                wl_ref = pad_or_truncate(wl, L)
        except Exception:
            continue
    if not X_list:
        raise RuntimeError(f"No valid samples for split={split}")
    X = np.stack(X_list, axis=0).astype(np.float32)
    C = np.stack(C_list, axis=0).astype(np.float32)  # (N,1)
    if wl_ref is None:
        wl_ref = np.arange(L, dtype=np.float32)
    return X, C, bases, wl_ref


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build from train, then split into train/val
    X, C, _, wl = build_from_train(args.root, args.length, args.positive_label, args.neg_pos_ratio, args.seed)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(X.shape[0])
    n_val = max(1, int(round(args.val_ratio * X.shape[0])))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr, X_va = X[tr_idx], X[val_idx]
    C_tr, C_va = C[tr_idx], C[val_idx]

    np.save(os.path.join(args.out_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(args.out_dir, "X_val.npy"),   X_va)
    np.save(os.path.join(args.out_dir, "C_train.npy"), C_tr)
    np.save(os.path.join(args.out_dir, "C_val.npy"),   C_va)
    np.save(os.path.join(args.out_dir, "wavelengths.npy"), wl.astype(np.float32))
    print(f"[save] X_train: {X_tr.shape}")
    print(f"[save] X_val  : {X_va.shape}")
    print(f"[save] C_train: {C_tr.shape}")
    print(f"[save] C_val  : {C_va.shape}")
    print(f"[save] wavelengths: {wl.shape}")


if __name__ == "__main__":
    main()

