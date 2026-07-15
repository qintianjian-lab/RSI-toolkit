#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Gandalf dataset directory (X.npy, params.npy, axis_labels.npy, ids.npy)
from your classifier directory layout, with optional class rebalancing.

Input layout:
  <fold_root>/<split>/{spectrum,label}
    - spectrum/*.csv  (two columns: wavelength,flux)
    - label/label.csv (at least: basename,label[,z])

Example:
  # Keep all samples (no rebalance)
  python prepare_from_dir.py \
    --fold_root data/lamost_folds/fold_1 \
    --split train \
    --length 4000 \
    --out_dir "train_npy"

  # Rebalance: keep ALL positives (label==1), sample negatives to 1:3
  python prepare_from_dir.py \
    --fold_root data/lamost_folds/fold_1 \
    --split train \
    --length 4000 \
    --out_dir "train_npy_v2" \
    --label_col label --positive_label 1 --neg_pos_ratio 1.0 --seed 42
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Gandalf dataset (X.npy, params.npy, axis_labels.npy, ids.npy)")
    p.add_argument("--fold_root", type=str, required=True, help="Path to fold directory (contains <split>/spectrum,label)")
    p.add_argument("--split", type=str, default="train", help="Split to convert (train|val|test)")
    p.add_argument("--length", type=int, default=4000, help="Target length L (pad/truncate)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory to write the four .npy files")
    p.add_argument("--label_col", type=str, default="label", help="Column name in label.csv to use as params (condition)")
    # Rebalance options
    p.add_argument("--positive_label", type=str, default=None, help="Value treated as positive class (e.g., '1'). If set, enables rebalancing.")
    p.add_argument("--neg_pos_ratio", type=float, default=3.0, help="Target negatives per positive (keeps ALL positives, samples negatives to ratio)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return p.parse_args()


def load_flux_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"Expected (wavelength,flux) in {path}")
    wl = arr[:, 0]
    fx = arr[:, 1]
    return wl, fx


def pad_or_truncate(x: np.ndarray, L: int) -> np.ndarray:
    if x.size >= L:
        return x[:L]
    y = np.zeros((L,), dtype=x.dtype)
    y[: x.size] = x
    return y


def main() -> None:
    args = parse_args()
    split_dir = os.path.join(args.fold_root, args.split)
    spec_dir = os.path.join(split_dir, "spectrum")
    lab_csv = os.path.join(split_dir, "label", "label.csv")
    if not os.path.isdir(spec_dir):
        raise FileNotFoundError(f"spectrum dir not found: {spec_dir}")
    if not os.path.isfile(lab_csv):
        raise FileNotFoundError(f"label csv not found: {lab_csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(lab_csv)
    if "basename" not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"label.csv must contain columns: basename,{args.label_col}")

    # Optional rebalance: keep ALL positives, sample negatives to a target ratio
    if args.positive_label is not None:
        pos_mask = df[args.label_col].astype(str) == str(args.positive_label)
        pos_df = df[pos_mask].copy()
        neg_df = df[~pos_mask].copy()
        n_pos = len(pos_df)
        n_neg_avail = len(neg_df)
        # target negatives
        n_neg_target = int(round(max(0.0, float(args.neg_pos_ratio)) * n_pos))
        n_neg_target = min(n_neg_target, n_neg_avail)
        if n_neg_target < n_neg_avail:
            rng = np.random.default_rng(args.seed)
            sampled_idx = rng.choice(neg_df.index.values, size=n_neg_target, replace=False)
            neg_df = neg_df.loc[sampled_idx]
        df = pd.concat([pos_df, neg_df], axis=0).reset_index(drop=True)
        # shuffle
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"[rebalance] positives={n_pos}, negatives_selected={len(neg_df)}, ratio ~ 1:{(len(neg_df)/max(1,n_pos)):.2f}")

    X_list: List[np.ndarray] = []
    P_list: List[np.ndarray] = []
    ids: List[str] = []
    wl_ref: np.ndarray | None = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"build {args.split}"):
        base = str(row["basename"])
        path = os.path.join(spec_dir, f"{base}.csv")
        if not os.path.isfile(path):
            continue
        try:
            wl, fx = load_flux_csv(path)
            # preserve order ascending
            if wl.size >= 2 and wl[0] > wl[-1]:
                wl = wl[::-1].copy()
                fx = fx[::-1].copy()
            fxL = pad_or_truncate(fx.astype(np.float64), args.length)
            X_list.append(fxL)
            P_list.append(np.array([float(row[args.label_col])], dtype=np.float64))
            ids.append(base)
            if wl_ref is None:
                wl_ref = pad_or_truncate(wl.astype(np.float64), args.length)
        except Exception:
            continue

    if not X_list:
        raise RuntimeError("No valid spectra found.")
    if wl_ref is None:
        wl_ref = np.arange(args.length, dtype=np.float64)

    X = np.stack(X_list, axis=0).astype(np.float64)
    params = np.stack(P_list, axis=0).astype(np.float64)  # (N,1)
    axis_labels = wl_ref.astype(np.float64)
    # Save ids with a non-object string dtype to be compatible with Gandalf loader (allow_pickle=False)
    ids_np = np.array(ids, dtype=np.str_)

    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "params.npy"), params)
    np.save(os.path.join(args.out_dir, "axis_labels.npy"), axis_labels)
    np.save(os.path.join(args.out_dir, "ids.npy"), ids_np)
    print(f"[save] X.npy         : {X.shape}")
    print(f"[save] params.npy    : {params.shape}")
    print(f"[save] axis_labels.npy: {axis_labels.shape}")
    print(f"[save] ids.npy       : {ids_np.shape}")


if __name__ == "__main__":
    main()

