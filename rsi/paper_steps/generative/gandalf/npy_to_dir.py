#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write generated spectra back to classifier directory:
  - spectrum/synth-XXXXXX.csv (wavelength,flux)
  - append rows to label/label.csv with label=1

Optional: linearly resample before writing
  - to a target wavelength axis (--interp_to)
  - or to a target length with same min/max as source (--interp_to_len)

Usage:
  python npy_to_dir.py \
    --generated runs/generative/gandalf/generated_psb.npy \
    --wavelengths runs/generative/gandalf/axis_labels.npy \
    --target_split_dir data/lamost_folds/fold_1/train \
    --prefix "synth-ea" \
    --label_value 1 \
    --interp_to runs/generative/reference_wavelengths.npy
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Insert generated spectra into classifier dir structure")
    p.add_argument("--generated", type=str, required=True, help="Generated npy (K, L)")
    p.add_argument("--wavelengths", type=str, required=True, help="Reference wavelengths npy (L,)")
    p.add_argument("--target_split_dir", type=str, required=True, help="Path to <fold>/<split> (contains spectrum,label)")
    p.add_argument("--prefix", type=str, default="synth-ea", help="Basename prefix for synthetic spectra")
    p.add_argument("--label_value", type=str, default="1", help="Label value to append for generated rows")
    p.add_argument("--interp_to", type=str, default=None, help="Optional target wavelength axis npy (L_tgt,) to linearly resample to")
    p.add_argument("--interp_to_len", type=int, default=None, help="Optional target length; builds linspace between src wl[0]..wl[-1]")
    return p.parse_args()


def write_spectrum_csv(wl: np.ndarray, fx: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.column_stack([wl.astype(np.float64), fx.astype(np.float64)])
    np.savetxt(path, arr, delimiter=",", fmt="%.6f")


def main() -> None:
    args = parse_args()
    X_src = np.load(args.generated).astype(np.float32)  # (K, L_src)
    wl_src = np.load(args.wavelengths).astype(np.float32)  # (L_src,)
    assert X_src.ndim == 2 and wl_src.ndim == 1 and X_src.shape[1] == wl_src.shape[0], "shape mismatch"

    # Decide whether to resample
    wl_out = wl_src
    X_out = X_src
    if args.interp_to is not None:
        wl_tgt = np.load(args.interp_to).astype(np.float32)
        assert wl_tgt.ndim == 1 and wl_tgt.size > 1, "--interp_to must be a 1D wavelength array"
        X_out = np.stack([
            np.interp(wl_tgt, wl_src, x, left=x[0], right=x[-1]).astype(np.float32)
        for x in X_src], axis=0)
        wl_out = wl_tgt
    elif args.interp_to_len is not None:
        L = int(args.interp_to_len)
        assert L >= 2, "--interp_to_len must be >= 2"
        wl_tgt = np.linspace(float(wl_src[0]), float(wl_src[-1]), L, dtype=np.float32)
        X_out = np.stack([
            np.interp(wl_tgt, wl_src, x, left=x[0], right=x[-1]).astype(np.float32)
        for x in X_src], axis=0)
        wl_out = wl_tgt

    spec_dir = os.path.join(args.target_split_dir, "spectrum")
    lab_csv = os.path.join(args.target_split_dir, "label", "label.csv")
    if not os.path.isdir(spec_dir):
        raise FileNotFoundError(f"spectrum dir not found: {spec_dir}")
    os.makedirs(os.path.dirname(lab_csv), exist_ok=True)

    # Load existing label.csv if exists
    if os.path.isfile(lab_csv):
        df = pd.read_csv(lab_csv)
        if "basename" not in df.columns or "label" not in df.columns:
            raise ValueError("label.csv must have columns: basename,label")
    else:
        df = pd.DataFrame(columns=["basename", "label"])

    # Find next index to avoid collision
    existing = set()
    if "basename" in df.columns:
        existing.update(df["basename"].astype(str).tolist())

    new_rows = []
    count = 0
    for i in tqdm(range(X_out.shape[0]), desc="writing"):
        k = 0
        while True:
            base = f"{args.prefix}-{i:06d}" if k == 0 else f"{args.prefix}-{i:06d}-{k}"
            if base not in existing and not os.path.exists(os.path.join(spec_dir, f"{base}.csv")):
                break
            k += 1
        path = os.path.join(spec_dir, f"{base}.csv")
        write_spectrum_csv(wl_out, X_out[i], path)
        new_rows.append({"basename": base, "label": args.label_value})
        existing.add(base)
        count += 1

    # Append to label.csv
    df_new = pd.DataFrame(new_rows)
    if df.empty:
        df_out = df_new
    else:
        df_out = pd.concat([df, df_new], ignore_index=True)
    df_out.to_csv(lab_csv, index=False)
    print(f"[save] wrote {count} spectra to {spec_dir}")
    print(f"[save] updated labels: {lab_csv}")


if __name__ == "__main__":
    main()

