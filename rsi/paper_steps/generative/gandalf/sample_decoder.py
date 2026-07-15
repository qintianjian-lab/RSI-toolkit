#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample spectra from a trained Gandalf decoder.

Assumes a training run exists under results/ with:
- results/models/<model_id>/autoencoder.keras (decoder is a sublayer)
- results/data/<model_id>/{param_scaler.pkl,X_scaler.pkl}

Example:
  python gandalf/sample_decoder.py \
    --root "gandalf" \
    --model_id auto \
    --latent 25 --n_class 1 --num 200 --cond_value 1.0 \
    --out_npy "generated_gandalf.npy" \
    --inverse_normalize

CUDA_VISIBLE_DEVICES="" python sample_decoder.py \
  --root gandalf \
  --model_id auto \
  --latent 25 \
  --n_class 1 \
  --num 2100 \
  --cond_value 1.0 \
  --out_npy output/generated_gandalf.npy

eg
python gandalf/sample_decoder.py \
  --root gandalf --model_id auto \
  --latent 25 --n_class 1 \
  --num 2000 --cond_value 1.0 \
  --z_mode empirical \
  --out_npy output/generated_gandalf.npy \
  --inverse_normalize
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import tensorflow as tf
from pickle import load
import glob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate samples with Gandalf decoder")
    p.add_argument("--root", type=str, default="gandalf", help="Root folder passed to train_cli.py (--root_folder)")
    p.add_argument("--model_id", type=str, default="auto", help="'auto' to pick the latest results/models/<model_id>")
    p.add_argument("--latent", type=int, required=True, help="Latent size used during training")
    p.add_argument("--n_class", type=int, default=1, help="Number of conditional parameters")
    p.add_argument("--num", type=int, default=64, help="Number of samples to generate")
    p.add_argument("--cond_value", type=float, default=1.0, help="Condition value to generate (e.g., 1.0 for E+A)")
    p.add_argument("--z_mode", type=str, default="empirical", choices=["empirical", "uniform", "normal"],
                   help="Latent sampling: empirical(from z_train), uniform(0,1) or normal(0,1)")
    p.add_argument("--out_npy", type=str, required=True, help="Output npy path")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--inverse_normalize", action="store_true", help="Apply inverse transform if training used --normalize")
    return p.parse_args()


def pick_latest_model_id(root: str) -> str:
    models_dir = os.path.join(root, "results", "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No models dir: {models_dir}")
    ids = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not ids:
        raise RuntimeError("No trained models found.")
    ids.sort()
    return ids[-1]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model_id = pick_latest_model_id(args.root) if args.model_id == "auto" else args.model_id
    model_dir = os.path.join(args.root, "results", "models", model_id)
    data_dir  = os.path.join(args.root, "results", "data", model_id)

    # Prefer loading the autoencoder and extracting the 'decoder' submodel (this is how training saves models)
    ae_paths = glob.glob(os.path.join(model_dir, "autoencoder*"))
    decoder = None
    if ae_paths:
        autoencoder = tf.keras.models.load_model(ae_paths[0], compile=False)
        try:
            decoder = autoencoder.get_layer("decoder")
        except Exception:
            pass
    # Fallback: try to load a standalone decoder if present
    if decoder is None:
        dec_paths = glob.glob(os.path.join(model_dir, "*decoder*"))
        if not dec_paths:
            raise FileNotFoundError(f"decoder not found under: {model_dir}")
        decoder = tf.keras.models.load_model(dec_paths[0], compile=False)

    # Sample latent z according to selected mode (Plan A default: empirical)
    if args.z_mode == "empirical":
        z_pool_path = os.path.join(data_dir, "z_train.npy")
        if not os.path.isfile(z_pool_path):
            raise FileNotFoundError(f"z_train not found: {z_pool_path}. Train first to create it.")
        z_pool = np.load(z_pool_path, allow_pickle=True)
        if z_pool.ndim != 2 or z_pool.shape[1] != args.latent:
            raise ValueError(f"Latent size mismatch: z_train shape {z_pool.shape}, expected latent {args.latent}")
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(z_pool.shape[0], size=args.num, replace=True)
        z = z_pool[sel].astype(np.float64)
    elif args.z_mode == "uniform":
        z = np.random.uniform(0.0, 1.0, size=(args.num, args.latent)).astype(np.float64)
    else:  # normal
        z = np.random.normal(0.0, 1.0, size=(args.num, args.latent)).astype(np.float64)
    y = np.full((args.num, args.n_class), float(args.cond_value), dtype=np.float64)

    # Always normalize conditions with the param scaler used in training
    param_scaler = load(open(os.path.join(data_dir, "param_scaler.pkl"), "rb"))
    y_in = param_scaler.transform(y)
    Xg = decoder.predict([z, y_in], verbose=0)

    # Optionally invert X normalization if training used --normalize (X_scaler may be None)
    if args.inverse_normalize:
        X_scaler = load(open(os.path.join(data_dir, "X_scaler.pkl"), "rb"))
        if X_scaler is not None:
            Xg = X_scaler.inverse_transform(Xg)
        else:
            print("[warn] X_scaler.pkl is None (training likely didn't use --normalize); skipping inverse normalization.")

    Xg = Xg.astype(np.float32)

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, Xg)
    print(f"[save] {Xg.shape} -> {args.out_npy}")


if __name__ == "__main__":
    main()


