#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample spectra from a trained Spectra_CVAE with a given condition.

eg:
  python sample_cvae.py \
    --weights "final_exp/model-weights-cvae/CVAE_latent8.pth" \
    --latent 8 --num 2100 --cond_value 1.0 --n_class 1 \
    --out_npy "final_exp/cvae_gen/generated_ea_cvae_2100.npy"
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from Model import Spectra_CVAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate samples with trained Spectra_CVAE")
    p.add_argument("--weights", type=str, required=True, help="Path to trained .pth weights")
    p.add_argument("--latent", type=int, required=True, help="Latent dimension used by the model")
    p.add_argument("--n_class", type=int, default=1, help="Condition vector size")
    p.add_argument("--cond_value", type=float, default=1.0, help="Value to fill condition vector (e.g., 1.0 for E+A)")
    p.add_argument("--num", type=int, default=32, help="Number of samples to generate")
    p.add_argument("--out_npy", type=str, required=True, help="Output .npy path (K, L)")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Spectra_CVAE(latent_dim=args.latent, n_class=args.n_class).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        z = torch.randn(args.num, args.latent, device=device)
        cond = torch.full((args.num, args.n_class), float(args.cond_value), device=device)
        # Spectra_CVAE.decoder expects concatenated [z, cond]
        spectra = model.decoder(torch.cat([z, cond], dim=1)).cpu().numpy().astype(np.float32)

    np.save(args.out_npy, spectra)
    print(f"Saved samples: {spectra.shape} -> {args.out_npy}")


if __name__ == "__main__":
    main()


