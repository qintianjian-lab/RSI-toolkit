#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample spectra from a trained Spectra_VAE and save to .npy

Example:
  python sample_vae.py \
    --weights "final_exp/model-weights-vae/VAE_latent8.pth" \
    --latent 8 --num 2100 \
    --out_npy "final_exp/vae_gen/generated_ea_vae_2200.npy"
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from Model import Spectra_VAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate samples with trained Spectra_VAE")
    p.add_argument("--weights", type=str, required=True, help="Path to trained .pth weights")
    p.add_argument("--latent", type=int, required=True, help="Latent dimension used by the model")
    p.add_argument("--num", type=int, default=32, help="Number of samples to generate")
    p.add_argument("--out_npy", type=str, required=True, help="Output .npy path (K, 4000)")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    model = Spectra_VAE(latent_dim=args.latent).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Sample z and decode
    with torch.no_grad():
        z = torch.randn(args.num, args.latent, device=device)
        spectra = model.decoder(z).cpu().numpy().astype(np.float32)  # (K, 4000)

    np.save(args.out_npy, spectra)
    print(f"Saved samples: {spectra.shape} -> {args.out_npy}")


if __name__ == "__main__":
    main()


