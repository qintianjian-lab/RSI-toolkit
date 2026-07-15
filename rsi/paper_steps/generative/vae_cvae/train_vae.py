"""
Train Spectra VAE generators from NPY arrays.

Example:
  python -m rsi vae-train \
    --train data/vae_npy/fold_1/train.npy \
    --test data/vae_npy/fold_1/val.npy \
    --out_dir runs/vae_weights/fold_1 \
    --latent 4 16 32 \
    --batch_size 32 --lr 5e-4 --alpha 1e3 --beta 1e-1
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Model import Spectra_VAE
from Model_train import train_VAE, KL_Loss, torch_seed

SEED = 123

# Device selection.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Spectra VAE on NPY datasets")
    p.add_argument("--train", type=str, required=True, help="Path to train.npy (N, 4000)")
    p.add_argument("--test", type=str, required=True, help="Path to test.npy (M, 4000)")
    p.add_argument("--out_dir", type=str, default="model-weights", help="Directory to save model weights")
    p.add_argument(
        "--latent",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="List of latent dimensions to train, e.g., --latent 2 4 8",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--alpha", type=float, default=1e3, help="Weight for reconstruction loss")
    p.add_argument("--beta", type=float, default=1e-1, help="Weight for KL loss")
    p.add_argument("--patience", type=int, default=100, help="Early stopping patience used internally")
    p.add_argument("--resume", action="store_true", help="If set, resume from existing checkpoint if found")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir or ".", exist_ok=True)

    # Load datasets
    train_np = np.load(args.train).astype(np.float32)
    test_np = np.load(args.test).astype(np.float32)
    if train_np.ndim != 2 or test_np.ndim != 2:
        raise ValueError(f"Expected 2D arrays; got train {train_np.shape}, test {test_np.shape}")
    print(f"train: {train_np.shape}, test: {test_np.shape}")

    # Torch tensors
    train_t = torch.from_numpy(train_np)
    test_t = torch.from_numpy(test_np)

    # Datasets and loaders (targets equal inputs for reconstruction)
    train_ds = TensorDataset(train_t, train_t)
    test_ds = TensorDataset(test_t, test_t)
    trainloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    MSE = torch.nn.MSELoss()

    # Train for each latent dimension requested
    for n_latent in args.latent:
        model = Spectra_VAE(latent_dim=n_latent).to(device)
        modelfile = os.path.join(args.out_dir, f"VAE_latent{n_latent}.pth")
        # Pre-save initial weights so recovery logic in Model_train never fails on first run
        if not os.path.isfile(modelfile):
            torch.save(model.state_dict(), modelfile)
        if args.resume and os.path.isfile(modelfile):
            print(f"[resume] Loading checkpoint: {modelfile}")
            model.load_state_dict(torch.load(modelfile, map_location=device))

        print(f"\n==> Training VAE (latent_dim={n_latent}) -> {modelfile}")
        torch_seed(SEED)
        _ = train_VAE(
            model=model,
            modelfile=modelfile,
            testloader=testloader,
            trainloader=trainloader,
            n_latent=n_latent,
            recon_loss_func=MSE,
            kl_loss_func=KL_Loss,
            patience=args.patience,
            lr=args.lr,
            alpha=args.alpha,
            beta=args.beta,
            SEED=SEED,
        )
        print(f"[done] Saved best weights to: {modelfile}")


if __name__ == "__main__":
    main()
