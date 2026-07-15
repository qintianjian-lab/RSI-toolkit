"""
Train Spectra CVAE generators from NPY arrays and condition vectors.

Example:
  python -m rsi cvae-train \
    --x_train data/cvae_npy/fold_1/X_train.npy \
    --x_val data/cvae_npy/fold_1/X_val.npy \
    --c_train data/cvae_npy/fold_1/C_train.npy \
    --c_val data/cvae_npy/fold_1/C_val.npy \
    --out_dir runs/cvae_weights/fold_1 \
    --latent 16 32 64 --n_class 1
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Model import Spectra_CVAE
from Model_train import train_CVAE, KL_Loss, torch_seed

SEED = 123

# Device selection.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Spectra CVAE on NPY datasets with conditions")
    p.add_argument("--x_train", type=str, required=True, help="Path to X_train.npy (N, L)")
    p.add_argument("--x_val", type=str, required=True, help="Path to X_val.npy (M, L)")
    p.add_argument("--c_train", type=str, required=True, help="Path to C_train.npy (N, n_class) or (N,)")
    p.add_argument("--c_val", type=str, required=True, help="Path to C_val.npy (M, n_class) or (M,)")
    p.add_argument("--out_dir", type=str, default="model-weights", help="Directory to save model weights")
    p.add_argument("--latent", type=int, nargs="+", default=[2, 4, 8], help="Latent dims to train")
    p.add_argument("--n_class", type=int, default=1, help="Condition vector size")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--alpha", type=float, default=1e3, help="Weight for reconstruction loss (a)")
    p.add_argument("--beta", type=float, default=1e-1, help="Weight for KL loss (beta)")
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if present")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir or ".", exist_ok=True)

    # Load arrays
    x_tr = np.load(args.x_train).astype(np.float32)
    x_va = np.load(args.x_val).astype(np.float32)
    c_tr = np.load(args.c_train).astype(np.float32)
    c_va = np.load(args.c_val).astype(np.float32)
    if c_tr.ndim == 1:
        c_tr = c_tr.reshape(-1, 1)
    if c_va.ndim == 1:
        c_va = c_va.reshape(-1, 1)
    # Sanity
    assert x_tr.shape[0] == c_tr.shape[0], "train X and C size mismatch"
    assert x_va.shape[0] == c_va.shape[0], "val X and C size mismatch"
    print(f"X_train: {x_tr.shape}, C_train: {c_tr.shape}")
    print(f"X_val  : {x_va.shape}, C_val  : {c_va.shape}")

    # Torch tensors
    Xtr_t = torch.from_numpy(x_tr)
    Xva_t = torch.from_numpy(x_va)
    Ctr_t = torch.from_numpy(c_tr)
    Cva_t = torch.from_numpy(c_va)

    # Datasets / loaders
    train_ds = TensorDataset(Xtr_t, Ctr_t)
    val_ds = TensorDataset(Xva_t, Cva_t)
    trainloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    valloader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    MSE = nn.MSELoss()

    # Train across latent dims
    for n_latent in args.latent:
        model = Spectra_CVAE(latent_dim=n_latent, n_class=args.n_class).to(device)
        modelfile = os.path.join(args.out_dir, f"CVAE_latent{n_latent}.pth")
        # Pre-save initial weights so recovery logic in Model_train never fails on first run
        if not os.path.isfile(modelfile):
            torch.save(model.state_dict(), modelfile)
        if args.resume and os.path.isfile(modelfile):
            print(f"[resume] Loading checkpoint: {modelfile}")
            model.load_state_dict(torch.load(modelfile, map_location=device))

        print(f"\n==> Training CVAE (latent_dim={n_latent}, n_class={args.n_class}) -> {modelfile}")
        torch_seed(SEED)
        _ = train_CVAE(
            model=model,
            modelfile=modelfile,
            testloader=valloader,
            trainloader=trainloader,
            n_latent=n_latent,
            n_class=args.n_class,
            recon_loss_func=MSE,
            kl_loss_func=KL_Loss,
            patience=args.patience,
            lr=args.lr,
            beta=args.beta,
            a=args.alpha,
            SEED=SEED,
        )
        print(f"[done] Saved best weights to: {modelfile}")


if __name__ == "__main__":
    main()
