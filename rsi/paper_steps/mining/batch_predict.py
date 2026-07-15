"""
Usage (FITS mode):
  python -m rsi mine \
    --checkpoint runs/checkpoints/best.ckpt \
    --input-csv data/archive_index.csv \
    --id-col basename \
    --fits-root data/archive_fits \
    --fits-flux-col flux \
    --output-csv runs/archive_scores.csv \
    --device cuda:0 \
    --batch-size 256 \
    --num-workers 4 \
    --decision threshold \
    --threshold 0.4343661367893219 \
    --only-positive

Usage (CSV mode; preprocessed two-column wavelength,flux spectra):
  python -m rsi mine \
    --checkpoint runs/checkpoints/best.ckpt \
    --input-csv data/preprocessed_archive/label/label.csv \
    --id-col basename \
    --csv-spectra-root data/preprocessed_archive/spectrum \
    --output-csv runs/archive_scores_csv.csv \
    --device cuda:0 \
    --batch-size 256 \
    --num-workers 4 \
    --decision threshold \
    --threshold 0.5

Notes:
- Supports FITS spectra or preprocessed CSV spectra via --csv-spectra-root.
- FITS mode prefers the requested table flux column and falls back to numeric arrays.
- CSV mode expects two-column wavelength,flux spectra and uses the flux column.
- Spectra are center-cropped or zero-padded to --spectrum-size / config['spectrum_size'].
- Models with forward_with_context receive z metadata when present.
- Top-K selection uses a heap and keeps O(K) memory.
"""
import argparse
import os
import sys
import heapq
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import config as project_config  # noqa: E402
from model.lightning import BuildLightningModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch predict (v2, LAMOST only, fast) with DataLoader.")
    # IO and model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt checkpoint.")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV path containing IDs (and optionally redshift).")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")

    # Data columns and dirs
    parser.add_argument("--id-col", type=str, default="basename", help="Column name for base id (without extension).")
    parser.add_argument("--z-col", type=str, default="z", help="Optional redshift column name.")
    parser.add_argument("--fits-root", type=str, default="", help="Directory containing FITS files.")
    parser.add_argument("--fits-flux-col", type=str, default="flux", help="FITS flux column name (table HDU).")
    parser.add_argument("--csv-spectra-root", type=str, default="",
                        help="Directory containing preprocessed CSV spectra (two columns wavelength,flux). If set, CSV mode is used.")

    # Inference params
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--spectrum-size", type=int, default=-1, help="Override spectrum size (default from config).")

    # Selection modes
    parser.add_argument("--decision", type=str, default="none", choices=["none", "threshold", "topk"],
                        help="none: only probs; threshold: binary; topk: streaming Top-K.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold when decision=threshold.")
    parser.add_argument("--only-positive", action="store_true", help="Output only positive rows when threshold mode.")
    parser.add_argument("--topk", type=int, default=0, help="K for decision=topk (required, >0).")

    return parser.parse_args()


class LAMOSTFITSInferenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        fits_root: str,
        id_col: str,
        z_col: Optional[str],
        spectrum_size: int,
        fits_flux_col: str = "flux"
    ):
        import pandas as pd
        from astropy.io import fits  # ensure import error is early if missing
        self.pd = pd
        self.fits_lib = fits
        self.df = pd.read_csv(csv_path)
        assert id_col in self.df.columns, f"ID column '{id_col}' not found in {csv_path}"
        self.id_col = id_col
        self.z_col = z_col if (z_col and z_col in self.df.columns) else None
        self.fits_root = fits_root
        self.spectrum_size = spectrum_size
        self.fits_flux_col = fits_flux_col

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, base: str) -> Optional[str]:
        b = (base or "").strip()
        if not b:
            return None
        lower = b.lower()
        path_options = []
        if lower.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
            path_options.append(b)
        else:
            path_options.extend([
                b + ".fits", b + ".fit", b + ".FITS", b + ".FIT",
                b + ".fits.gz", b + ".fit.gz", b + ".FITS.GZ", b + ".FIT.GZ"
            ])
        for name in path_options:
            p = os.path.join(self.fits_root, name)
            if os.path.isfile(p):
                return p
        return None

    def _read_flux_fast(self, fpath: str) -> Optional[np.ndarray]:
        try:
            with self.fits_lib.open(fpath, memmap=False) as hdul:
                # prefer table HDU with specified col
                for hdu in hdul:
                    data = getattr(hdu, "data", None)
                    if data is None:
                        continue
                    if hasattr(data, "columns"):
                        names = list(getattr(data.columns, "names", []) or [])
                        # exact name or case-insensitive fallback
                        chosen = self.fits_flux_col if self.fits_flux_col in names else None
                        if chosen is None:
                            chosen = next((n for n in names if n.lower() == self.fits_flux_col.lower()), None)
                        if chosen is not None:
                            arr = np.asarray(data[chosen], dtype=np.float32)
                            if arr.ndim > 1:
                                arr = arr.reshape(-1)
                            if arr.size > 0:
                                return arr
                # fallback: first numeric array HDU
                for hdu in hdul:
                    data = getattr(hdu, "data", None)
                    if isinstance(data, np.ndarray) and data.size > 0:
                        arr = np.asarray(data, dtype=np.float32)
                        if arr.ndim > 1:
                            arr = arr.reshape(-1)
                        if arr.size > 0:
                            return arr
        except Exception:
            return None
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        base = str(row[self.id_col]).strip()
        fpath = self._resolve_path(base)
        if fpath is None:
            # return zero tensor, mark invalid id to let caller skip
            spectrum = torch.zeros(1, self.spectrum_size, dtype=torch.float32)
            meta = {'id': base, 'z': 0.0, 'path': '', 'status': 'missing'}
            # carry optional fields if present
            for k in ['obsid', 'ra', 'dec']:
                if k in self.df.columns:
                    meta[k] = row.get(k)
            return spectrum, meta

        flux = self._read_flux_fast(fpath)
        if flux is None or flux.size == 0:
            spectrum = torch.zeros(1, self.spectrum_size, dtype=torch.float32)
            meta = {'id': base, 'z': 0.0, 'path': fpath, 'status': 'read_failed'}
            for k in ['obsid', 'ra', 'dec']:
                if k in self.df.columns:
                    meta[k] = row.get(k)
            return spectrum, meta

        # center-crop/pad to spectrum_size
        L = flux.shape[0]
        if L > self.spectrum_size:
            start = (L - self.spectrum_size) // 2
            flux = flux[start:start + self.spectrum_size]
        elif L < self.spectrum_size:
            pad = self.spectrum_size - L
            left = pad // 2
            flux = np.pad(flux, (left, pad - left), mode='constant', constant_values=0.0)

        spectrum = torch.from_numpy(flux).float().unsqueeze(0)  # [1, L]
        z_val = 0.0
        if self.z_col is not None:
            try:
                zv = float(row[self.z_col])
                if not np.isnan(zv):
                    z_val = zv
            except Exception:
                z_val = 0.0
        meta = {'id': base, 'z': z_val, 'path': fpath, 'status': 'ok'}
        for k in ['obsid', 'ra', 'dec']:
            if k in self.df.columns:
                meta[k] = row.get(k)
        return spectrum, meta


class CSVSpectraInferenceDataset(Dataset):
    """
    Read preprocessed two-column CSV spectra using IDs from input-csv.
    """
    def __init__(
        self,
        csv_path: str,
        spectra_root: str,
        id_col: str,
        z_col: Optional[str],
        spectrum_size: int
    ):
        import pandas as pd
        self.pd = pd
        self.df = pd.read_csv(csv_path)
        assert id_col in self.df.columns, f"ID column '{id_col}' not found in {csv_path}"
        self.id_col = id_col
        self.z_col = z_col if (z_col and z_col in self.df.columns) else None
        self.spectra_root = spectra_root
        self.spectrum_size = spectrum_size

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, base: str) -> Optional[str]:
        b = (base or "").strip()
        if not b:
            return None
        name = b if b.lower().endswith(".csv") else (b + ".csv")
        p = os.path.join(self.spectra_root, name)
        return p if os.path.isfile(p) else None

    def _read_flux_csv(self, fpath: str) -> Optional[np.ndarray]:
        try:
            arr = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
            if arr.ndim == 1:
                # One-column CSV is interpreted as flux.
                return arr.reshape(-1)
            # Two-column CSV uses the second column as flux.
            if arr.shape[1] >= 2:
                return arr[:, 1].reshape(-1)
            return None
        except Exception:
            return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        base = str(row[self.id_col]).strip()
        fpath = self._resolve_path(base)
        if fpath is None:
            spectrum = torch.zeros(1, self.spectrum_size, dtype=torch.float32)
            meta = {'id': base, 'z': 0.0, 'path': '', 'status': 'missing'}
            for k in ['obsid', 'ra', 'dec']:
                if k in self.df.columns:
                    meta[k] = row.get(k)
            return spectrum, meta

        flux = self._read_flux_csv(fpath)
        if flux is None or flux.size == 0:
            spectrum = torch.zeros(1, self.spectrum_size, dtype=torch.float32)
            meta = {'id': base, 'z': 0.0, 'path': fpath, 'status': 'read_failed'}
            for k in ['obsid', 'ra', 'dec']:
                if k in self.df.columns:
                    meta[k] = row.get(k)
            return spectrum, meta

        # center-crop/pad to spectrum_size
        L = flux.shape[0]
        if L > self.spectrum_size:
            start = (L - self.spectrum_size) // 2
            flux = flux[start:start + self.spectrum_size]
        elif L < self.spectrum_size:
            pad = self.spectrum_size - L
            left = pad // 2
            flux = np.pad(flux, (left, pad - left), mode='constant', constant_values=0.0)

        spectrum = torch.from_numpy(flux).float().unsqueeze(0)  # [1, L]
        z_val = 0.0
        if self.z_col is not None:
            try:
                zv = float(row[self.z_col])
                if not np.isnan(zv):
                    z_val = zv
            except Exception:
                z_val = 0.0
        meta = {'id': base, 'z': z_val, 'path': fpath, 'status': 'ok'}
        for k in ['obsid', 'ra', 'dec']:
            if k in self.df.columns:
                meta[k] = row.get(k)
        return spectrum, meta


@torch.no_grad()
def run_batch_inference(model: torch.nn.Module,
                        device: torch.device,
                        batch_spectra: torch.Tensor,
                        batch_meta: List[dict]) -> np.ndarray:
    batch_spectra = batch_spectra.to(device, non_blocking=True)
    if hasattr(model, "forward_with_context"):
        # collate z
        z_tensor = torch.tensor([m['z'] for m in batch_meta], dtype=torch.float32, device=device)
        logits = model.forward_with_context(batch_spectra, {'z': z_tensor})
    else:
        logits = model(batch_spectra)
    probs = torch.softmax(logits, dim=1)
    pos_prob = probs[:, 1] if probs.shape[1] >= 2 else probs.squeeze(-1)
    return pos_prob.detach().cpu().numpy()


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    torch.set_float32_matmul_precision('high')
    lm = BuildLightningModel.load_from_checkpoint(checkpoint_path)
    model = lm.model
    model.eval()
    model.to(device)
    return model


def main():
    args = parse_args()
    import pandas as pd

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    spectrum_size = args.spectrum_size if args.spectrum_size and args.spectrum_size > 0 else project_config["spectrum_size"]

    # Dataset + loader. Use CSV mode when --csv-spectra-root is provided.
    if args.csv_spectra_root and len(args.csv_spectra_root.strip()) > 0:
        assert os.path.isdir(args.csv_spectra_root), f"csv-spectra-root not found: {args.csv_spectra_root}"
        dataset = CSVSpectraInferenceDataset(
            csv_path=args.input_csv,
            spectra_root=args.csv_spectra_root,
            id_col=args.id_col,
            z_col=args.z_col,
            spectrum_size=spectrum_size
        )
        source_mode = "csv"
    else:
        assert args.fits_root and os.path.isdir(args.fits_root), f"FITS root not found: {args.fits_root}"
        dataset = LAMOSTFITSInferenceDataset(
            csv_path=args.input_csv,
            fits_root=args.fits_root,
            id_col=args.id_col,
            z_col=args.z_col,
            spectrum_size=spectrum_size,
            fits_flux_col=args.fits_flux_col
        )
        source_mode = "fits"
    pin_mem = device.type == 'cuda'
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=pin_mem, shuffle=False, drop_last=False)

    # Model
    model = load_model(args.checkpoint, device)

    # Selection
    use_topk = (args.decision == "topk")
    if use_topk:
        assert args.topk and args.topk > 0, "--topk must be > 0 with decision=topk"
        topk_heap: List[Tuple[float, Tuple[str, str, float]]] = []  # (prob, (id, path, z))
        K = int(args.topk)

    ids: List[str] = []
    paths: List[str] = []
    z_vals: List[float] = []
    probs: List[float] = []

    total = len(dataset)
    processed = 0
    missing_count = 0
    read_failed_count = 0
    with tqdm(total=total, ncols=100, desc="Predict v2") as pbar:
        for batch in loader:
            spectra, metas = batch
            # metas is dict of lists? DataLoader default collate on dict turns into dict of lists
            # Harmonize meta list:
            meta_list = []
            if isinstance(metas, dict):
                # keys: 'id', 'z', 'path'
                ids_b = metas.get('id', [])
                zs_b = metas.get('z', [])
                paths_b = metas.get('path', [])
                status_b = metas.get('status', [])
                obsid_b = metas.get('obsid', [])
                ra_b = metas.get('ra', [])
                dec_b = metas.get('dec', [])
                # ensure list-like
                if torch.is_tensor(zs_b):
                    zs_b = zs_b.tolist()
                if torch.is_tensor(status_b):
                    status_b = status_b.tolist()
                if torch.is_tensor(obsid_b):
                    obsid_b = obsid_b.tolist()
                if torch.is_tensor(ra_b):
                    ra_b = ra_b.tolist()
                if torch.is_tensor(dec_b):
                    dec_b = dec_b.tolist()
                if not isinstance(ids_b, list):
                    ids_b = list(ids_b)
                if not isinstance(paths_b, list):
                    paths_b = list(paths_b)
                if not isinstance(status_b, list):
                    status_b = list(status_b)
                for i in range(len(ids_b)):
                    meta_item = {
                        'id': ids_b[i],
                        'z': float(zs_b[i]) if i < len(zs_b) else 0.0,
                        'path': paths_b[i] if i < len(paths_b) else '',
                        'status': status_b[i] if i < len(status_b) else 'ok'
                    }
                    if isinstance(obsid_b, list) and i < len(obsid_b):
                        meta_item['obsid'] = obsid_b[i]
                    if isinstance(ra_b, list) and i < len(ra_b):
                        try:
                            meta_item['ra'] = float(ra_b[i])
                        except Exception:
                            meta_item['ra'] = ra_b[i]
                    if isinstance(dec_b, list) and i < len(dec_b):
                        try:
                            meta_item['dec'] = float(dec_b[i])
                        except Exception:
                            meta_item['dec'] = dec_b[i]
                    meta_list.append(meta_item)
            else:
                # fallback (shouldn't happen)
                meta_list = metas

            valid_indices = []
            for i, m in enumerate(meta_list):
                status = m.get('status', 'ok')
                if status == 'ok':
                    valid_indices.append(i)
                elif status == 'missing':
                    missing_count += 1
                else:
                    read_failed_count += 1

            pbar.update(len(meta_list))

            if not valid_indices:
                continue

            if len(valid_indices) != len(meta_list):
                spectra = spectra[valid_indices]
                meta_list = [meta_list[i] for i in valid_indices]

            pos_prob = run_batch_inference(model, device, spectra, meta_list)  # np.array [B]

            if use_topk:
                for m, p in zip(meta_list, pos_prob.tolist()):
                    sid, spath, sz = m.get('id'), m.get('path'), m.get('z', 0.0)
                    obsid = m.get('obsid', '')
                    ra = m.get('ra', np.nan)
                    dec = m.get('dec', np.nan)
                    item = (sid, spath, sz, obsid, ra, dec)
                    if len(topk_heap) < K:
                        heapq.heappush(topk_heap, (p, item))
                    else:
                        if p > topk_heap[0][0]:
                            heapq.heapreplace(topk_heap, (p, item))
            else:
                for m, p in zip(meta_list, pos_prob.tolist()):
                    ids.append(m['id'])
                    paths.append(m['path'])
                    z_vals.append(m['z'])
                    probs.append(p)
                    # attach optional columns
                    # initialize lists on first use
                    if 'obsid_vals' not in locals():
                        obsid_vals = []
                        ra_vals = []
                        dec_vals = []
                    obsid_vals.append(m.get('obsid', ''))
                    ra_vals.append(m.get('ra', np.nan))
                    dec_vals.append(m.get('dec', np.nan))

            processed += len(pos_prob)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    invalid_count = missing_count + read_failed_count
    if processed == 0:
        raise RuntimeError(
            f"No valid spectra were loaded in {source_mode} mode. "
            f"missing={missing_count}, read_failed={read_failed_count}. "
            f"Please check --input-csv, --id-col and the spectra root path."
        )
    if invalid_count > 0:
        print(
            f"[Warning] Skipped invalid spectra: missing={missing_count}, "
            f"read_failed={read_failed_count}, kept={processed}"
        )

    if use_topk:
        items = [heapq.heappop(topk_heap) for _ in range(len(topk_heap))]
        items.sort(key=lambda x: x[0], reverse=True)
        ids_o = [it[1][0] for it in items]
        paths_o = [it[1][1] for it in items]
        z_o = [it[1][2] for it in items]
        obsid_o = [it[1][3] for it in items]
        ra_o = [it[1][4] for it in items]
        dec_o = [it[1][5] for it in items]
        probs_o = [it[0] for it in items]
        out_df = pd.DataFrame({
            "id": ids_o,
            "fits_path": paths_o,
            "z": z_o,
            "obsid": obsid_o,
            "ra": ra_o,
            "dec": dec_o,
            "prob_pos": probs_o,
            "selected_topk": [1] * len(items),
            "rank": np.arange(1, len(items) + 1),
            "topk_used": [len(items)] * len(items),
        })
        out_df.to_csv(args.output_csv, index=False)
        print(f"[Info] Saved Top-{len(items)} to {args.output_csv}")
    else:
        data = {
            "id": ids,
            "fits_path": paths,
            "z": z_vals,
            "prob_pos": probs
        }
        # append optional columns if collected
        if 'obsid_vals' in locals():
            data["obsid"] = obsid_vals
            data["ra"] = ra_vals
            data["dec"] = dec_vals
        out_df = pd.DataFrame(data)
        if args.decision == "threshold":
            pred_label = (out_df["prob_pos"].values >= args.threshold).astype(int)
            out_df["pred_label"] = pred_label
            out_df["threshold"] = args.threshold
            if args.only_positive:
                out_df = out_df[out_df["pred_label"] == 1].reset_index(drop=True)
        out_df.to_csv(args.output_csv, index=False)
        print(f"[Info] Saved predictions to {args.output_csv} (rows={len(out_df)})")

    print(
        f"[Info] Done. processed_valid={processed}, total_rows={total}, "
        f"missing={missing_count}, read_failed={read_failed_count}, source_mode={source_mode}"
    )


if __name__ == "__main__":
    main()
