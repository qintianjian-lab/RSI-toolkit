#!/usr/bin/env python3
"""
Evaluate an RSI classifier checkpoint on a labeled split.

The script reports threshold-independent metrics (AUROC/AUPRC), PSB-centered
threshold operating points, and optional top-K retrieval metrics.

Example:
  python custom_test.py \
    --test_data_dir data/lamost_folds/fold_1/test \
    --spectrum_dir spectrum \
    --label_file label/label.csv \
    --model_path runs/checkpoints/example.ckpt \
    --topk 150 200 \
    --batch_size 64 \
    --device auto
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)

# Add project root for local execution.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load project configuration when available.
try:
    from config.config import config
    from config.model_config import get_model_kwargs
    from model.lightning import BuildLightningModel
except ImportError:
    config = {'type_list': [0, 1], 'spectrum_size': 3857}
    get_model_kwargs = lambda _name, _cfg: _cfg.get('model_kwargs') or {}
    print("[Warning] Project configuration not found; using fallback defaults.")

import wandb


# ==========================================
# Helper classes and metric utilities
# ==========================================

class TimeTracker:
    """Collect elapsed-time statistics for named steps."""
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str):
        self.start_times[name] = time.time()
    
    def end(self, name: str):
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name in self.times:
                self.times[name].append(elapsed)
            else:
                self.times[name] = [elapsed]
            del self.start_times[name]
            return elapsed
        return 0
    
    def record(self, name: str, duration: float):
        if name in self.times:
            self.times[name].append(duration)
        else:
            self.times[name] = [duration]
            
    def get_all_stats(self):
        stats = {}
        for name, times in self.times.items():
            stats[name] = {
                'total': sum(times),
                'mean': sum(times) / len(times) if times else 0,
                'count': len(times)
            }
        return stats

class NumpyEncoder(json.JSONEncoder):
    """Serialize NumPy values to JSON-compatible objects."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calculate_metrics_at_threshold(y_true, y_probs, threshold):
    """
    Compute accuracy, macro metrics, positive-class metrics, and per-class
    support at a fixed threshold.
    """
    # Predicted binary labels at the requested operating point.
    y_pred = (y_probs > threshold).astype(int)
    
    # Overall accuracy.
    acc = accuracy_score(y_true, y_pred)
    
    # Per-class Precision/Recall/F1/Support.
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    # Macro metrics average both classes and are close to the balanced view.
    p_macro = float(np.mean(precisions))
    r_macro = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))
    
    # Positive-class metrics for rare-object retrieval interpretation.
    f1_binary = float(f1s[1])
    p_binary = float(precisions[1])
    r_binary = float(recalls[1])
    
    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        # macro (balanced-style) metrics
        'f1_macro': float(f1_macro),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        # binary positive-class metrics (kept for reference)
        'f1_binary': float(f1_binary),
        'precision_binary': float(p_binary),
        'recall_binary': float(r_binary),
        'per_class': {
            'class_0': {
                'precision': float(precisions[0]),
                'recall': float(recalls[0]),
                'f1': float(f1s[0]),
                'support': int(supports[0])
            },
            'class_1': {
                'precision': float(precisions[1]),
                'recall': float(recalls[1]),
                'f1': float(f1s[1]),
                'support': int(supports[1])
            }
        }
    }


def analyze_binary_performance(y_true, y_probs, fixed_threshold=0.5):
    """
    Find binary operating points and compute threshold-dependent metrics.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    def compute_topk_metrics(y_true_arr: np.ndarray, y_prob_arr: np.ndarray, ks):
        """
        Top-K search metrics based on ranking by positive probability.
        Returns Precision@K / Recall@K for positive class (label==1).
        """
        y_true_arr = np.asarray(y_true_arr).astype(int)
        y_prob_arr = np.asarray(y_prob_arr).astype(float)
        n = int(y_true_arr.shape[0])
        total_pos = int((y_true_arr == 1).sum())
        if n <= 0 or total_pos <= 0:
            return {"total_pos": int(total_pos), "items": []}
        order = np.argsort(-y_prob_arr)  # descending
        items = []
        for k in (ks or []):
            try:
                k_int = int(k)
            except Exception:
                continue
            if k_int <= 0:
                continue
            k_eff = min(k_int, n)
            top_idx = order[:k_eff]
            tp = int((y_true_arr[top_idx] == 1).sum())
            items.append({
                "k": int(k_int),
                "k_eff": int(k_eff),
                "tp": int(tp),
                "precision_at_k": float(tp / max(k_eff, 1)),
                "recall_at_k": float(tp / max(total_pos, 1)),
            })
        return {"total_pos": int(total_pos), "items": items}
    
    # --- 1. Threshold-independent metrics ---
    try:
        auroc = roc_auc_score(y_true, y_probs)
    except:
        auroc = 0.0
    auprc = average_precision_score(y_true, y_probs)
    
    # --- 2. PR-curve points for threshold selection ---
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # thresholds is one element shorter than precision/recall. The padded
    # version is only for plotting; threshold search uses aligned points.
    thresholds_padded = np.append(thresholds, 1.0)
    # Arrays aligned as P[1:], R[1:], T[:].
    P_sel = precisions[1:] if len(precisions) > 1 else np.array([])
    R_sel = recalls[1:] if len(recalls) > 1 else np.array([])
    T_sel = thresholds.copy() if thresholds is not None else np.array([])
    
    # --- 3. Select operating-point thresholds ---
    
    # A. Best-F1 threshold.
    if P_sel.size > 0 and R_sel.size > 0 and T_sel.size > 0:
        numerator = 2 * P_sel * R_sel
        denominator = P_sel + R_sel
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores_sel = np.divide(numerator, denominator)
            f1_scores_sel[np.isnan(f1_scores_sel)] = 0
        best_idx_sel = int(np.argmax(f1_scores_sel))
        best_thr_val = float(T_sel[best_idx_sel])
    else:
        # Degenerate case: no valid point, fall back just below 1.0.
        best_thr_val = float(np.nextafter(1.0, 0.0))
    
    # B. Highest precision among points satisfying recall >= target_recall.
    target_recall = float(getattr(analyze_binary_performance, "_target_recall", 0.90))
    if R_sel.size > 0 and T_sel.size > 0:
        valid_r_mask = R_sel >= target_recall
        if np.any(valid_r_mask):
            idxs = np.where(valid_r_mask)[0]
            best_p_at_r_idx = idxs[int(np.argmax(P_sel[idxs]))]
            thr_recall90 = float(T_sel[best_p_at_r_idx])
        else:
            thr_recall90 = float(np.nextafter(1.0, 0.0))
    else:
        thr_recall90 = float(np.nextafter(1.0, 0.0))
        
    # C. Highest recall among points satisfying precision >= target_precision.
    target_prec = float(getattr(analyze_binary_performance, "_target_precision", 0.75))
    if P_sel.size > 0 and T_sel.size > 0:
        valid_p_mask = P_sel >= target_prec
        if np.any(valid_p_mask):
            idxs = np.where(valid_p_mask)[0]
            best_r_at_p_idx = idxs[int(np.argmax(R_sel[idxs]))]
            thr_prec90 = float(T_sel[best_r_at_p_idx])
        else:
            thr_prec90 = float(np.nextafter(1.0, 0.0))
    else:
        thr_prec90 = float(np.nextafter(1.0, 0.0))

    # --- 4. Metrics at selected operating points ---
    metrics_best_f1 = calculate_metrics_at_threshold(y_true, y_probs, best_thr_val)
    metrics_fixed = calculate_metrics_at_threshold(y_true, y_probs, fixed_threshold)
    metrics_recall90 = calculate_metrics_at_threshold(y_true, y_probs, thr_recall90)
    metrics_prec90 = calculate_metrics_at_threshold(y_true, y_probs, thr_prec90)

    # --- 5. Top-K metrics (optional; K list injected by caller) ---
    ks = getattr(analyze_binary_performance, "_topk_ks", [])
    topk = compute_topk_metrics(y_true, y_probs, ks) if ks else {"total_pos": int((y_true == 1).sum()), "items": []}
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'best_f1': metrics_best_f1,
        'fixed_threshold': metrics_fixed,
        'search_recall_90': metrics_recall90,
        'search_precision_90': metrics_prec90,
        'topk': topk
    }


def _unwrap_logits(output):
    """
    Support models that return dict outputs by extracting a logits/pred tensor.
    """
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "pred" in output:
            return output["pred"]
    return output


def _run_inference_collect(model, dataloader, device, args):
    """Run model inference and collect probs/labels/ids (shared by test and calibration)."""
    all_probs = []
    all_labels = []
    all_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100):
            spectra, labels, metas = batch
            spectra = spectra.to(device)
            labels = labels.to(device)

            if hasattr(model, 'forward_with_context'):
                output = model.forward_with_context(spectra, metas)
            else:
                output = model(spectra)
            output = _unwrap_logits(output)

            if args.temperature != 1.0:
                output = output / args.temperature

            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            ids = metas['id']
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            all_ids.extend(list(ids))

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    return all_probs, all_labels, all_ids


def _compute_thresholds_from_labels(y_true, pos_probs, args):
    """Compute operating-point thresholds using labeled set (e.g., val) for calibration."""
    analyze_binary_performance._topk_ks = []  # no need for topk in threshold calibration
    analyze_binary_performance._target_recall = float(getattr(args, "target_recall", 0.90))
    analyze_binary_performance._target_precision = float(getattr(args, "target_precision", 0.75))
    metrics = analyze_binary_performance(y_true, pos_probs, fixed_threshold=args.threshold)
    return {
        "best_f1_threshold": float(metrics["best_f1"]["threshold"]),
        "high_recall_threshold": float(metrics["search_recall_90"]["threshold"]),
        "high_precision_threshold": float(metrics["search_precision_90"]["threshold"]),
        "target_recall": float(getattr(args, "target_recall", 0.90)),
        "target_precision": float(getattr(args, "target_precision", 0.75)),
    }


def _evaluate_at_thresholds(y_true, pos_probs, thresholds: dict):
    """Evaluate on a labeled set (e.g., test) using pre-selected thresholds."""
    return {
        "best_f1": calculate_metrics_at_threshold(y_true, pos_probs, thresholds["best_f1_threshold"]),
        "high_recall": calculate_metrics_at_threshold(y_true, pos_probs, thresholds["high_recall_threshold"]),
        "high_precision": calculate_metrics_at_threshold(y_true, pos_probs, thresholds["high_precision_threshold"]),
    }


class CustomSpectrumDataset(torch.utils.data.Dataset):
    """Spectrum dataset for checkpoint evaluation."""
    
    def __init__(self, test_data_dir, spectrum_dir='spectra', label_file='index.csv', 
                 type_list=[0, 1], spectrum_size=3857, time_tracker=None):
        super().__init__()
        self.test_data_dir = test_data_dir
        self.spectrum_dir = spectrum_dir
        self.type_list = type_list
        self.spectrum_size = spectrum_size
        self.time_tracker = time_tracker
        
        label_path = os.path.join(test_data_dir, label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        self.label_df = pd.read_csv(label_path)
        
        # Keep only labels listed in type_list.
        original_len = len(self.label_df)
        
        # Detect label column.
        label_col = next((c for c in ['label', 'class', 'target', 'y'] if c in self.label_df.columns), None)
        if label_col:
            self.label_col = label_col
            self.label_df = self.label_df[self.label_df[label_col].isin(type_list)].reset_index(drop=True)
            if len(self.label_df) < original_len:
                print(f"[Info] Filtered {original_len - len(self.label_df)} samples with labels outside type_list.")
        else:
            raise KeyError("Label column not found (label/class/target)")

        # Detect spectrum identifier column.
        self.id_col = next((c for c in ['basename', 'id', 'filename', 'spec_id'] if c in self.label_df.columns), None)
        if not self.id_col:
             # SDSS-style plate/mjd/fiber identifiers are combined below.
             if all(c in self.label_df.columns for c in ['plate', 'mjd', 'fiber']):
                 self.id_col = 'combined_id' 
             else:
                 raise KeyError("Spectrum ID column not found")

        print(f"[Info] Loaded evaluation set: {len(self.label_df)} samples")
        print(f"[Info] Label distribution: {self.label_df[self.label_col].value_counts().to_dict()}")

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if self.time_tracker: self.time_tracker.start('data_loading')
        
        row = self.label_df.iloc[idx]
        label_val = row[self.label_col]
        label_idx = self.type_list.index(label_val)
        
        # Build spectrum filename.
        if self.id_col == 'combined_id':
            plate = str(int(row['plate'])).zfill(4)
            mjd = str(int(row['mjd']))
            fiber = str(int(row['fiber'])).zfill(4)
            fname = f"spec-{plate}-{mjd}-{fiber}.fits"
            data_id = f"{plate}-{mjd}-{fiber}"
        else:
            data_id = str(row[self.id_col])
            # Detect common file suffixes.
            if data_id.endswith('.csv') or data_id.endswith('.fits'):
                fname = data_id
            else:
                # Prefer FITS when available, otherwise fall back to CSV.
                if os.path.exists(os.path.join(self.test_data_dir, self.spectrum_dir, data_id + '.fits')):
                    fname = data_id + '.fits'
                else:
                    fname = data_id + '.csv'

        fpath = os.path.join(self.test_data_dir, self.spectrum_dir, fname)
        
        # Read spectrum data from FITS or CSV.
        try:
            if fname.endswith('.fits'):
                from astropy.io import fits
                with fits.open(fpath) as hdul:
                    spectrum_np = None
                    for i in range(len(hdul)):
                        hdu = hdul[i]
                        data = getattr(hdu, 'data', None)
                        if data is None:
                            continue
                        # Table HDU: prefer common flux column names.
                        if hasattr(data, 'names') and data.names is not None:
                            names = list(data.names)
                            names_lower = {n.lower(): n for n in names}
                            chosen = None
                            for cand in ['flux', 'flx', 'intensity', 'spec', 'f_flux']:
                                if cand in names_lower:
                                    chosen = names_lower[cand]
                                    break
                                # Fallback: case-insensitive exact match.
                                chosen = next((orig for orig in names if orig.lower() == cand), chosen)
                            if chosen is not None:
                                flux_raw = data[chosen]
                                flux_arr = np.array(
                                    flux_raw,
                                    dtype=object if getattr(flux_raw, 'dtype', None) == object else None
                                )
                                if isinstance(flux_arr, np.ndarray) and flux_arr.dtype == object:
                                    if flux_arr.size == 1 and isinstance(flux_arr[0], (np.ndarray, list, tuple)):
                                        flux_arr = np.asarray(flux_arr[0])
                                if hasattr(flux_arr, 'ndim') and flux_arr.ndim == 2:
                                    if 1 in flux_arr.shape:
                                        flux_arr = flux_arr.reshape(-1)
                                    else:
                                        flux_arr = flux_arr[0]
                                spectrum_np = np.asarray(flux_arr, dtype=np.float32).reshape(-1)
                                if spectrum_np.size > 0:
                                    break
                        # Array HDU: flatten directly.
                        if isinstance(data, np.ndarray):
                            tmp = data.astype(np.float32, copy=False)
                            spectrum_np = tmp.reshape(-1) if tmp.ndim > 1 else tmp
                            if spectrum_np.size > 0:
                                break
                    if spectrum_np is None or spectrum_np.size == 0:
                        raise ValueError("Empty FITS")
                    spectrum = spectrum_np
            else:
                # CSV
                spectrum = np.loadtxt(fpath, delimiter=',').astype(np.float32)
                if spectrum.ndim == 2: spectrum = spectrum[:, 1] # second column is flux

            # Pad or truncate to the configured spectrum length.
            spectrum = spectrum.flatten()
            if len(spectrum) > self.spectrum_size:
                spectrum = spectrum[:self.spectrum_size]
            elif len(spectrum) < self.spectrum_size:
                spectrum = np.pad(spectrum, (0, self.spectrum_size - len(spectrum)))
            
            spectrum_tensor = torch.from_numpy(spectrum).float().unsqueeze(0)
            
        except Exception as e:
            print(f"[Error] Reading {fname}: {e}")
            spectrum_tensor = torch.zeros((1, self.spectrum_size))
            label_idx = -1 # invalid sample marker

        if self.time_tracker: self.time_tracker.end('data_loading')
        
        # Metadata: redshift if present.
        z = 0.0
        for k in ['z', 'redshift', 'Z']:
            if k in row:
                z = float(row[k])
                break
        
        return spectrum_tensor, label_idx, {'z': torch.tensor(z).float(), 'id': data_id}


def _build_model_from_config(used_model_name: str | None, device):
    """Build a model from project config when no checkpoint is provided."""
    um = (used_model_name or config.get('used_model', '') or '').strip()
    if not um:
        raise RuntimeError("used_model was not provided and could not be read from config.")
    lm = BuildLightningModel(
        model_name=um,
        learn_rate=config.get('learn_rate', 1e-3),
        cos_annealing_t_0=config.get('cos_annealing_t_0', 10),
        cos_annealing_t_mult=config.get('cos_annealing_t_mult', 1),
        cos_annealing_eta_min=config.get('cos_annealing_eta_min', 1e-6),
        weight_decay=config.get('weight_decay', 1e-2),
        in_channel=config.get('in_channel', 1),
        spectrum_size=config.get('spectrum_size', 3857),
        num_classes=len(config.get('type_list', [0, 1])),
        classes_name_list=config.get('type_list', [0, 1]),
        enable_torch_2=config.get('enable_torch_2.0', False),
        torch_2_compile_mode=config.get('torch_2.0_compile_mode', 'default'),
        model_kwargs=get_model_kwargs(um, config),
    )
    model = lm.model
    model.to(device)
    model.eval()
    return model


def load_model(
    model_path,
    device,
    used_model_name: str | None = None,
    *,
    strict_ckpt: bool = True,
):
    """
    Load an evaluation model.

    With strict_ckpt=True, a non-empty checkpoint path must load successfully;
    the evaluator will not silently fall back to a randomly initialized
    config-built model.
    """
    path = model_path.strip() if isinstance(model_path, str) else ""
    if not path:
        print("[Info] No checkpoint provided. Will build a fresh model from config/used_model.")
        model = _build_model_from_config(used_model_name, device)
        model.to(device)
        model.eval()
        return model

    print(f"[Info] Loading model from {path}")

    errors: list[tuple[str, BaseException]] = []

    def _try_build_lightning() -> torch.nn.Module:
        lightning_model = BuildLightningModel.load_from_checkpoint(path)
        return lightning_model.model

    try:
        model = _try_build_lightning()
    except BaseException as e:
        errors.append(("BuildLightningModel.load_from_checkpoint", e))
        model = None

        if model is None:
            if strict_ckpt:
                msg = "\n".join(f"  - {name}: {exc!r}" for name, exc in errors)
                raise RuntimeError(
                    f"[strict_ckpt] Could not load checkpoint and fallback is disabled: {path}\n{msg}"
                ) from errors[0][1]
            print("[Warning] All checkpoint loaders failed; falling back to config-built model (not recommended).")
            model = _build_model_from_config(used_model_name, device)

    model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device, num_classes, args):
    """Run evaluation and collect metric summaries."""
    print("[Info] Starting Inference...")
    all_probs, all_labels, all_ids = _run_inference_collect(model, dataloader, device, args)
    
    results = {}
    
    # --- Binary PSB/E+A detection analysis ---
    if num_classes == 2:
        print("\n[Analysis] Binary task detected; computing threshold analysis...")
        pos_probs = all_probs[:, 1] # positive-class probability

        # Inject Top-K list into analyze function (keeps signature stable)
        analyze_binary_performance._topk_ks = list(getattr(args, "topk", []) or [])
        analyze_binary_performance._target_recall = float(getattr(args, "target_recall", 0.90))
        analyze_binary_performance._target_precision = float(getattr(args, "target_precision", 0.75))
        
        metrics = analyze_binary_performance(all_labels, pos_probs, fixed_threshold=args.threshold)
        
        results.update(metrics)
        # Use the best-F1 operating point for prediction export.
        results['predictions'] = (pos_probs > metrics['best_f1']['threshold']).astype(int) 
        results['pos_probs'] = pos_probs
        
    else:
        # Standard multiclass analysis.
        preds = np.argmax(all_probs, axis=1)
        results['accuracy'] = accuracy_score(all_labels, preds)
        p, r, f, _ = precision_recall_fscore_support(all_labels, preds, average='macro')
        results['macro_metrics'] = {'p': p, 'r': r, 'f1': f}
        results['predictions'] = preds

    results['labels'] = all_labels
    results['ids'] = all_ids
    return results


def fit_sklearn_baseline_from_dir(
    model: torch.nn.Module,
    fit_data_dir: str,
    device: torch.device,
    *,
    spectrum_dir: str,
    label_file: str,
    batch_size: int,
    num_workers: int = 4,
) -> None:
    """
    Fit sklearn-style baselines that expose cache_with_labels.
    """
    if not hasattr(model, "cache_with_labels"):
        return
    fd = (fit_data_dir or "").strip()
    if not fd:
        raise ValueError("fit_data_dir is required for RF/SVM evaluation")
    if not os.path.isdir(fd):
        raise FileNotFoundError(f"Fit dataset directory not found: {fd}")
    spec_dir = (spectrum_dir or "spectra").strip() or "spectra"
    lf = (label_file or "index.csv").strip() or "index.csv"
    fit_dataset = CustomSpectrumDataset(
        fd,
        spectrum_dir=spec_dir,
        label_file=lf,
        type_list=config.get("type_list", [0, 1]),
        spectrum_size=config.get("spectrum_size", 3857),
    )
    fit_loader = torch.utils.data.DataLoader(
        fit_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    print(f"[Info] Fitting sklearn baseline on: {fd}")
    with torch.no_grad():
        for batch in fit_loader:
            spectra, labels, metas = batch
            spectra = spectra.to(device)
            labels = labels.to(device)
            try:
                model.cache_with_labels(spectra, labels)
            except Exception:
                pass
        maybe_fit = getattr(model, "_maybe_fit", None)
        if callable(maybe_fit):
            try:
                maybe_fit()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='Evaluate an RSI classifier checkpoint on a labeled split.')
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='', help='Optional checkpoint path; SVM/RF baselines may omit it.')
    parser.add_argument('--used_model', type=str, default='', help='Model name used when building from config without a checkpoint.')
    parser.add_argument('--output_dir', type=str, default='./test_results_pro')
    # Dataset layout controls, compatible with {split}/spectrum and label/label.csv.
    parser.add_argument('--spectrum_dir', type=str, default='spectra', help="Spectrum directory name under test_data_dir.")
    parser.add_argument('--label_file', type=str, default='index.csv', help="Label CSV path relative to test_data_dir.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--threshold', type=float, default=0.5, help='Fixed threshold for threshold-dependent metrics.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Logit temperature scaling factor.')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--topk', type=int, nargs='*', default=[],
                        help='Optional Top-K retrieval metrics ranked by positive-class probability.')
    parser.add_argument('--target_recall', type=float, default=0.90, help='Target recall for the high-recall operating point.')
    parser.add_argument('--target_precision', type=float, default=0.75, help='Target precision for the high-precision operating point.')
    # calibration (val-derived thresholds -> report on test)
    parser.add_argument('--calib_data_dir', type=str, default='',
                        help='Optional labeled validation directory for threshold calibration.')
    parser.add_argument('--calib_spectrum_dir', type=str, default='',
                        help="Calibration spectrum directory name; defaults to --spectrum_dir.")
    parser.add_argument('--calib_label_file', type=str, default='',
                        help="Calibration label CSV path; defaults to --label_file.")
    # Separate fit split for sklearn baselines before calibration/evaluation.
    parser.add_argument('--fit_data_dir', type=str, default='',
                        help='Training directory used to fit RF/SVM baselines.')
    parser.add_argument('--fit_spectrum_dir', type=str, default='',
                        help="Fit-set spectrum directory name; defaults to --spectrum_dir.")
    parser.add_argument('--fit_label_file', type=str, default='',
                        help="Fit-set label CSV path; defaults to --label_file.")
    parser.add_argument(
        '--strict_ckpt',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Require checkpoint loading to succeed when --model_path is provided.',
    )
    args = parser.parse_args()
    # auto default calib_data_dir: sibling "val" of test_data_dir (if exists)
    if (not isinstance(args.calib_data_dir, str)) or (not args.calib_data_dir.strip()):
        try:
            td = os.path.abspath(args.test_data_dir)
            base = os.path.basename(td.rstrip(os.sep))
            parent = os.path.dirname(td.rstrip(os.sep))
            cand = os.path.join(parent, "val") if base.lower() == "test" else os.path.join(td, "val")
            if os.path.isdir(cand):
                args.calib_data_dir = cand
                print(f"[Info] calib_data_dir not provided. Auto set to: {args.calib_data_dir}")
        except Exception:
            pass
    # auto default fit_data_dir: sibling "train" of test_data_dir (if exists)
    if (not isinstance(args.fit_data_dir, str)) or (not args.fit_data_dir.strip()):
        try:
            td = os.path.abspath(args.test_data_dir)
            base = os.path.basename(td.rstrip(os.sep))
            parent = os.path.dirname(td.rstrip(os.sep))
            cand = os.path.join(parent, "train") if base.lower() == "test" else os.path.join(td, "train")
            if os.path.isdir(cand):
                args.fit_data_dir = cand
                print(f"[Info] fit_data_dir not provided. Auto set to: {args.fit_data_dir}")
        except Exception:
            pass
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device)
    print(f"[Info] Device: {device}")
    
    # Setup Data
    dataset = CustomSpectrumDataset(
        args.test_data_dir,
        spectrum_dir=args.spectrum_dir,
        label_file=args.label_file,
        type_list=config.get('type_list', [0, 1]),
        spectrum_size=config.get('spectrum_size', 3857)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    # Optional: calibration dataloader (val) for choosing thresholds
    calib_results = None
    calib_thresholds = None
    # Load model
    model = load_model(
        args.model_path,
        device,
        used_model_name=args.used_model or config.get('used_model', ''),
        strict_ckpt=args.strict_ckpt,
    )
    # Fit sklearn-style baselines before calibration/evaluation.
    if hasattr(model, "cache_with_labels"):
        fit_used = False
        # Choose fit dataset (prefer explicit --fit_data_dir; fallback to calib; else try sibling train auto; else error on eval)
        fit_dir = args.fit_data_dir.strip() if isinstance(args.fit_data_dir, str) else ''
        if not fit_dir:
            fit_dir = args.calib_data_dir.strip() if isinstance(args.calib_data_dir, str) else ''
        if fit_dir:
            fit_spec_dir = args.fit_spectrum_dir.strip() if args.fit_spectrum_dir.strip() else args.spectrum_dir
            fit_label_file = args.fit_label_file.strip() if args.fit_label_file.strip() else args.label_file
            try:
                fit_sklearn_baseline_from_dir(
                    model,
                    fit_dir,
                    device,
                    spectrum_dir=fit_spec_dir,
                    label_file=fit_label_file,
                    batch_size=args.batch_size,
                    num_workers=4,
                )
                fit_used = True
            except Exception as e:
                print(f"[Warning] Failed to build fit dataset at {fit_dir}: {e}")
        if not fit_used:
            print("[Warning] SVM/RF requires a labeled dataset to fit before evaluation. "
                  "Provide --fit_data_dir (recommended: train split) or --calib_data_dir.")
    # Calibration thresholds using calib set (VAL) — purely for threshold search, not for fitting
    if isinstance(args.calib_data_dir, str) and args.calib_data_dir.strip():
        calib_spec_dir = args.calib_spectrum_dir.strip() if args.calib_spectrum_dir.strip() else args.spectrum_dir
        calib_label_file = args.calib_label_file.strip() if args.calib_label_file.strip() else args.label_file
        try:
            calib_dataset = CustomSpectrumDataset(
                args.calib_data_dir,
                spectrum_dir=calib_spec_dir,
                label_file=calib_label_file,
                type_list=config.get('type_list', [0, 1]),
                spectrum_size=config.get('spectrum_size', 3857)
            )
            calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
            calib_probs, calib_labels, _ = _run_inference_collect(model, calib_loader, device, args)
            calib_pos_probs = calib_probs[:, 1]
            calib_thresholds = _compute_thresholds_from_labels(calib_labels, calib_pos_probs, args)
            print(f"[Info] Calibrated thresholds from calib set: {calib_thresholds}")
        except Exception as e:
            print(f"[Warning] Skip threshold calibration due to calib set error: {e}")

    # Setup Model
    # Evaluation
    results = evaluate_model(model, dataloader, device, len(config.get('type_list', [0, 1])), args)

    # If calibrated thresholds exist, evaluate test using thresholds chosen on calib
    if calib_thresholds is not None and 'pos_probs' in results and 'labels' in results:
        calib_results = _evaluate_at_thresholds(results['labels'], results['pos_probs'], calib_thresholds)
        results['calibrated_thresholds'] = calib_thresholds
        results['calibrated_on_test'] = calib_results
    
    # =========================================================
    # Print evaluation report
    # =========================================================
    def print_per_class(metrics_dict, indent="   "):
        """Print per-class metrics."""
        pc = metrics_dict['per_class']
        print(f"{indent}Per-class metrics (Precision, Recall, F1, Support):")
        print(f"{indent}  Class 0 (Normal): P={pc['class_0']['precision']:.4f}, R={pc['class_0']['recall']:.4f}, F1={pc['class_0']['f1']:.4f}, N={pc['class_0']['support']}")
        print(f"{indent}  Class 1 (E+A)   : P={pc['class_1']['precision']:.4f}, R={pc['class_1']['recall']:.4f}, F1={pc['class_1']['f1']:.4f}, N={pc['class_1']['support']}")

    print("\n" + "="*60)
    print(" >>> Model Evaluation Report <<<")
    print("="*60)
    
    if 'best_f1' in results: # binary classification
        print(f"1. Threshold-independent ranking metrics:")
        print(f"   - AUROC: {results['auroc']:.4f}")
        print(f"   - AUPRC: {results['auprc']:.4f}")
        print("-" * 30)

        # --- Top-K (optional) ---
        topk = results.get("topk", {})
        if isinstance(topk, dict) and topk.get("items"):
            print(f"1.5 Top-K retrieval metrics:")
            print(f"   - Total Positives: {topk.get('total_pos', 0)}")
            for it in topk["items"]:
                print(f"   - @K={it['k']} (effective={it['k_eff']}): "
                      f"Precision@K={it['precision_at_k']:.4f}, Recall@K={it['recall_at_k']:.4f}, TP@K={it['tp']}")
            print("-" * 30)
        
        # --- Best F1 ---
        bf1 = results['best_f1']
        print(f"2. Best-F1 operating point:")
        print(f"   - Threshold: {bf1['threshold']:.4f}")
        print(f"   - Overall Accuracy (micro): {bf1['accuracy']:.4f}")
        print(f"   - Balanced Accuracy (= Macro Recall): {bf1['recall_macro']:.4f}")
        print(f"   - Macro F1: {bf1['f1_macro']:.4f}")
        print(f"   - Macro Precision: {bf1['precision_macro']:.4f}")
        print(f"   - Macro Recall: {bf1['recall_macro']:.4f}")
        print_per_class(bf1)
        print("-" * 30)
        
        # --- Fixed Threshold ---
        fix = results['fixed_threshold']
        print(f"3. Fixed threshold = {args.threshold}:")
        print(f"   - Overall Accuracy (micro): {fix['accuracy']:.4f}")
        print(f"   - Balanced Accuracy (= Macro Recall): {fix['recall_macro']:.4f}")
        print(f"   - Macro F1: {fix['f1_macro']:.4f}")
        print(f"   - Macro Precision: {fix['precision_macro']:.4f}")
        print(f"   - Macro Recall: {fix['recall_macro']:.4f}")
        print_per_class(fix)
        print("-" * 30)
        
        # --- Search Scenarios ---
        r90 = results['search_recall_90']
        p90 = results['search_precision_90']
        
        print(f"4. Retrieval operating points:")
        print(f"   [A] High-recall target (~{args.target_recall*100:.0f}% recall):")
        print(f"       - Threshold: {r90['threshold']:.4f}")
        print(f"       - Overall Accuracy (micro): {r90['accuracy']:.4f}")
        print(f"       - Balanced Accuracy (= Macro Recall): {r90['recall_macro']:.4f}")
        print(f"       - Macro metrics: Recall={r90['recall_macro']:.4f}, Precision={r90['precision_macro']:.4f}, F1={r90['f1_macro']:.4f}")
        print_per_class(r90, indent="       ")
        
        print(f"\n   [B] High-precision target (~{args.target_precision*100:.0f}% precision):")
        print(f"       - Threshold: {p90['threshold']:.4f}")
        print(f"       - Overall Accuracy (micro): {p90['accuracy']:.4f}")
        print(f"       - Balanced Accuracy (= Macro Recall): {p90['recall_macro']:.4f}")
        print(f"       - Macro metrics: Recall={p90['recall_macro']:.4f}, Precision={p90['precision_macro']:.4f}, F1={p90['f1_macro']:.4f}")
        print_per_class(p90, indent="       ")

        # --- Calibrated thresholds (val -> test) ---
        if isinstance(results.get('calibrated_on_test', None), dict) and isinstance(results.get('calibrated_thresholds', None), dict):
            cal = results['calibrated_on_test']
            thr = results['calibrated_thresholds']
            print("-" * 30)
            print("5. Thresholds calibrated on validation and evaluated on test:")
            print(f"   - calib_data_dir: {args.calib_data_dir}")
            print(f"   - target_recall={thr.get('target_recall', args.target_recall):.2f}, target_precision={thr.get('target_precision', args.target_precision):.2f}")
            # best_f1
            b = cal["best_f1"]
            print("   [Best-F1 threshold from VAL]:")
            print(f"     - threshold: {b['threshold']:.4f}")
            print(f"     - Overall Accuracy (micro): {b['accuracy']:.4f}")
            print(f"     - Balanced Accuracy (= Macro Recall): {b['recall_macro']:.4f}")
            print(f"     - Macro Precision/Recall/F1: {b['precision_macro']:.4f}/{b['recall_macro']:.4f}/{b['f1_macro']:.4f}")
            print_per_class(b, indent="     ")
            # high recall
            hr = cal["high_recall"]
            print(f"   [High Recall threshold from VAL (~{args.target_recall*100:.0f}% target)]:")
            print(f"     - threshold: {hr['threshold']:.4f}")
            print(f"     - Overall Accuracy (micro): {hr['accuracy']:.4f}")
            print(f"     - Balanced Accuracy (= Macro Recall): {hr['recall_macro']:.4f}")
            print(f"     - Macro Precision/Recall/F1: {hr['precision_macro']:.4f}/{hr['recall_macro']:.4f}/{hr['f1_macro']:.4f}")
            print_per_class(hr, indent="     ")
            # high precision
            hp = cal["high_precision"]
            print(f"   [High Precision threshold from VAL (~{args.target_precision*100:.0f}% target)]:")
            print(f"     - threshold: {hp['threshold']:.4f}")
            print(f"     - Overall Accuracy (micro): {hp['accuracy']:.4f}")
            print(f"     - Balanced Accuracy (= Macro Recall): {hp['recall_macro']:.4f}")
            print(f"     - Macro Precision/Recall/F1: {hp['precision_macro']:.4f}/{hp['recall_macro']:.4f}/{hp['f1_macro']:.4f}")
            print_per_class(hp, indent="     ")
    
    else:
        # Multiclass summary.
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_metrics']['f1']:.4f}")

    print("="*60)
    
    # --- Save outputs ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Detailed metric JSON.
    serializable_results = {k: v for k, v in results.items() if k not in ['labels', 'ids', 'predictions', 'pos_probs']}
    with open(os.path.join(args.output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4, cls=NumpyEncoder)
        
    # 2. Per-sample prediction CSV.
    df_pred = pd.DataFrame({
        'id': results['ids'],
        'true_label': results['labels'],
        'pred_label_best': results['predictions'],
        'prob_pos': results['pos_probs'] if 'pos_probs' in results else []
    })
    df_pred.to_csv(os.path.join(args.output_dir, 'predictions_detailed.csv'), index=False)
    
    # 3. ROC and PR curve figure.
    if 'best_f1' in results:
        plt.figure(figsize=(12, 5))
        
        # ROC Curve
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(results['labels'], results['pos_probs'])
        plt.plot(fpr, tpr, label=f"AUC = {results['auroc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('ROC Curve')
        plt.legend()
        
        # PR Curve
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(results['labels'], results['pos_probs'])
        plt.plot(recall, precision, label=f"AUPRC = {results['auprc']:.4f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        plt.savefig(os.path.join(args.output_dir, 'performance_curves.png'))
        print(f"[Info] Saved result figures to {args.output_dir}")
        
    # --- Optional WandB logging ---
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=args)
        wandb_metrics = {
            'test/auroc': results.get('auroc', 0),
            'test/auprc': results.get('auprc', 0),
            'test/best_f1': results.get('best_f1', {}).get('f1_binary', 0),
            'test/best_threshold': results.get('best_f1', {}).get('threshold', 0),
        }
        # Top-K (optional)
        topk = results.get("topk", {})
        if isinstance(topk, dict) and topk.get("items"):
            for it in topk["items"]:
                k = it.get("k")
                if k is None:
                    continue
                wandb_metrics[f"test/topk_precision@{k}"] = it.get("precision_at_k", 0.0)
                wandb_metrics[f"test/topk_recall@{k}"] = it.get("recall_at_k", 0.0)
        wandb.log(wandb_metrics)
        if os.path.exists(os.path.join(args.output_dir, 'performance_curves.png')):
            wandb.log({"performance_curves": wandb.Image(os.path.join(args.output_dir, 'performance_curves.png'))})

if __name__ == '__main__':
    main()
