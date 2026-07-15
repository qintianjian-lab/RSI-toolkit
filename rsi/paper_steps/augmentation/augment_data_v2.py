import argparse
import os
import numpy as np
import pandas as pd
import math
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# ================= Configuration =================

# Runtime overrides used by the augmentation CLI.
# - ALLOWED_STRATEGIES: restrict sampling to a non-empty strategy list.
# - FORCE_NOISE_PROB: override the probability of adding final noise.
ALLOWED_STRATEGIES = None  # type: object
FORCE_NOISE_PROB = None    # type: object
# - AUGMENT_EXPECTED: fractional expected generation ratio using floor + Bernoulli sampling.
AUGMENT_EXPECTED = None    # type: object
RANDOM_SEED = 42

EA_LABEL = 1

FOLD_DATASET_ROOT = os.environ.get("RSI_AUGMENT_DATASET_ROOT", "data/lamost_folds")
# Use "all" to scan fold_* directories, or provide an explicit fold list.
FOLD_NAMES = "all"
# Augment training splits only.
FOLD_SPLIT_NAME = "train"

# Augmentation ratio.
AUGMENT_RATIO = 4

TARGET_SNRS = [5, 10, 15, 20, 30] 
MIXUP_ALPHA_RANGE = (0.05, 0.25) 
SHIFT_PIXELS = [-5, -3, -1, 1, 3, 5]
SCALE_RANGE = (0.8, 1.2)

# Cutout parameters.
CUTOUT_NUM_HOLES = 3
CUTOUT_MAX_WIDTH = 50

# Blur parameters.
BLUR_SIGMA_RANGE = (0.5, 2.0)

# ===========================================

# Rest-frame spectra should usually avoid wavelength-shift augmentation.
REST_FRAME = True
# Explicit shift control.
ALLOW_SHIFT = False if REST_FRAME else True

# IO helpers.
def read_wave_flux_from_csv(csv_path: str):
    try:
        arr = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
        if arr.ndim == 1 and arr.size >= 2: arr = arr.reshape(-1, 2)
        if arr.shape[1] < 2: return None, None
        wave, flux = arr[:, 0], arr[:, 1]
        if wave.size >= 2 and wave[0] > wave[-1]:
            wave, flux = wave[::-1], flux[::-1]
        return wave, flux
    except: return None, None

def method_noise_injection(flux, target_snr):
    """Noise injection."""
    signal = np.median(flux)
    if signal <= 0: signal = 1e-5
    noise_sigma = signal / target_snr
    noise = np.random.normal(0, noise_sigma, size=flux.shape)
    return flux + noise

def method_mixup(flux_ea, flux_normal, alpha):
    """Linear mixup between positive and reference spectra."""
    min_len = min(len(flux_ea), len(flux_normal))
    return (1 - alpha) * flux_ea[:min_len] + alpha * flux_normal[:min_len]

def method_shift_scale(flux, shift, scale):
    """Pixel shift and amplitude scaling."""
    flux = flux * scale
    # np.roll wraps; replace wrapped values with edge values.
    shifted = np.roll(flux, shift)
    if shift > 0:
        shifted[:shift] = shifted[shift]
    elif shift < 0:
        shifted[shift:] = shifted[shift-1]
    return shifted

# Wavelength-coordinate shift by linear interpolation without wraparound.
def method_shift_scale_with_wave(wave, flux, shift_pixels, scale):
    """
    Apply f_new(w) = f_old(w - shift_pixels * dw), with zero boundary fill.
    """
    if wave is None or len(wave) < 2:
        return method_shift_scale(flux, shift_pixels, scale)
    flux = flux * scale
    dw = float(np.median(np.diff(wave)))
    delta = shift_pixels * dw
    x = wave
    y = flux
    return np.interp(x, x - delta, y, left=0.0, right=0.0)

def method_cutout(flux, num_holes, max_width):
    """
    Random cutout filled by local median plus noise.
    """
    aug_flux = flux.copy()
    length = len(flux)
    for _ in range(num_holes):
        # Random location and width.
        x = np.random.randint(0, length)
        w = np.random.randint(10, max_width)
        x1 = int(np.clip(x - w // 2, 0, length))
        x2 = int(np.clip(x + w // 2, 0, length))
        if x2 <= x1:
            continue
        # Estimate local baseline/noise from neighboring samples.
        l0 = max(0, x1 - w)
        r0 = min(length, x2 + w)
        neigh = np.r_[aug_flux[l0:x1], aug_flux[x2:r0]]
        if neigh.size == 0:
            med = float(np.median(aug_flux))
            resid = aug_flux - med
        else:
            med = float(np.median(neigh))
            resid = neigh - med
        mad = float(np.median(np.abs(resid))) if resid.size > 0 else 0.0
        sigma = 1.4826 * mad if mad > 0 else float(np.std(resid)) if resid.size > 0 else float(np.std(aug_flux))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1e-6
        # Fill with local median plus Gaussian noise.
        aug_flux[x1:x2] = np.random.normal(loc=med, scale=sigma, size=(x2 - x1,))
    return aug_flux

def method_blur(flux, sigma):
    """
    Gaussian blur for resolution/seeing variation.
    """
    return gaussian_filter1d(flux, sigma=sigma)

def method_continuum_distortion(flux):
    """
    Smooth continuum distortion.
    """
    length = len(flux)
    x = np.linspace(0, 1, length)
    # Generate a low-frequency sinusoidal distortion.
    factor = 1.0 + 0.1 * np.sin(x * np.pi * np.random.uniform(0.5, 2.0) + np.random.uniform(0, np.pi))
    return flux * factor

# ================= Main logic =================

def _resolve_fold_splits():
    """
    Return split directories such as root/fold_1/train.
    """
    root = FOLD_DATASET_ROOT
    if not os.path.isdir(root):
        print(f"[Error] fold dataset root does not exist: {root}")
        return []

    split_name = str(FOLD_SPLIT_NAME).strip()
    if not split_name:
        print("[Error] split name is empty")
        return []

    dirs = []
    if FOLD_NAMES == "all":
        fold_names = sorted([d for d in os.listdir(root) if d.startswith("fold_") and os.path.isdir(os.path.join(root, d))])
        for fn in fold_names:
            dirs.append(os.path.join(root, fn, split_name))
    else:
        try:
            fold_ids = list(FOLD_NAMES)
        except Exception:
            print("[Error] folds must be 'all' or a list such as [1,2,3]")
            return []
        for fid in fold_ids:
            dirs.append(os.path.join(root, f"fold_{int(fid)}", split_name))
    return dirs


def _augment_one_fold_split(fold_split_dir: str):
    # Basic path and CSV checks.
    spec_dir = os.path.join(fold_split_dir, "spectrum")
    label_dir = os.path.join(fold_split_dir, "label")
    label_path = os.path.join(label_dir, "label.csv")
    if not (os.path.isdir(spec_dir) and os.path.isfile(label_path)):
        print(f"[Warn] Skip missing split directory or label.csv: {fold_split_dir}")
        return

    labels_df = pd.read_csv(label_path)
    if "basename" not in labels_df.columns or "label" not in labels_df.columns:
        print(f"[Warn] Skip label.csv without required columns 'basename' and 'label': {label_path}")
        return

    pos_df = labels_df[labels_df["label"] == EA_LABEL].copy()
    neg_df = labels_df[labels_df["label"] != EA_LABEL].copy()
    normal_basenames = neg_df["basename"].astype(str).tolist()

    new_label_rows = []
    generated_count = 0
    existing_basenames = set(labels_df["basename"].astype(str).tolist())

    # Strategy pool and sampling weights.
    base_strategies = ['noise', 'mixup', 'shift', 'blur', 'cutout', 'distortion']
    base_weights    = [ 0.20,     0.25,    0.15,    0.15,   0.15,      0.10]
    # Restrict strategies if requested by orchestration code.
    if isinstance(ALLOWED_STRATEGIES, list) and len(ALLOWED_STRATEGIES) > 0:
        pairs = [(s, w) for s, w in zip(base_strategies, base_weights) if s in ALLOWED_STRATEGIES]
        if pairs:
            base_strategies, base_weights = zip(*pairs)
            base_strategies, base_weights = list(base_strategies), list(base_weights)
    if not ALLOW_SHIFT:
        # Remove shift when disabled.
        pairs = [(s, w) for s, w in zip(base_strategies, base_weights) if s != 'shift']
        strategies, weights = zip(*pairs)
        weights = list(weights)
    else:
        strategies, weights = base_strategies, base_weights

    fold_tag = os.path.basename(os.path.dirname(fold_split_dir))
    split_tag = os.path.basename(fold_split_dir)
    desc = f"augment {fold_tag}/{split_tag}"
    for _, row in tqdm(pos_df.iterrows(), total=len(pos_df), desc=desc):
        base = str(row["basename"])
        src_csv = os.path.join(spec_dir, f"{base}.csv")
        wave_orig, flux_orig = read_wave_flux_from_csv(src_csv)
        if wave_orig is None or flux_orig is None:
            continue

        # Compute generated count from fractional expected ratio or integer ratio.
        loops = int(max(0, int(AUGMENT_RATIO)))
        try:
            if AUGMENT_EXPECTED is not None:
                expv = float(AUGMENT_EXPECTED)
                if expv > 0:
                    base = int(math.floor(expv))
                    frac = float(expv - base)
                    extra = 1 if (random.random() < frac) else 0
                    loops = base + extra
        except Exception:
            pass
        for _i in range(loops):
            # Sample one augmentation strategy.
            strategy = random.choices(strategies, weights=weights, k=1)[0]
            
            new_flux = flux_orig.copy()
            suffix = ""

            # Use one primary augmentation, with optional additional noise.
            
            if strategy == 'mixup' and len(normal_basenames) > 0:
                rand_normal = random.choice(normal_basenames)
                normal_csv = os.path.join(spec_dir, f"{rand_normal}.csv")
                n_wave, n_flux = read_wave_flux_from_csv(normal_csv)
                if n_flux is not None and n_flux.shape == flux_orig.shape:
                    alpha = random.uniform(*MIXUP_ALPHA_RANGE)
                    new_flux = method_mixup(new_flux, n_flux, alpha)
                    suffix = f"_mix"
                else:
                    # Fallback when mixup partner shape is incompatible.
                    strategy = 'shift' if ALLOW_SHIFT else ('blur' if random.random() < 0.5 else 'noise')

            if strategy == 'shift' and ALLOW_SHIFT:
                shift = random.choice(SHIFT_PIXELS)
                scale = random.uniform(*SCALE_RANGE)
                # new_flux = method_shift_scale(new_flux, shift, scale)
                new_flux = method_shift_scale_with_wave(wave_orig, new_flux, shift, scale)
                suffix = f"_sh"
            elif strategy == 'shift' and not ALLOW_SHIFT:
                # Replace disabled shift with blur or noise.
                if random.random() < 0.5:
                    sigma = random.uniform(*BLUR_SIGMA_RANGE)
                    new_flux = method_blur(new_flux, sigma)
                    suffix = f"_bl"
                else:
                    target_snr = random.choice(TARGET_SNRS)
                    new_flux = method_noise_injection(new_flux, target_snr)
                    suffix = f"_ns"

            if strategy == 'blur':
                sigma = random.uniform(*BLUR_SIGMA_RANGE)
                new_flux = method_blur(new_flux, sigma)
                suffix = f"_bl"

            if strategy == 'cutout':
                new_flux = method_cutout(new_flux, CUTOUT_NUM_HOLES, CUTOUT_MAX_WIDTH)
                suffix = f"_cut"
            
            if strategy == 'distortion':
                new_flux = method_continuum_distortion(new_flux)
                suffix = f"_dis"

            # Optionally add final noise.
            noise_prob = 0.5 if (FORCE_NOISE_PROB is None) else float(FORCE_NOISE_PROB)
            if noise_prob > 0 and (random.random() < noise_prob):
                target_snr = random.choice(TARGET_SNRS)
                new_flux = method_noise_injection(new_flux, target_snr)
                suffix += f"_ns"

            # Save augmented spectrum and label row.
            name_prefix = f"{base}{suffix}"
            idx = 0
            new_basename = f"{name_prefix}_{idx}"
            while new_basename in existing_basenames or os.path.exists(os.path.join(spec_dir, f"{new_basename}.csv")):
                idx += 1
                new_basename = f"{name_prefix}_{idx}"
            existing_basenames.add(new_basename)

            out_csv = os.path.join(spec_dir, f"{new_basename}.csv")
            arr = np.column_stack((wave_orig, new_flux))
            np.savetxt(out_csv, arr, delimiter=",", fmt="%.6f")

            item = {"basename": new_basename, "label": EA_LABEL}
            if "z" in labels_df.columns:
                item["z"] = row.get("z", 0)
            new_label_rows.append(item)
            generated_count += 1

    # Save updated label table.
    if new_label_rows:
        labels_df = pd.concat([labels_df, pd.DataFrame(new_label_rows)], ignore_index=True)
        labels_df.to_csv(label_path, index=False)
    print(f"[Info] {fold_tag}/{split_tag} augmentation complete. generated={generated_count}")


def _parse_folds_arg(raw_folds):
    if not raw_folds or raw_folds == ["all"]:
        return "all"
    out = []
    for item in raw_folds:
        text = str(item).strip()
        if not text:
            continue
        if text.lower() == "all":
            return "all"
        if text.startswith("fold_"):
            text = text.split("_")[-1]
        out.append(int(text))
    return out or "all"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run positive-class conventional augmentation on fold-format spectra."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=FOLD_DATASET_ROOT,
        help="Fold root containing fold_*/train/spectrum and fold_*/train/label/label.csv.",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        default=["all"],
        help='Fold ids/names to process, e.g. "--folds 1 2" or "--folds fold_1"; default: all.',
    )
    parser.add_argument("--split", type=str, default=FOLD_SPLIT_NAME, help="Split to augment; default: train.")
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=float(AUGMENT_RATIO),
        help="Expected generated positives per real positive spectrum.",
    )
    parser.add_argument("--positive-label", type=int, default=int(EA_LABEL), help="Positive-class label to augment.")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        choices=["noise", "mixup", "blur", "cutout", "distortion", "shift"],
        help="Optional subset of augmentation strategies.",
    )
    parser.add_argument("--allow-shift", action="store_true", help="Allow wavelength/pixel shift augmentation.")
    parser.add_argument("--noise-prob", type=float, default=None, help="Override final noise probability.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    global FOLD_DATASET_ROOT, FOLD_NAMES, FOLD_SPLIT_NAME
    global AUGMENT_RATIO, AUGMENT_EXPECTED, EA_LABEL, ALLOW_SHIFT
    global ALLOWED_STRATEGIES, FORCE_NOISE_PROB, RANDOM_SEED

    args = parse_args(argv)
    FOLD_DATASET_ROOT = args.dataset_root
    FOLD_NAMES = _parse_folds_arg(args.folds)
    FOLD_SPLIT_NAME = args.split
    AUGMENT_RATIO = float(args.augment_ratio)
    AUGMENT_EXPECTED = float(args.augment_ratio)
    EA_LABEL = int(args.positive_label)
    ALLOW_SHIFT = bool(args.allow_shift)
    ALLOWED_STRATEGIES = list(args.strategies) if args.strategies else None
    FORCE_NOISE_PROB = args.noise_prob
    RANDOM_SEED = int(args.seed)

    split_dirs = _resolve_fold_splits()
    if not split_dirs:
        return

    print(f"[Info] Processing {len(split_dirs)} split(s): {split_dirs[:3]}{' ...' if len(split_dirs) > 3 else ''}")
    for d in split_dirs:
        # Fold-dependent deterministic seed offset.
        fold_name = os.path.basename(os.path.dirname(d))
        offset = sum([ord(c) for c in fold_name]) % 10_000
        random.seed(RANDOM_SEED + offset)
        np.random.seed(RANDOM_SEED + offset)
        _augment_one_fold_split(d)

if __name__ == "__main__":
    main()
