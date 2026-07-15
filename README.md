# RSI Toolkit

Code for Rare-object Spectral Identification (RSI), a framework for rare-target
identification in one-dimensional galaxy spectra.

## Installation

From this directory:

```bash
pip install -r requirements.txt
```

The GANDALF-inspired generator additionally needs TensorFlow:

```bash
pip install -r requirements-gandalf.txt
```

This repository is intended to run from the source tree:

```bash
python -m rsi --help
```

Use `python -m rsi <command> --help` to view the full argument list for a
specific command.

## Expected Data

### Fold-Format Dataset

The fold-format dataset is used for downstream classifier training, repeated
fold evaluation, positive-sample augmentation, and cross-domain supervised
pre-training when source-domain spectra are provided in the same layout.

```text
data/lamost_folds/
  fold_1/
    train/spectrum/*.csv
    train/label/label.csv
    val/spectrum/*.csv
    val/label/label.csv
    test/spectrum/*.csv
    test/label/label.csv
```

Each spectrum CSV should contain two columns:

```text
wavelength,flux
```

Each `label.csv` should contain at least:

```text
basename,label
```

An optional `z` column is used by models or scripts that accept redshift
metadata.

Note: Conventional augmentation should be applied to fold-format spectra on a
common wavelength grid before final amplitude normalization such as z-score
scaling. After augmentation, apply the same final preprocessing choices used for
downstream training and evaluation. Other workflows assume the fold-format
dataset already uses the final preprocessing adopted for training and
evaluation.

### In-Domain Masked-Reconstruction Pre-training

In-domain masked-reconstruction pre-training uses a separate cache-index format:

```text
data/mae_cache/
  train_index.csv
  val_index.csv
  npy/*.npy
```

```text
basename,npy_path
```

Unlike downstream classifier training, `pretrain-mae` reads train and validation
index CSV files rather than fold directories. Each `npy_path` points to a
NumPy `.npy` file containing one preprocessed one-dimensional flux array,
typically length 4000, and should be readable from the directory where the
command is run. Set `RSI_MAE_TRAIN_INDEX` and `RSI_MAE_VAL_INDEX` to the train
and validation cache-index CSV files when running `pretrain-mae`.

## Workflow Commands

Typical workflow:

```text
prepare fold-format data
  -> optional augmentation and/or pre-training
  -> train
  -> evaluate or repeat
  -> mine
```

| Workflow step | Command |
| --- | --- |
| Conventional positive-sample augmentation | `augment-conventional` |
| VAE generative augmentation | `vae-dir-to-npy` -> `vae-train` -> `vae-sample` -> `generative-npy-to-dir` |
| CVAE generative augmentation | `cvae-dir-to-npy` -> `cvae-train` -> `cvae-sample` -> `generative-npy-to-dir` |
| GANDALF generative augmentation | `gandalf-prepare` -> `gandalf-train` -> `gandalf-sample` -> `gandalf-npy-to-dir` |
| In-domain masked-reconstruction pre-training | `pretrain-mae` |
| Cross-domain supervised pre-training | `train` on source-domain fold data, then downstream `train` with `--pretrained_ckpt` |
| Downstream classifier training | `train` |
| Checkpoint evaluation | `evaluate` |
| Repeated fold training/evaluation | `repeat` |
| Archive scoring | `mine` |
