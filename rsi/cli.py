from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandSpec:
    script: str
    description: str


COMMANDS: dict[str, CommandSpec] = {
    # Pre-training and augmentation on fold-format data.
    "pretrain-mae": CommandSpec(
        "rsi/paper_steps/pretraining/pretrain_sbm_universal_mae.py",
        "Run SBM in-domain masked-reconstruction pre-training.",
    ),
    "augment-conventional": CommandSpec(
        "rsi/paper_steps/augmentation/augment_data_v2.py",
        "Run train-fold conventional augmentation on fold-format spectra.",
    ),
    # Downstream classifier and archive scoring.
    "train": CommandSpec(
        "train.py",
        "Train a downstream classifier.",
    ),
    "repeat": CommandSpec(
        "rsi/paper_steps/repeat_runs.py",
        "Run repeated fold training/evaluation and aggregate metrics.",
    ),
    "evaluate": CommandSpec(
        "custom_test.py",
        "Evaluate a checkpoint with threshold and top-K metrics.",
    ),
    "mine": CommandSpec(
        "rsi/paper_steps/mining/batch_predict.py",
        "Score an archive table with a trained classifier.",
    ),
    # Generative augmentation helpers.
    "vae-dir-to-npy": CommandSpec(
        "rsi/paper_steps/generative/vae_cvae/dir_to_npy.py",
        "Convert classifier fold data to VAE NPY arrays.",
    ),
    "vae-train": CommandSpec(
        "rsi/paper_steps/generative/vae_cvae/train_vae.py",
        "Train a VAE generator on fold-local training data.",
    ),
    "vae-sample": CommandSpec(
        "rsi/paper_steps/generative/vae_cvae/sample_vae.py",
        "Sample spectra from a trained VAE generator.",
    ),
    "cvae-dir-to-npy": CommandSpec(
        "rsi/paper_steps/generative/vae_cvae/dir_to_npy_cvae.py",
        "Convert classifier fold data to CVAE NPY arrays.",
    ),
    "cvae-train": CommandSpec(
        "rsi/paper_steps/generative/vae_cvae/train_cvae.py",
        "Train a CVAE generator on fold-local training data.",
    ),
    "cvae-sample": CommandSpec(
        "rsi/paper_steps/generative/vae_cvae/sample_cvae.py",
        "Sample label-conditioned spectra from a trained CVAE generator.",
    ),
    "generative-npy-to-dir": CommandSpec(
        "rsi/paper_steps/generative/vae_cvae/npy_to_dir.py",
        "Write generated VAE/CVAE spectra back into classifier fold format.",
    ),
    "gandalf-prepare": CommandSpec(
        "rsi/paper_steps/generative/gandalf/prepare_from_dir.py",
        "Convert classifier fold data to GANDALF-style arrays.",
    ),
    "gandalf-train": CommandSpec(
        "rsi/paper_steps/generative/gandalf/gandalf/train/train_cli.py",
        "Train the GANDALF-inspired conditional autoencoder branch.",
    ),
    "gandalf-sample": CommandSpec(
        "rsi/paper_steps/generative/gandalf/sample_decoder.py",
        "Sample spectra from a trained GANDALF decoder.",
    ),
    "gandalf-npy-to-dir": CommandSpec(
        "rsi/paper_steps/generative/gandalf/npy_to_dir.py",
        "Write generated GANDALF spectra back into classifier fold format.",
    ),
}


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT)]
    old = env.get("PYTHONPATH")
    if old:
        paths.append(old)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _print_help() -> None:
    print("RSI toolkit")
    print("")
    print("Usage:")
    print("  python -m rsi <command> [script arguments]")
    print("")
    print("RSI workflow commands:")
    for name, spec in COMMANDS.items():
        print(f"  {name:<24} {spec.description}")
    print("")
    print("Examples:")
    print("  python -m rsi pretrain-mae --help")
    print("  python -m rsi augment-conventional --help")
    print("  python -m rsi train --used_model sbm_universal_v2 --cross_name fold_1")
    print("  python -m rsi repeat --used_model sbm_universal_v2 --dataset_dir data/lamost_folds --use_fold_test")
    print("  python -m rsi evaluate --test_data_dir data/lamost_folds/fold_1/test --spectrum_dir spectrum --label_file label/label.csv --model_path runs/checkpoints/best.ckpt")
    print("  python -m rsi mine --checkpoint runs/checkpoints/best.ckpt --input-csv data/archive.csv --csv-spectra-root data/archive_spectra --output-csv runs/scores.csv")


def run_script(command: str, args: Sequence[str]) -> int:
    spec = COMMANDS[command]
    script = REPO_ROOT / spec.script
    if not script.is_file():
        raise FileNotFoundError(f"Command script not found: {script}")
    cmd = [sys.executable, str(script), *args]
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=_build_env())


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    command = argv[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run `python -m rsi --help` to list available commands.", file=sys.stderr)
        return 2
    return run_script(command, argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
