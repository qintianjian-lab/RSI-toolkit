#!/usr/bin/env python3
"""
Run repeated fold-level training/evaluation jobs and aggregate metrics.

This script launches `train.py`, evaluates the selected checkpoint with
`custom_test.py`, and writes per-fold/per-run metrics plus mean/std summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], env: dict[str, str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_fh.write(line)
            output_lines.append(line)
        proc.wait()
    output = "".join(output_lines)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return output


def _parse_best_model_path(stdout_text: str) -> str:
    match = re.search(r"\[Info\]\s*best model path:\s*(.+)", stdout_text)
    if not match:
        raise ValueError("Could not parse best checkpoint path from train.py output.")
    path = match.group(1).strip()
    if not path:
        raise ValueError("train.py printed an empty best checkpoint path.")
    return path


def _flatten(prefix: str, obj: Any, out: dict[str, float]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_key = f"{prefix}_{key}" if prefix else str(key)
            _flatten(next_key, value, out)
        return
    if isinstance(obj, list):
        return
    if isinstance(obj, (int, float)) and obj == obj:
        out[prefix] = float(obj)


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics_summary.json: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    flat: dict[str, Any] = {}
    _flatten("", data, flat)

    topk = data.get("topk", {}) if isinstance(data, dict) else {}
    if isinstance(topk, dict):
        total_pos = topk.get("total_pos")
        if isinstance(total_pos, (int, float)):
            flat["topk_total_pos"] = float(total_pos)
        for item in topk.get("items", []) or []:
            if not isinstance(item, dict) or "k" not in item:
                continue
            try:
                k = int(item["k"])
            except Exception:
                continue
            for source_key, target_prefix in [
                ("precision_at_k", "topk_precision_at"),
                ("recall_at_k", "topk_recall_at"),
                ("tp", "topk_tp_at"),
            ]:
                value = item.get(source_key)
                if isinstance(value, (int, float)):
                    flat[f"{target_prefix}_{k}"] = float(value)
    return flat


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def _format_metric(summary: dict[str, Any], key: str) -> str:
    item = summary.get(key)
    if isinstance(item, dict) and "mean" in item and "std" in item:
        return f"{item['mean']:.6f} +/- {item['std']:.6f}"
    return "N/A"


def _write_summary(session_dir: Path, summary: dict[str, Any]) -> None:
    with (session_dir / "aggregate_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    lines = [
        "RSI repeat summary",
        f"session_dir: {session_dir}",
        f"expected runs: {summary['num_runs_expected']}",
        f"successful runs: {summary['num_runs_success']}",
        f"failed runs: {summary['num_runs_failed']}",
        "",
        "Core metrics (mean +/- std)",
        f"auprc: {_format_metric(summary, 'auprc')}",
        f"auroc: {_format_metric(summary, 'auroc')}",
        f"calibrated best-f1 macro: {_format_metric(summary, 'calibrated_on_test_best_f1_f1_macro')}",
        f"best-f1 macro: {_format_metric(summary, 'best_f1_f1_macro')}",
        f"fixed-threshold macro F1: {_format_metric(summary, 'fixed_threshold_f1_macro')}",
    ]

    topk_keys = sorted(k for k in summary if k.startswith("topk_precision_at_"))
    if topk_keys:
        lines.extend(["", "Top-K metrics"])
        for key in topk_keys:
            k = key.removeprefix("topk_precision_at_")
            lines.append(
                f"K={k}: precision={_format_metric(summary, key)}, "
                f"recall={_format_metric(summary, f'topk_recall_at_{k}')}, "
                f"tp={_format_metric(summary, f'topk_tp_at_{k}')}"
            )

    if summary["failed_runs"]:
        lines.extend(["", "Failed runs"])
        for item in summary["failed_runs"]:
            lines.append(f"- fold={item.get('fold')} run={item.get('run')} error={item.get('error')}")

    with (session_dir / "summary.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_project_dataset_dir() -> str:
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from config.config import config as project_config  # type: ignore

        return str(project_config.get("dataset_dir", ""))
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeat RSI fold training/evaluation and aggregate metrics.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeats per fold.")
    parser.add_argument("--used_model", type=str, required=True)
    parser.add_argument("--cross_name", type=str, default="", help="Single fold name.")
    parser.add_argument(
        "--cross_names",
        type=str,
        default="",
        help="Comma-separated fold names. Overrides --cross_name when non-empty. Defaults to fold_1,...,fold_5 when neither option is set.",
    )
    parser.add_argument("--dataset_dir", type=str, default="", help="Override config['dataset_dir'] for train.py.")
    parser.add_argument("--spectrum_size", type=int, default=-1, help="Override config['spectrum_size'] for train.py.")
    parser.add_argument("--test_data_dir", type=str, default="", help="Shared test directory when not using --use_fold_test.")
    parser.add_argument(
        "--use_fold_test",
        action="store_true",
        help="Evaluate each fold at <dataset_dir>/<fold>/test.",
    )
    parser.add_argument("--spectrum_dir", type=str, default="spectrum")
    parser.add_argument("--label_file", type=str, default="label/label.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--target_recall", type=float, default=0.90)
    parser.add_argument("--target_precision", type=float, default=0.75)
    parser.add_argument("--topk", type=int, nargs="*", default=[])

    parser.add_argument(
        "--train_strategy",
        choices=["direct", "two_stage_head", "two_stage_head_pool"],
        default="direct",
    )
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--reset_head", action="store_true")
    parser.add_argument("--epochs_stage1", type=int, default=-1)
    parser.add_argument("--epochs_stage2", type=int, default=-1)
    parser.add_argument("--lr_stage1", type=float, default=-1.0)
    parser.add_argument("--lr_stage2", type=float, default=-1.0)
    parser.add_argument("--weight_decay", type=float, default=-1.0)
    parser.add_argument("--monitor_metric", type=str, default="")
    parser.add_argument("--monitor_mode", type=str, default="")

    parser.add_argument("--device_ids", type=str, default="0", help='Comma-separated GPU ids, e.g. "0,1,2,3".')
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="runs/repeat")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cross_names.strip():
        folds = _split_csv(args.cross_names)
    elif args.cross_name.strip():
        folds = [args.cross_name.strip()]
    else:
        folds = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    if args.runs <= 0:
        parser.error("--runs must be positive.")
    if args.train_strategy != "direct" and not args.pretrained_ckpt.strip():
        parser.error("--pretrained_ckpt is required for two-stage training strategies.")
    if not args.use_fold_test and not args.test_data_dir.strip():
        parser.error("Provide --test_data_dir or set --use_fold_test.")

    dataset_dir = args.dataset_dir.strip() or _load_project_dataset_dir()
    if args.use_fold_test and not dataset_dir:
        parser.error("--use_fold_test requires --dataset_dir or config['dataset_dir'].")

    device_ids = _split_csv(args.device_ids) or ["0"]
    max_workers = max(1, min(args.max_parallel, len(device_ids)))

    session_name = f"{args.used_model}_{'multi_folds' if len(folds) > 1 else folds[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = (PROJECT_ROOT / args.save_dir / session_name).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / "repeat_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2)

    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu_id in device_ids:
        gpu_queue.put(gpu_id)

    def test_dir_for_fold(fold: str) -> str:
        if args.use_fold_test:
            return str(Path(dataset_dir) / fold / "test")
        return args.test_data_dir

    def train_cmd(fold: str, pretrained: str, freeze_backbone: bool, train_attnpool: bool,
                  reset_head: bool, epochs: int, lr: float) -> list[str]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--used_model",
            args.used_model,
            "--cross_name",
            fold,
            "--skip_test",
        ]
        if pretrained:
            cmd.extend(["--pretrained_ckpt", pretrained])
        if freeze_backbone:
            cmd.append("--freeze_backbone")
        if train_attnpool:
            cmd.append("--train_attnpool")
        if reset_head:
            cmd.append("--reset_head")
        if args.strict_load:
            cmd.append("--strict_load")
        if dataset_dir:
            cmd.extend(["--dataset_dir", dataset_dir])
        if args.spectrum_size > 0:
            cmd.extend(["--spectrum_size", str(args.spectrum_size)])
        if epochs > 0:
            cmd.extend(["--epochs", str(epochs)])
        if lr > 0:
            cmd.extend(["--learn_rate", str(lr)])
        if args.weight_decay >= 0:
            cmd.extend(["--weight_decay", str(args.weight_decay)])
        if args.monitor_metric:
            cmd.extend(["--monitor_metric", args.monitor_metric])
        if args.monitor_mode:
            cmd.extend(["--monitor_mode", args.monitor_mode])
        return cmd

    def run_one(fold: str, run_idx: int) -> dict[str, Any]:
        gpu_id = gpu_queue.get()
        run_dir = session_dir / fold / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        try:
            print(f"\n========== fold={fold} run={run_idx}/{args.runs} gpu={gpu_id} ==========")
            if args.train_strategy == "direct":
                stdout = _run(
                    train_cmd(
                        fold,
                        args.pretrained_ckpt,
                        freeze_backbone=False,
                        train_attnpool=False,
                        reset_head=args.reset_head,
                        epochs=args.epochs_stage2,
                        lr=args.lr_stage2,
                    ),
                    env,
                    run_dir / "train.log",
                )
                best_ckpt = _parse_best_model_path(stdout)
            else:
                stage1_stdout = _run(
                    train_cmd(
                        fold,
                        args.pretrained_ckpt,
                        freeze_backbone=True,
                        train_attnpool=args.train_strategy == "two_stage_head_pool",
                        reset_head=True,
                        epochs=args.epochs_stage1,
                        lr=args.lr_stage1,
                    ),
                    env,
                    run_dir / "train_stage1.log",
                )
                stage1_ckpt = _parse_best_model_path(stage1_stdout)
                stage2_stdout = _run(
                    train_cmd(
                        fold,
                        stage1_ckpt,
                        freeze_backbone=False,
                        train_attnpool=False,
                        reset_head=False,
                        epochs=args.epochs_stage2,
                        lr=args.lr_stage2,
                    ),
                    env,
                    run_dir / "train_stage2.log",
                )
                best_ckpt = _parse_best_model_path(stage2_stdout)

            eval_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "custom_test.py"),
                "--test_data_dir",
                test_dir_for_fold(fold),
                "--spectrum_dir",
                args.spectrum_dir,
                "--label_file",
                args.label_file,
                "--output_dir",
                str(run_dir),
                "--batch_size",
                str(args.batch_size),
                "--threshold",
                str(args.threshold),
                "--temperature",
                str(args.temperature),
                "--target_recall",
                str(args.target_recall),
                "--target_precision",
                str(args.target_precision),
                "--device",
                "auto",
            ]
            eval_cmd.extend(["--model_path", best_ckpt])
            if args.topk:
                eval_cmd.extend(["--topk", *[str(k) for k in args.topk]])
            _run(eval_cmd, env, run_dir / "evaluate.log")

            metrics = _read_metrics(run_dir)
            metrics.update({"fold": fold, "run": run_idx, "gpu_id": gpu_id, "checkpoint": best_ckpt})
            with (run_dir / "run_record.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, ensure_ascii=False, indent=2)
            return metrics
        except Exception as exc:
            error = {
                "fold": fold,
                "run": run_idx,
                "gpu_id": gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            with (run_dir / "error.json").open("w", encoding="utf-8") as fh:
                json.dump(error, fh, ensure_ascii=False, indent=2)
            print(f"[FAILED] fold={fold} run={run_idx}: {exc}")
            return {"_failed": True, **error}
        finally:
            gpu_queue.put(gpu_id)

    futures = []
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fold in folds:
            for run_idx in range(1, args.runs + 1):
                futures.append(executor.submit(run_one, fold, run_idx))
        for future in as_completed(futures):
            item = future.result()
            if item.get("_failed"):
                failed.append(item)
            else:
                results.append(item)

    numeric_keys: set[str] = set()
    for item in results:
        for key, value in item.items():
            if key not in {"run"} and isinstance(value, (int, float)):
                numeric_keys.add(key)

    summary: dict[str, Any] = {
        key: _mean_std([float(item[key]) for item in results if isinstance(item.get(key), (int, float))])
        for key in sorted(numeric_keys)
    }
    summary.update(
        {
            "num_runs_expected": len(folds) * args.runs,
            "num_runs_success": len(results),
            "num_runs_failed": len(failed),
            "runs_detail": results,
            "failed_runs": failed,
        }
    )
    _write_summary(session_dir, summary)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
