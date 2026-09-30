#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
from pathlib import Path


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _set_global_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _format_command_line() -> str:
    return " ".join(shlex.quote(str(arg)) for arg in sys.argv)


def _write_outputs(args: argparse.Namespace, rows: list[dict], summary: dict) -> None:
    command_line = str(summary.get("command_line", _format_command_line()))
    if args.output_csv:
        path = Path(args.output_csv).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "suite",
                    "dataset",
                    "accuracy",
                    "n_train",
                    "n_test",
                    "n_classes",
                    "effective_v2_batch_size",
                    "effective_signal_encoder_batch_size",
                    "effective_task_level_chorus_ensemble_batch_size",
                    "status",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Eval] wrote CSV: {path}")

        cmd_path = Path(str(path) + ".cmd.txt")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(command_line + "\n")
        print(f"[Eval] wrote command: {cmd_path}")

    if args.output_json:
        path = Path(args.output_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": rows}, f, ensure_ascii=False, indent=2)
        print(f"[Eval] wrote JSON: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ChorusTIC on UCR/UEA with training-free in-context classification."
    )
    parser.add_argument(
        "--rssc_ckpt",
        required=True,
        help="ChorusTIC checkpoint containing signal-level Chorus and Task-level Chorus tensors.",
    )
    parser.add_argument("--rssc_hparams_json", default=None, help="Defaults to model_hparams_latest.json near --rssc_ckpt.")
    parser.add_argument(
        "--task_level_chorus_ckpt",
        default=None,
        help="Optional checkpoint providing Task-level Chorus weights.",
    )
    parser.add_argument(
        "--task_level_chorus_hparams_json",
        default=None,
        help="Optional hparams JSON for --task_level_chorus_ckpt.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ucr_path", default=None)
    parser.add_argument("--uea_path", default=None)
    parser.add_argument("--suite", choices=["ucr", "uea", "both"], default="both")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", choices=["classifier_v2"], default="classifier_v2")

    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--v2_n_augmentations", type=int, default=2)
    parser.add_argument("--v2_crop_rate_lo", type=float, default=0.0)
    parser.add_argument("--v2_crop_rate_hi", type=float, default=0.15)
    parser.add_argument("--v2_batch_size", type=int, default=8)
    parser.add_argument("--v2_class_shift", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--v2_context_mode",
        choices=["full", "stratified_random", "class_proto_kcenter"],
        default="full",
        help="Context selection mode for classifier_v2.",
    )
    parser.add_argument(
        "--v2_context_size",
        type=int,
        default=0,
        help="Maximum context size. If <=0 or >= train size, use the full training split.",
    )
    parser.add_argument("--v2_context_seed", type=int, default=None)
    parser.add_argument("--v2_context_allocation", choices=["balanced", "sqrt"], default="balanced")
    parser.add_argument("--v2_context_proto_ratio", type=float, default=0.5)
    parser.add_argument("--softmax_temperature", type=float, default=0.9)

    parser.add_argument("--signal_encoder_batch_size", type=int, default=None)
    parser.add_argument("--rssc_eval_ensembles", type=int, default=8)
    parser.add_argument("--rssc_encoder_ensemble_batch_size", type=int, default=1)
    parser.add_argument("--task_level_chorus_ensemble_batch_size", type=int, default=64)
    parser.add_argument("--oom_retry", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--oom_min_v2_batch_size", type=int, default=1)
    parser.add_argument("--oom_min_signal_encoder_batch_size", type=int, default=1)
    parser.add_argument("--oom_min_task_level_chorus_ensemble_batch_size", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uea_mode", choices=["center", "random"], default="center")
    parser.add_argument("--uea_fusion", choices=["sum_embed", "concat"], default="sum_embed")
    parser.add_argument("--uea_use_var_selector", action="store_true")
    parser.add_argument("--uea_var_num_channels", type=int, default=None)

    parser.add_argument("--strict_rssc_encoder", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--strict_task_level_chorus", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset and args.suite == "both":
        raise ValueError("--dataset requires --suite ucr or --suite uea, not --suite both.")
    if args.suite in {"ucr", "both"} and not args.ucr_path:
        raise ValueError("--ucr_path is required when --suite is ucr or both.")
    if args.suite in {"uea", "both"} and not args.uea_path:
        raise ValueError("--uea_path is required when --suite is uea or both.")
    if int(args.rssc_eval_ensembles) <= 0:
        raise ValueError("--rssc_eval_ensembles must be > 0.")
    if int(args.v2_batch_size) <= 0:
        raise ValueError("--v2_batch_size must be > 0.")
    if int(args.rssc_encoder_ensemble_batch_size) <= 0:
        raise ValueError("--rssc_encoder_ensemble_batch_size must be > 0.")
    if int(args.task_level_chorus_ensemble_batch_size) <= 0:
        raise ValueError("--task_level_chorus_ensemble_batch_size must be > 0.")
    if int(args.oom_min_task_level_chorus_ensemble_batch_size) <= 0:
        raise ValueError("--oom_min_task_level_chorus_ensemble_batch_size must be > 0.")
    if not (0.0 <= float(args.v2_context_proto_ratio) <= 1.0):
        raise ValueError("--v2_context_proto_ratio must be in [0, 1].")

    crop_lo = float(args.v2_crop_rate_lo)
    crop_hi = float(args.v2_crop_rate_hi)
    if not (0.0 <= crop_lo <= crop_hi < 1.0):
        raise ValueError(f"Invalid crop range: {crop_lo}, {crop_hi}")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)

    import numpy as np
    import torch

    from chorustic.evaluation.datasets import dataset_names, prepare_ucr_uea_arrays
    from chorustic.evaluation.inference import predict_classifier_v2_with_oom_retry
    from chorustic.model import build_chorustic_from_checkpoint, load_json, resolve_hparams_path
    from chorustic.evaluation.data_reader import DataReader

    _set_global_seed(int(args.seed))

    device = torch.device(args.device)
    rssc_hparams_path = resolve_hparams_path(args.rssc_ckpt, args.rssc_hparams_json)
    rssc_hparams = load_json(rssc_hparams_path)
    task_level_chorus_hparams = (
        load_json(args.task_level_chorus_hparams_json) if args.task_level_chorus_hparams_json else None
    )

    model, model_info = build_chorustic_from_checkpoint(
        rssc_ckpt=str(args.rssc_ckpt),
        rssc_hparams=rssc_hparams,
        task_level_chorus_ckpt=(
            None if args.task_level_chorus_ckpt is None else str(args.task_level_chorus_ckpt)
        ),
        task_level_chorus_hparams=task_level_chorus_hparams,
        device=device,
        signal_encoder_batch_size=args.signal_encoder_batch_size,
        rssc_eval_ensembles=int(args.rssc_eval_ensembles),
        rssc_encoder_ensemble_batch_size=int(args.rssc_encoder_ensemble_batch_size),
        task_level_chorus_ensemble_batch_size=int(args.task_level_chorus_ensemble_batch_size),
        strict_rssc_encoder=bool(args.strict_rssc_encoder),
        strict_task_level_chorus=bool(args.strict_task_level_chorus),
    )

    print("[Eval][ModelLoad]")
    print(json.dumps(model_info, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    print(f"[Eval] RSSC ckpt: {args.rssc_ckpt}")
    print(f"[Eval] RSSC hparams: {rssc_hparams_path}")
    task_ckpt = args.task_level_chorus_ckpt if args.task_level_chorus_ckpt else args.rssc_ckpt
    print(f"[Eval] Task-level Chorus ckpt: {task_ckpt}")
    if args.task_level_chorus_hparams_json:
        print(f"[Eval] Task-level Chorus hparams: {args.task_level_chorus_hparams_json}")
    print(f"[Eval] suite={args.suite} mode={args.mode} device={device}")

    reader = DataReader(
        UEA_data_path=str(args.uea_path or ""),
        UCR_data_path=str(args.ucr_path or ""),
        transform_ts_size=int(model_info["mantis_seq_len"]),
    )
    jobs = dataset_names(reader, args.suite, args.dataset)
    print(f"[Eval] dataset_count={len(jobs)}")

    base_v2_batch_size = int(args.v2_batch_size)
    base_signal_encoder_batch_size = int(
        getattr(model.pre_mantis_encoder, "mantis_batch_size", args.signal_encoder_batch_size or 16)
    )
    base_task_level_chorus_batch_size = int(args.task_level_chorus_ensemble_batch_size)
    crop_range = (float(args.v2_crop_rate_lo), float(args.v2_crop_rate_hi))

    rows: list[dict] = []
    ok_accs: list[float] = []
    ok_accs_by_suite: dict[str, list[float]] = {"ucr": [], "uea": []}

    for idx, (suite, name) in enumerate(jobs, start=1):
        row = {
            "suite": suite,
            "dataset": name,
            "accuracy": "",
            "n_train": "",
            "n_test": "",
            "n_classes": "",
            "status": "failed",
            "error": "",
        }
        try:
            print(f"[Eval][{idx}/{len(jobs)}] start {suite}:{name}")
            X_tr, y_tr, X_te, y_te, classes = prepare_ucr_uea_arrays(
                args,
                suite,
                name,
                reader,
                int(model_info["mantis_seq_len"]),
            )
            row.update({"n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_classes": int(len(classes))})

            (
                y_pred,
                used_v2_batch_size,
                used_signal_encoder_batch_size,
                used_task_level_chorus_batch_size,
            ) = predict_classifier_v2_with_oom_retry(
                custom_model=model,
                X_train=X_tr,
                y_train=y_tr,
                X_test=X_te,
                device=device,
                args=args,
                crop_rate_range=crop_range,
                initial_v2_batch_size=base_v2_batch_size,
                initial_signal_encoder_batch_size=base_signal_encoder_batch_size,
                initial_task_level_chorus_batch_size=base_task_level_chorus_batch_size,
            )
            row.update(
                {
                    "effective_v2_batch_size": int(used_v2_batch_size),
                    "effective_signal_encoder_batch_size": int(used_signal_encoder_batch_size),
                    "effective_task_level_chorus_ensemble_batch_size": int(used_task_level_chorus_batch_size),
                }
            )

            acc = float(np.mean(y_pred == y_te))
            row["accuracy"] = acc
            row["status"] = "ok"
            ok_accs.append(acc)
            ok_accs_by_suite.setdefault(str(suite), []).append(acc)
            print(f"{suite}:{name}: {acc:.4f}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"{suite}:{name}: failed: {exc}")
        rows.append(row)

    summary = {
        "num_jobs": len(jobs),
        "num_ok": len(ok_accs),
        "mean_accuracy": (float(np.mean(ok_accs)) if ok_accs else None),
        "ucr_num_ok": len(ok_accs_by_suite.get("ucr", [])),
        "ucr_mean_accuracy": (float(np.mean(ok_accs_by_suite["ucr"])) if ok_accs_by_suite.get("ucr") else None),
        "uea_num_ok": len(ok_accs_by_suite.get("uea", [])),
        "uea_mean_accuracy": (float(np.mean(ok_accs_by_suite["uea"])) if ok_accs_by_suite.get("uea") else None),
        "suite": args.suite,
        "argv": [str(arg) for arg in sys.argv],
        "command_line": _format_command_line(),
    }
    if ok_accs:
        print(f"\nEvaluated {len(ok_accs)}/{len(jobs)} datasets | mean accuracy: {summary['mean_accuracy']:.4f}")
        if summary["ucr_mean_accuracy"] is not None:
            print(f"UCR mean accuracy: {summary['ucr_mean_accuracy']:.4f} ({summary['ucr_num_ok']} datasets)")
        if summary["uea_mean_accuracy"] is not None:
            print(f"UEA mean accuracy: {summary['uea_mean_accuracy']:.4f} ({summary['uea_num_ok']} datasets)")
    else:
        print("\nNo datasets evaluated successfully.")
    _write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
