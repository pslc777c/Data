from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
EVAL_ML1 = DATA / "eval" / "ml1_nn"
EVAL_ML2 = DATA / "eval" / "ml2_nn"
GOLD_ML1 = DATA / "gold" / "ml1_nn"
GOLD_ML2 = DATA / "gold" / "ml2_nn"
MODELS_ML2 = DATA / "models" / "ml2_nn"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("run_benchmark_chain_ml1_ml2")
    ap.add_argument("--dataset", default=str(GOLD_ML1 / "ds_ml1_nn_v1.parquet"))
    ap.add_argument(
        "--ml1-apply-input",
        default=None,
        help="Optional input parquet for apply_target_model_hybrid_ml1. Defaults to latest pred_ml1_multitask_nn_*.parquet.",
    )
    ap.add_argument("--val-quantile", type=float, default=0.70)
    ap.add_argument("--min-val-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-extra", action="store_true")
    ap.add_argument("--include-rf", action="store_true")
    ap.add_argument("--disable-zero-inflated", action="store_true")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD for build_ds_ml2_nn_v1.")
    ap.add_argument("--ml2-run-id", default=None, help="If provided and --no-retrain-ml2, uses this model.")
    ap.add_argument(
        "--retrain-ml2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train ML2 with the rebuilt ML1 input before apply.",
    )
    ap.add_argument(
        "--anchor-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass-through for apply_multitask_nn_ml2.",
    )
    ap.add_argument(
        "--focus-tallos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply hybrid override only on target_factor_tallos_dia.",
    )
    return ap.parse_args()


def _run(cmd: list[str]) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if p.stdout:
        print(p.stdout, end="")
    if p.stderr:
        print(p.stderr, end="", file=sys.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def _latest(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for pattern '{pattern}' in {path}")
    return files[-1]


def _latest_ml1_hybrid_pred() -> Path:
    files = []
    for p in GOLD_ML1.glob("pred_ml1_target_hybrid_*.parquet"):
        if "_post_" in p.name:
            continue
        files.append(p)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No ML1 hybrid prediction files found in {GOLD_ML1}")
    return files[-1]


def _latest_ml1_multitask_pred() -> Path:
    files = []
    for p in GOLD_ML1.glob("pred_ml1_multitask_nn_*.parquet"):
        if "_post_" in p.name:
            continue
        files.append(p)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No ML1 multitask prediction files found in {GOLD_ML1}")
    return files[-1]


def _train_ml2_and_get_run_id() -> str:
    _run([sys.executable, "src/models/ml2/train_multitask_nn_ml2.py"])
    meta = _latest(MODELS_ML2, "ml2_multitask_nn_*_meta.json")
    obj = json.loads(meta.read_text(encoding="utf-8"))
    run_id = str(obj.get("run_id", "")).strip()
    if not run_id:
        raise ValueError(f"Unable to parse run_id from {meta}")
    return run_id


def main() -> None:
    args = _parse_args()

    # 1) Benchmark per target (includes hybrid summary alias).
    bench_cmd = [
        sys.executable,
        "src/models/ml1/benchmark_per_target_models_ml1.py",
        "--dataset",
        str(Path(args.dataset)),
        "--val-quantile",
        str(args.val_quantile),
        "--min-val-n",
        str(args.min_val_n),
        "--seed",
        str(args.seed),
        "--save-models",
    ]
    if args.include_extra:
        bench_cmd.append("--include-extra")
    if args.include_rf:
        bench_cmd.append("--include-rf")
    if args.disable_zero_inflated:
        bench_cmd.append("--disable-zero-inflated")
    _run(bench_cmd)
    hybrid_summary = _latest(EVAL_ML1, "ml1_target_model_hybrid_summary_*.json")

    # 2) Apply hybrid ML1 and rebuild full chain.
    if args.ml1_apply_input:
        ml1_apply_input = Path(args.ml1_apply_input)
    else:
        try:
            ml1_apply_input = _latest_ml1_multitask_pred()
        except FileNotFoundError:
            ml1_apply_input = Path(args.dataset)

    _run(
        [
            sys.executable,
            "src/models/ml1/apply_target_model_hybrid_ml1.py",
            "--summary",
            str(hybrid_summary),
            "--input",
            str(ml1_apply_input),
        ]
        + (["--focus-tallos"] if args.focus_tallos else ["--no-focus-tallos"])
    )
    ml1_hybrid_pred = _latest_ml1_hybrid_pred()

    # 3) Build ML2 dataset from hybrid ML1 output.
    ds_cmd = [
        sys.executable,
        "src/gold/build_ds_ml2_nn_v1.py",
        "--ml1-input",
        str(ml1_hybrid_pred),
    ]
    if args.asof:
        ds_cmd.extend(["--asof", str(args.asof)])
    _run(ds_cmd)

    # 4) Train/resolve ML2 run.
    if args.retrain_ml2:
        ml2_run_id = _train_ml2_and_get_run_id()
    else:
        ml2_run_id = str(args.ml2_run_id).strip() if args.ml2_run_id else ""
        if not ml2_run_id:
            meta = _latest(MODELS_ML2, "ml2_multitask_nn_*_meta.json")
            ml2_run_id = str(json.loads(meta.read_text(encoding="utf-8")).get("run_id", "")).strip()
        if not ml2_run_id:
            raise ValueError("ml2_run_id is required when --no-retrain-ml2 and no latest meta is available.")

    # 5) Apply ML2.
    apply_ml2_cmd = [
        sys.executable,
        "src/models/ml2/apply_multitask_nn_ml2.py",
        "--run-id",
        ml2_run_id,
    ]
    if args.anchor_real:
        apply_ml2_cmd.append("--anchor-real")
    else:
        apply_ml2_cmd.append("--no-anchor-real")
    _run(apply_ml2_cmd)

    pred_puro = GOLD_ML2 / f"pred_ml2_multitask_nn_puro_{ml2_run_id}.parquet"
    pred_global = GOLD_ML2 / f"pred_ml2_multitask_nn_global_{ml2_run_id}.parquet"
    pred_oper = GOLD_ML2 / f"pred_ml2_multitask_nn_operativo_{ml2_run_id}.parquet"
    if not pred_puro.exists() or not pred_oper.exists() or not pred_global.exists():
        raise FileNotFoundError("ML2 output files were not generated as expected.")

    # 6) Build comparative views.
    _run(
        [
            sys.executable,
            "src/gold/build_views_ml1_ml2_real_compare.py",
            "--pred-ml2-global",
            str(pred_global),
            "--pred-ml2-puro",
            str(pred_puro),
            "--pred-ml2-operativo",
            str(pred_oper),
        ]
    )

    # 7) Chain summary.
    EVAL_ML2.mkdir(parents=True, exist_ok=True)
    chain_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_summary = EVAL_ML2 / f"ml1_ml2_benchmark_chain_{chain_id}.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_ml1": str(Path(args.dataset).resolve()),
        "hybrid_summary": str(hybrid_summary.resolve()),
        "ml1_hybrid_pred": str(ml1_hybrid_pred.resolve()),
        "ml2_run_id": ml2_run_id,
        "pred_ml2_global": str(pred_global.resolve()),
        "pred_ml2_puro": str(pred_puro.resolve()),
        "pred_ml2_operativo": str(pred_oper.resolve()),
        "retrain_ml2": bool(args.retrain_ml2),
        "anchor_real": bool(args.anchor_real),
    }
    out_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Chain summary: {out_summary}")


if __name__ == "__main__":
    main()
