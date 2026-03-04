from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from common.io import read_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "ml1_nn"
EVAL_DIR = DATA_DIR / "eval" / "ml1_nn"
GOLD_DIR = DATA_DIR / "gold" / "ml1_nn"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("build_pdf_ml1_nn_run_report")
    ap.add_argument("--run-id", default=None, help="If omitted, latest run is used.")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def _latest_meta() -> Path:
    files = sorted(MODELS_DIR.glob("ml1_multitask_nn_*_meta.json"))
    if not files:
        raise FileNotFoundError(f"No ML1-NN metadata found in {MODELS_DIR}")
    return files[-1]


def _meta_for_run(run_id: str) -> Path:
    p = MODELS_DIR / f"ml1_multitask_nn_{run_id}_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"Metadata not found: {p}")
    return p


def _draw_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    txt = "\n".join(lines)
    fig.text(0.03, 0.95, txt, va="top", ha="left", family="monospace", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _dict_lines(d: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{prefix}{k}:")
            out.extend(_dict_lines(v, prefix=prefix + "  "))
        else:
            out.append(f"{prefix}{k}: {v}")
    return out


def main() -> None:
    args = _parse_args()
    meta_path = _latest_meta() if args.run_id is None else _meta_for_run(args.run_id)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    run_id = str(meta["run_id"])

    history_path = Path(meta["history_path"])
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    pval_path = Path(meta["pvalues_path"])
    if not pval_path.is_absolute():
        pval_path = ROOT / pval_path

    pred_path = GOLD_DIR / f"pred_ml1_multitask_nn_{run_id}.parquet"
    post_total_path = GOLD_DIR / f"pred_ml1_multitask_nn_post_dia_total_{run_id}.parquet"

    out_path = Path(args.out) if args.out else (EVAL_DIR / f"ml1_nn_run_report_{run_id}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hist = read_parquet(history_path) if history_path.exists() else pd.DataFrame()
    pvals = read_parquet(pval_path) if pval_path.exists() else pd.DataFrame()
    pred = read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    post_total = read_parquet(post_total_path) if post_total_path.exists() else pd.DataFrame()

    with PdfPages(out_path) as pdf:
        lines = [
            f"run_id: {run_id}",
            f"meta: {meta_path}",
            "",
            "split:",
        ]
        lines.extend(_dict_lines(meta.get("split", {}), prefix="  "))
        lines.append("")
        lines.append("network:")
        lines.extend(_dict_lines(meta.get("network", {}), prefix="  "))
        lines.append("")
        lines.append("metrics_train:")
        lines.extend(_dict_lines(meta.get("metrics_train", {}), prefix="  "))
        lines.append("")
        lines.append("metrics_val:")
        lines.extend(_dict_lines(meta.get("metrics_val", {}), prefix="  "))
        _draw_text_page(pdf, "ML1 NN Run Summary", lines)

        if not hist.empty and {"epoch", "mae_train_avg", "mae_val_avg"} <= set(hist.columns):
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.plot(hist["epoch"], hist["mae_train_avg"], label="mae_train_avg")
            ax.plot(hist["epoch"], hist["mae_val_avg"], label="mae_val_avg")
            ax.set_title("Training History")
            ax.set_xlabel("epoch")
            ax.set_ylabel("masked MAE")
            ax.grid(True, alpha=0.3)
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not pred.empty:
            stage = pred["stage"].astype("string").str.upper().str.strip() if "stage" in pred.columns else pd.Series([], dtype="string")
            stage_rows = stage.value_counts(dropna=False).to_dict() if len(stage) else {}
            lines = ["rows by stage:"]
            for k, v in stage_rows.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("target mask coverage:")
            for c in sorted([c for c in pred.columns if c.startswith("mask_target_")]):
                cov = float(pd.to_numeric(pred[c], errors="coerce").fillna(0).mean())
                lines.append(f"  {c}: {cov:.2%}")
            _draw_text_page(pdf, "Dataset Coverage", lines)

        if not pvals.empty and {"target", "feature", "p_value"} <= set(pvals.columns):
            pvals = pvals[~pvals["feature"].astype("string").str.startswith(("bloque_base=", "area="), na=False)].copy()
            for target in sorted(pvals["target"].dropna().unique().tolist()):
                sub = pvals[pvals["target"] == target].copy().sort_values("p_value").head(25)
                lines = [f"target: {target}", "", "top features by p-value:"]
                for _, r in sub.iterrows():
                    lines.append(f"  {r['feature'][:80]} | p={r['p_value']:.3e}")
                _draw_text_page(pdf, f"P-Values - {target}", lines)

        if not post_total.empty and {"fecha_evento", "pred_cajas_post_ml1_nn", "cajas_verde_ref"} <= set(post_total.columns):
            p = post_total.copy()
            p["fecha_evento"] = pd.to_datetime(p["fecha_evento"], errors="coerce")
            p = p.sort_values("fecha_evento")

            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.plot(p["fecha_evento"], p["cajas_verde_ref"], label="cajas_verde_ref")
            ax.plot(p["fecha_evento"], p["pred_cajas_post_ml1_nn"], label="pred_cajas_post_ml1_nn")
            ax.set_title("Cajas Verdes vs Cajas Finales Ajustadas")
            ax.set_xlabel("fecha_evento")
            ax.set_ylabel("cajas")
            ax.grid(True, alpha=0.3)
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            lines = [
                "post totals:",
                f"  rows: {len(p):,}",
                f"  cajas_verde_ref total: {float(pd.to_numeric(p['cajas_verde_ref'], errors='coerce').sum()):,.2f}",
                f"  pred_cajas_post_ml1_nn total: {float(pd.to_numeric(p['pred_cajas_post_ml1_nn'], errors='coerce').sum()):,.2f}",
                f"  pred_kg_post_ml1_nn total: {float(pd.to_numeric(p.get('pred_kg_post_ml1_nn'), errors='coerce').sum()):,.2f}",
            ]
            _draw_text_page(pdf, "Postcosecha Totals", lines)

        elif not post_total.empty and {"fecha_evento", "cajas_post_ML1", "cajas_verde_ML1"} <= set(post_total.columns):
            p = post_total.copy()
            p["fecha_evento"] = pd.to_datetime(p["fecha_evento"], errors="coerce")
            p = p.sort_values("fecha_evento")

            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.plot(p["fecha_evento"], p["cajas_verde_ML1"], label="cajas_verde_ML1")
            ax.plot(p["fecha_evento"], p["cajas_post_ML1"], label="cajas_post_ML1")
            ax.set_title("Cajas Verdes vs Cajas Finales Ajustadas")
            ax.set_xlabel("fecha_evento")
            ax.set_ylabel("cajas")
            ax.grid(True, alpha=0.3)
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            lines = [
                "post totals:",
                f"  rows: {len(p):,}",
                f"  cajas_verde_ML1 total: {float(pd.to_numeric(p['cajas_verde_ML1'], errors='coerce').sum()):,.2f}",
                f"  cajas_post_ML1 total: {float(pd.to_numeric(p['cajas_post_ML1'], errors='coerce').sum()):,.2f}",
                f"  kg_post_ML1 total: {float(pd.to_numeric(p.get('kg_post_ML1'), errors='coerce').sum()):,.2f}",
                f"  aprovechamiento_ML1 avg: {float(pd.to_numeric(p.get('aprovechamiento_ML1'), errors='coerce').mean()):.4f}",
            ]
            _draw_text_page(pdf, "Postcosecha Totals", lines)

    print(f"[OK] PDF report written: {out_path}")


if __name__ == "__main__":
    main()
