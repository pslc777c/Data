from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
IN_DEFAULT = ROOT / "data" / "features" / "features_cosecha_bloque_fecha.parquet"
OUT_DEFAULT = ROOT / "data" / "eval" / "ml1_nn" / "share_grado_real_hist_global.pdf"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("build_pdf_share_grado_real_hist")
    ap.add_argument("--input", default=str(IN_DEFAULT), help="Parquet histórico con columnas reales de grado.")
    ap.add_argument("--output", default=str(OUT_DEFAULT), help="PDF de salida.")
    return ap.parse_args()


def _load_real_grade_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_parquet(path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "fecha" not in df.columns or "grado" not in df.columns:
        raise ValueError("Input requiere columnas 'fecha' y 'grado'.")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce").round().astype("Int64")

    # Prefer direct real stems if present.
    tallos_real = pd.to_numeric(df.get("tallos_real_grado"), errors="coerce")
    if tallos_real.notna().any():
        df["tallos_real_grado"] = tallos_real
    else:
        share = pd.to_numeric(df.get("share_grado_real"), errors="coerce")
        tallos_dia = pd.to_numeric(df.get("tallos_real_dia"), errors="coerce")
        if not share.notna().any() or not tallos_dia.notna().any():
            raise ValueError(
                "No encuentro 'tallos_real_grado' ni combinación ('share_grado_real' x 'tallos_real_dia')."
            )
        df["tallos_real_grado"] = share * tallos_dia

    df["tallos_real_grado"] = pd.to_numeric(df["tallos_real_grado"], errors="coerce")
    df = df.loc[
        df["fecha"].notna()
        & df["grado"].notna()
        & df["tallos_real_grado"].notna()
        & (df["tallos_real_grado"] > 0.0)
    ].copy()
    if df.empty:
        raise ValueError("No hay filas válidas con tallos reales por grado.")
    return df


def _global_share(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("grado", dropna=False, as_index=False)
        .agg(tallos_real_grado=("tallos_real_grado", "sum"))
        .sort_values("grado", kind="mergesort")
        .reset_index(drop=True)
    )
    tot = float(out["tallos_real_grado"].sum())
    out["share_global"] = np.where(tot > 0.0, out["tallos_real_grado"] / tot, np.nan)
    return out


def _monthly_share(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    m["month"] = m["fecha"].dt.to_period("M").dt.to_timestamp()
    out = (
        m.groupby(["month", "grado"], dropna=False, as_index=False)
        .agg(tallos_real_grado=("tallos_real_grado", "sum"))
        .sort_values(["month", "grado"], kind="mergesort")
        .reset_index(drop=True)
    )
    den = out.groupby("month", dropna=False)["tallos_real_grado"].transform("sum")
    out["share_month"] = np.where(den > 0.0, out["tallos_real_grado"] / den, np.nan)
    return out


def _render_pdf(out_pdf: Path, df: pd.DataFrame, gshare: pd.DataFrame, mshare: pd.DataFrame) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # Page 1: global share by grade
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        x = gshare["grado"].astype(int).tolist()
        y = (gshare["share_global"] * 100.0).to_numpy(dtype=float)
        ax.bar(x, y, color="#1f77b4")
        ax.set_title("Share Global Real por Grado (Histórico)")
        ax.set_xlabel("Grado")
        ax.set_ylabel("Share (%)")
        ax.grid(axis="y", alpha=0.25)
        p70 = float(gshare.loc[gshare["grado"].eq(70), "share_global"].sum() * 100.0)
        p75 = float(gshare.loc[gshare["grado"].eq(75), "share_global"].sum() * 100.0)
        p70_75 = p70 + p75
        ax.text(
            0.01,
            0.98,
            (
                f"Rango fechas: {df['fecha'].min().date()} a {df['fecha'].max().date()}\n"
                f"Total tallos reales: {df['tallos_real_grado'].sum():,.0f}\n"
                f"Share grado 70: {p70:.2f}%\n"
                f"Share grado 75: {p75:.2f}%\n"
                f"Share 70+75: {p70_75:.2f}%"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: monthly trend for top grades by volume
        top_grades = (
            gshare.sort_values("tallos_real_grado", ascending=False)
            .head(8)["grado"]
            .astype("Int64")
            .tolist()
        )
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        for g in top_grades:
            s = mshare.loc[mshare["grado"].eq(g)].copy()
            if s.empty:
                continue
            ax.plot(s["month"], s["share_month"] * 100.0, marker="o", linewidth=1.4, label=f"G{int(g)}")
        ax.set_title("Evolución Mensual de Share Real por Grado (Top 8)")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Share mensual (%)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: summary table
        tbl = gshare.copy()
        tbl["grado"] = tbl["grado"].astype("Int64")
        tbl["share_pct"] = (tbl["share_global"] * 100.0).round(3)
        tbl["tallos_real_grado"] = tbl["tallos_real_grado"].round(0).astype("Int64")
        tbl = tbl[["grado", "tallos_real_grado", "share_pct"]]

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.set_title("Tabla Resumen - Share Global Real Histórico", pad=20)
        table_data = [["Grado", "Tallos Reales", "Share Global (%)"]] + tbl.astype(str).values.tolist()
        tab = ax.table(cellText=table_data, loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1.1, 1.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    if out_path.suffix.lower() != ".pdf":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_path.with_name(f"{out_path.stem}_{ts}.pdf")

    df = _load_real_grade_history(in_path)
    gshare = _global_share(df)
    mshare = _monthly_share(df)
    _render_pdf(out_path, df=df, gshare=gshare, mshare=mshare)

    print(f"[OK] PDF: {out_path}")
    print(f"     fecha=[{df['fecha'].min().date()} .. {df['fecha'].max().date()}] rows={len(df):,}")
    print("     share_global_by_grade:")
    for r in gshare.itertuples(index=False):
        print(f"       - grado={int(r.grado):>2d}: share={float(r.share_global)*100.0:6.3f}%")
    p70 = float(gshare.loc[gshare["grado"].eq(70), "share_global"].sum() * 100.0)
    p75 = float(gshare.loc[gshare["grado"].eq(75), "share_global"].sum() * 100.0)
    print(f"     share_70={p70:.3f}% share_75={p75:.3f}% share_70_75={(p70+p75):.3f}%")


if __name__ == "__main__":
    main()
