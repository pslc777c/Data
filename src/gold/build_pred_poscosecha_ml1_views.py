from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


IN_PATH = Path("data/gold/pred_poscosecha_ml1_full_grado_dia_bloque_destino.parquet")

OUT_DIA_BLOQUE_DEST = Path("data/gold/pred_poscosecha_ml1_dia_bloque_destino.parquet")
OUT_DIA_DEST = Path("data/gold/pred_poscosecha_ml1_dia_destino.parquet")
OUT_DIA_TOTAL = Path("data/gold/pred_poscosecha_ml1_dia_total.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    df = read_parquet(IN_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # ---- llaves mínimas esperadas
    req = ["fecha", "bloque_base", "variedad_canon", "grado", "destino"]
    _require(df, req, "pred_poscosecha_ml1_full_grado_dia_bloque_destino")

    # ---- normalización
    df["fecha"] = _to_date(df["fecha"])
    df["bloque_base"] = _canon_int(df["bloque_base"])
    df["grado"] = _canon_int(df["grado"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])
    df["destino"] = _canon_str(df["destino"])

    # ==========================================================
    # cajas iniciales: DEBE SER SPLIT (porque ya está a grano destino)
    # ==========================================================
    cajas_in_col = None
    for c in [
        "cajas_split_grado_dia",      # <- correcto
        "cajas_split",                # <- alias posible
        "cajas_iniciales",            # <- si algún step ya lo renombró
        "cajas_in",                   # <- alias
        "cajas_ml1_grado_dia",        # <- fallback (NO ideal)
    ]:
        if c in df.columns:
            cajas_in_col = c
            break

    if cajas_in_col is None:
        raise ValueError(
            "No encontré columna de cajas iniciales a grano destino. "
            "Esperaba cajas_split_grado_dia (o similar)."
        )

    # Si por error solo existe cajas_ml1_grado_dia, avisamos (porque puede reintroducir mismatch)
    if cajas_in_col == "cajas_ml1_grado_dia":
        print("[WARN] Usando cajas_ml1_grado_dia como iniciales (no hay split). Puede reintroducir mismatch.")

    # ==========================================================
    # cajas_postcosecha_ml1: si falta, derivarla desde factores
    # ==========================================================
    if "cajas_postcosecha_ml1" not in df.columns:
        fh = pd.to_numeric(df.get("factor_hidr_ml1"), errors="coerce").fillna(1.0)
        fd = pd.to_numeric(df.get("factor_desp_ml1"), errors="coerce").fillna(1.0)
        aj = pd.to_numeric(df.get("ajuste_ml1"), errors="coerce").replace(0, np.nan).fillna(1.0)
        ci = pd.to_numeric(df[cajas_in_col], errors="coerce").fillna(0.0)
        df["cajas_postcosecha_ml1"] = ci * fh * fd / aj

    # ==========================================================
    # fecha_post_pred_ml1: si no existe, construir desde dh
    # ==========================================================
    if "fecha_post_pred_ml1" not in df.columns:
        dh_col = None
        for c in ["dh_dias_ml1", "dh_dias_pred_ml1", "dh_dias_pred", "dh_dias"]:
            if c in df.columns:
                dh_col = c
                break
        if dh_col is not None:
            df["fecha_post_pred_ml1"] = df["fecha"] + pd.to_timedelta(
                pd.to_numeric(df[dh_col], errors="coerce").fillna(0).astype(int),
                unit="D",
            )
        else:
            df["fecha_post_pred_ml1"] = pd.NaT

    # ==========================================================
    # 1) DIA + BLOQUE + DESTINO
    # ==========================================================
    g1 = ["fecha", "fecha_post_pred_ml1", "bloque_base", "destino"]

    out1 = (
        df.groupby(g1, dropna=False, as_index=False)
          .agg(
              cajas_iniciales=(cajas_in_col, "sum"),
              cajas_postcosecha_ml1=("cajas_postcosecha_ml1", "sum"),
          )
    )
    out1["created_at"] = created_at
    out1 = out1.sort_values(["bloque_base", "destino", "fecha"]).reset_index(drop=True)

    write_parquet(out1, OUT_DIA_BLOQUE_DEST)
    print(f"OK -> {OUT_DIA_BLOQUE_DEST} | rows={len(out1):,} | cajas_base={cajas_in_col}")

    # ==========================================================
    # 2) DIA + DESTINO
    # ==========================================================
    g2 = ["fecha", "fecha_post_pred_ml1", "destino"]
    out2 = (
        out1.groupby(g2, dropna=False, as_index=False)
            .agg(
                cajas_iniciales=("cajas_iniciales", "sum"),
                cajas_postcosecha_ml1=("cajas_postcosecha_ml1", "sum"),
            )
    )
    out2["created_at"] = created_at
    out2 = out2.sort_values(["destino", "fecha"]).reset_index(drop=True)

    write_parquet(out2, OUT_DIA_DEST)
    print(f"OK -> {OUT_DIA_DEST} | rows={len(out2):,}")

    # ==========================================================
    # 3) DIA TOTAL
    # ==========================================================
    g3 = ["fecha", "fecha_post_pred_ml1"]
    out3 = (
        out2.groupby(g3, dropna=False, as_index=False)
            .agg(
                cajas_iniciales=("cajas_iniciales", "sum"),
                cajas_postcosecha_ml1=("cajas_postcosecha_ml1", "sum"),
            )
    )
    out3["created_at"] = created_at
    out3 = out3.sort_values(["fecha"]).reset_index(drop=True)

    write_parquet(out3, OUT_DIA_TOTAL)
    print(f"OK -> {OUT_DIA_TOTAL} | rows={len(out3):,}")

    # ==========================================================
    # SANITY: mass-balance interno (sum destinos = total)
    # ==========================================================
    chk = (
        out2.groupby(["fecha", "fecha_post_pred_ml1"], dropna=False)["cajas_iniciales"]
            .sum()
            .reset_index()
            .rename(columns={"cajas_iniciales": "cajas_iniciales_sum_dest"})
    )
    chk = chk.merge(
        out3[["fecha", "fecha_post_pred_ml1", "cajas_iniciales"]],
        on=["fecha", "fecha_post_pred_ml1"],
        how="left",
    ).rename(columns={"cajas_iniciales": "cajas_iniciales_total"})

    chk["abs_diff"] = (chk["cajas_iniciales_sum_dest"] - chk["cajas_iniciales_total"]).abs()
    max_abs = float(chk["abs_diff"].max()) if len(chk) else float("nan")
    print(f"[CHECK] mass-balance dia (sum destinos vs total) | max_abs_diff={max_abs:.12f}")
    if max_abs > 1e-6:
        raise ValueError("[FATAL] Mass-balance día no cuadra. Revisa agregaciones.")


if __name__ == "__main__":
    main()
