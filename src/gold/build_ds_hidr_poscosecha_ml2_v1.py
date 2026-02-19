from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"

# ✅ base con hidr ML1 y demás columnas
IN_BASE = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"
# ✅ dh_final con dh_dias_final + fecha_post_pred_final
IN_DH_FINAL = GOLD / "pred_poscosecha_ml2_dh_grado_dia_bloque_destino_final.parquet"

IN_REAL_HIDR = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"

OUT_DS = GOLD / "ml2_datasets" / "ds_hidr_poscosecha_ml2_v1.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_of_date_today_minus_1() -> pd.Timestamp:
    return (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).normalize()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _factor_from_hidr_pct(hidr_pct: pd.Series) -> pd.Series:
    x = pd.to_numeric(hidr_pct, errors="coerce")
    # Si viene como "porcentaje" (p.ej. 8 = 8%), convertir a fracción: 0.08
    # Si viene como fracción (p.ej. 0.08), dejar tal cual
    delta = np.where(x.isna(), np.nan, np.where(x > 3.5, x / 100.0, x)).astype(float)
    # Como el fact trae (post/pre - 1), el factor es delta + 1
    return (1.0 + delta).astype(float)



def main() -> None:
    as_of_date = _as_of_date_today_minus_1()
    created_at = pd.Timestamp.utcnow()

    base = read_parquet(IN_BASE).copy()
    base.columns = [str(c).strip() for c in base.columns]

    _require(base, ["fecha", "bloque_base", "variedad_canon", "grado", "destino"], "pred_poscosecha_ml2_full")
    # hidr ML1 debe existir aquí
    if "factor_hidr_ml1" not in base.columns:
        raise KeyError(f"pred_poscosecha_ml2_full no tiene factor_hidr_ml1. Columnas={list(base.columns)}")

    base["fecha"] = _to_date(base["fecha"])
    base["destino"] = _canon_str(base["destino"])
    base["grado"] = _canon_int(base["grado"])
    base["bloque_base"] = _canon_str(base["bloque_base"])
    base["variedad_canon"] = _canon_str(base["variedad_canon"])

    # filtro hoy-1
    base = base.loc[base["fecha"].notna() & (base["fecha"] <= as_of_date)].copy()

    # traer DH final
    dh = read_parquet(IN_DH_FINAL).copy()
    dh.columns = [str(c).strip() for c in dh.columns]
    _require(dh, ["fecha", "bloque_base", "variedad_canon", "grado", "destino", "dh_dias_final", "fecha_post_pred_final"], "pred_poscosecha_ml2_dh_final")

    dh["fecha"] = _to_date(dh["fecha"])
    dh["destino"] = _canon_str(dh["destino"])
    dh["grado"] = _canon_int(dh["grado"])
    dh["bloque_base"] = _canon_str(dh["bloque_base"])
    dh["variedad_canon"] = _canon_str(dh["variedad_canon"])
    dh["fecha_post_pred_final"] = _to_date(dh["fecha_post_pred_final"])

    key = ["fecha", "bloque_base", "variedad_canon", "grado", "destino"]
    dh_take = dh[key + ["dh_dias_final", "fecha_post_pred_final"]].copy()

    uni = base.merge(dh_take, on=key, how="left")

    # features calendario por fecha_post_pred_final
    uni["dow"] = uni["fecha_post_pred_final"].dt.dayofweek.astype("Int64")
    uni["month"] = uni["fecha_post_pred_final"].dt.month.astype("Int64")
    uni["weekofyear"] = uni["fecha_post_pred_final"].dt.isocalendar().week.astype("Int64")

    # pesos por tallos si existe
    tallos_col = None
    for c in ["tallos", "tallos_final_grado_dia", "tallos_pred_ml2_grado_dia", "tallos_pred_ml1_grado_dia"]:
        if c in uni.columns:
            tallos_col = c
            break
    uni["tallos_w"] = pd.to_numeric(uni[tallos_col], errors="coerce").fillna(0.0).clip(lower=0.0) if tallos_col else 1.0

    uni["factor_hidr_ml1"] = pd.to_numeric(uni["factor_hidr_ml1"], errors="coerce")

    # ---------------- REAL HIDR
    real = read_parquet(IN_REAL_HIDR).copy()
    real.columns = [str(c).strip() for c in real.columns]
    _require(real, ["fecha_post", "grado", "destino"], "fact_hidratacion_real_post_grado_destino")

    if "hidr_pct" in real.columns:
        real["factor_hidr_real"] = _factor_from_hidr_pct(real["hidr_pct"])
    else:
        _require(real, ["peso_base_g", "peso_post_g"], "fact_hidratacion_real_post_grado_destino")
        pb = pd.to_numeric(real["peso_base_g"], errors="coerce")
        pp = pd.to_numeric(real["peso_post_g"], errors="coerce")
        real["factor_hidr_real"] = np.where(pb > 0, pp / pb, np.nan)

    real["fecha_post"] = _to_date(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])
    real["grado"] = _canon_int(real["grado"])

    # agregación real por llave
    if "tallos" in real.columns:
        real["tallos"] = pd.to_numeric(real["tallos"], errors="coerce").fillna(0.0)
        g = real.groupby(["fecha_post", "grado", "destino"], dropna=False)
        real2 = g.apply(
            lambda x: pd.Series({
                "factor_hidr_real": float(np.nansum(x["factor_hidr_real"] * x["tallos"]) / np.nansum(x["tallos"])) if np.nansum(x["tallos"]) > 0 else float(np.nanmedian(x["factor_hidr_real"])),
                "tallos_real_sum": float(np.nansum(x["tallos"])),
            })
        ).reset_index()
    else:
        real2 = (
            real.groupby(["fecha_post", "grado", "destino"], dropna=False, as_index=False)
                .agg(factor_hidr_real=("factor_hidr_real", "median"))
        )
        real2["tallos_real_sum"] = np.nan

    ds = uni.merge(
        real2,
        left_on=["fecha_post_pred_final", "grado", "destino"],
        right_on=["fecha_post", "grado", "destino"],
        how="left",
    ).drop(columns=["fecha_post"], errors="ignore")

    # target log error
    eps = 1e-9
    ds["ratio_hidr"] = ds["factor_hidr_real"] / ds["factor_hidr_ml1"].clip(lower=eps)
    ds["log_error_hidr"] = np.log(ds["ratio_hidr"].clip(lower=eps))

    clip = 1.2
    ds["log_error_hidr_clipped"] = ds["log_error_hidr"].clip(lower=-clip, upper=clip)
    ds["is_in_real"] = ds["factor_hidr_real"].notna()

    ds["as_of_date"] = as_of_date
    ds["created_at"] = created_at

    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(ds, OUT_DS)

    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(f"     rows={len(ds):,} rows_with_real={int(ds['is_in_real'].sum()):,} as_of_date={as_of_date.date()} clip=±{clip}")


if __name__ == "__main__":
    main()
