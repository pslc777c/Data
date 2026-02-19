from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"

# Universe: salida final de HIDR (ya tiene fecha_post_pred usada y ML1 hidr aplicado)
IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_hidr_grado_dia_bloque_destino_final.parquet"

# Real desperdicio (por fecha_post, destino)
IN_REAL = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_DS = GOLD / "ml2_datasets" / "ds_desp_poscosecha_ml2_v1.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_of_date(default_tz: str = "America/Guayaquil") -> pd.Timestamp:
    # “hoy” no se usa (día no cerrado) => as_of_date = hoy-1
    # pd.Timestamp.now() toma tz local del OS; usamos normalize y -1 día.
    return (pd.Timestamp.now().normalize() - pd.Timedelta(days=1))


def _resolve_fecha_post_pred(df: pd.DataFrame) -> str:
    for c in [
        "fecha_post_pred_final",
        "fecha_post_pred_used",
        "fecha_post_pred_ml1",
        "fecha_post_pred",
    ]:
        if c in df.columns:
            return c
    raise KeyError(
        "No encuentro columna de fecha_post_pred. Espero una de: "
        "fecha_post_pred_final / fecha_post_pred_used / fecha_post_pred_ml1 / fecha_post_pred"
    )


def _resolve_factor_desp_ml1(df: pd.DataFrame) -> str:
    for c in ["factor_desp_ml1", "factor_desp_pred_ml1", "factor_desp"]:
        if c in df.columns:
            return c
    raise KeyError(
        "No encuentro factor_desp ML1 en universe. Espero: factor_desp_ml1 / factor_desp_pred_ml1 / factor_desp"
    )


def _resolve_tallos(df: pd.DataFrame) -> str:
    for c in [
        "tallos",
        "tallos_total_ml2",
        "tallos_real_dia",
        "tallos_pred_ml1_grado_dia",
        "tallos_final_grado_dia",
        "tallos_pred_ml1_grado_dia",
    ]:
        if c in df.columns:
            return c
    # si no hay tallos, usamos 1.0 como peso
    return ""


def main() -> None:
    as_of = _as_of_date()

    uni = read_parquet(IN_UNIVERSE).copy()
    uni.columns = [str(c).strip() for c in uni.columns]

    # Canon mínimas
    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino"}
    miss = need - set(uni.columns)
    if miss:
        raise ValueError(f"Universe sin columnas: {sorted(miss)}")

    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_str(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])
    uni["destino"] = _canon_str(uni["destino"])
    # grado a str/cat
    if pd.api.types.is_numeric_dtype(uni["grado"]):
        uni["grado"] = _canon_int(uni["grado"]).astype("Int64")
    else:
        uni["grado"] = _canon_str(uni["grado"])

    # Cortar al as_of_date
    uni = uni.loc[uni["fecha"].notna() & (uni["fecha"] <= as_of)].copy()

    fpp_col = _resolve_fecha_post_pred(uni)
    fd_ml1_col = _resolve_factor_desp_ml1(uni)

    uni[fpp_col] = _to_date(uni[fpp_col])
    uni[fd_ml1_col] = pd.to_numeric(uni[fd_ml1_col], errors="coerce")

    tallos_col = _resolve_tallos(uni)
    if tallos_col:
        uni["tallos_w"] = pd.to_numeric(uni[tallos_col], errors="coerce").fillna(0.0)
    else:
        uni["tallos_w"] = 1.0

    # REAL: desperdicio por fecha_post + destino
    real = read_parquet(IN_REAL).copy()
    real.columns = [str(c).strip() for c in real.columns]
    need_r = {"fecha_post", "destino", "factor_desp"}
    miss_r = need_r - set(real.columns)
    if miss_r:
        raise ValueError(f"Real desperdicio sin columnas: {sorted(miss_r)}")

    real["fecha_post"] = _to_date(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])
    real["factor_desp_real"] = pd.to_numeric(real["factor_desp"], errors="coerce")

    # Colapsar a 1 fila por (fecha_post, destino) usando mediana (robusto)
    real2 = (
        real.groupby(["fecha_post", "destino"], dropna=False, as_index=False)
        .agg(factor_desp_real=("factor_desp_real", "median"))
    )

    # Join: fecha_post_pred + destino
    ds = uni.merge(
        real2,
        left_on=[fpp_col, "destino"],
        right_on=["fecha_post", "destino"],
        how="left",
    )

    ds["factor_desp_ml1"] = pd.to_numeric(ds[fd_ml1_col], errors="coerce")
    ds["factor_desp_real"] = pd.to_numeric(ds["factor_desp_real"], errors="coerce")

    # Target: log_ratio_desp = log(real/ml1)
    eps = 1e-12
    ratio = (ds["factor_desp_real"] + eps) / (ds["factor_desp_ml1"] + eps)
    ds["log_ratio_desp"] = np.log(ratio)

    # Clip target (para estabilidad)
    CLIP = 1.2
    ds["log_ratio_desp_clipped"] = ds["log_ratio_desp"].clip(lower=-CLIP, upper=CLIP)

    # Features calendario sobre fecha_post_pred usada
    ds["fecha_post_pred_used"] = ds[fpp_col]
    ds["dow"] = ds["fecha_post_pred_used"].dt.dayofweek.astype("Int64")
    ds["month"] = ds["fecha_post_pred_used"].dt.month.astype("Int64")
    ds["weekofyear"] = ds["fecha_post_pred_used"].dt.isocalendar().week.astype("Int64")

    # Flags de disponibilidad real
    ds["has_real"] = ds["factor_desp_real"].notna()

    ds["as_of_date"] = as_of
    ds["created_at"] = pd.Timestamp(datetime.now()).normalize()

    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(ds, OUT_DS)

    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(
        f"     rows={len(ds):,} rows_with_real={int(ds['has_real'].sum()):,} "
        f"as_of_date={as_of.date()} clip=±{CLIP}"
    )


if __name__ == "__main__":
    main()
