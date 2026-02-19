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

IN_GRID = GOLD / "universe_harvest_grid_ml2.parquet"
IN_CLIMA = SILVER / "dim_clima_bloque_dia.parquet"

# ML1 peso por tallo (ajusta si tu nombre es distinto; dejo dos candidatos)
IN_ML1_A = GOLD / "pred_peso_tallo_grado_ml1.parquet"
IN_ML1_B = GOLD / "pred_peso_tallo_grado_ml1_full.parquet"

# Reales (Opción A: ya tienes peso/tallo real por grado-día)
IN_REAL_PT = SILVER / "fact_peso_tallo_real_grado_dia.parquet"

# Reales (Opción B: derivar desde kg_real y tallos_real)
IN_REAL_KG = SILVER / "fact_kg_real_grado_dia.parquet"
IN_REAL_TALLOS = SILVER / "fact_cosecha_real_grado_dia.parquet"

OUT_DS = GOLD / "ml2_datasets" / "ds_peso_tallo_ml2_v1.parquet"

EPS = 0.1  # gramos, para estabilidad (peso/tallo suele ser > 0)
CLIP_LOG_ERR = 0.8  # exp(±0.8) ≈ [0.45 .. 2.23]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _resolve_block_base(df: pd.DataFrame) -> pd.DataFrame:
    if "bloque_base" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_base"])
        return df
    if "bloque_padre" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_padre"])
        return df
    if "bloque" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque"])
        return df
    raise KeyError("No encuentro columna de bloque: bloque_base / bloque_padre / bloque")


def _pick_ml1_file() -> Path:
    if IN_ML1_A.exists():
        return IN_ML1_A
    if IN_ML1_B.exists():
        return IN_ML1_B
    raise FileNotFoundError(f"No encuentro ML1 peso/tallo: {IN_ML1_A} ni {IN_ML1_B}")


def _load_real_peso_tallo() -> pd.DataFrame:
    # Opción A
    if IN_REAL_PT.exists():
        real = read_parquet(IN_REAL_PT).copy()
        # Esperamos algo como: fecha, bloque(_base), variedad, grado, peso_tallo_real (g)
        real["fecha"] = _to_date(real["fecha"])
        real = _resolve_block_base(real)
        if "grado" in real.columns:
            real["grado"] = _canon_str(real["grado"])
        if "variedad" in real.columns:
            real["variedad"] = _canon_str(real["variedad"])

        # detectar col peso
        cand = [c for c in ["peso_tallo_real_g", "peso_tallo_real", "peso_tallo_g_real", "peso_tallo_g"] if c in real.columns]
        if not cand and ("peso_real_g" in real.columns) and ("tallos_real" in real.columns):
            tmp = real[["fecha", "bloque_base", "grado", "peso_real_g", "tallos_real"]].copy()
            tmp["peso_real_g"] = pd.to_numeric(tmp["peso_real_g"], errors="coerce").fillna(0.0)
            tmp["tallos_real"] = pd.to_numeric(tmp["tallos_real"], errors="coerce").fillna(0.0)
            tmp["peso_tallo_real_g"] = np.where(tmp["tallos_real"] > 0, tmp["peso_real_g"] / tmp["tallos_real"], np.nan)
            return tmp[["fecha", "bloque_base", "grado", "peso_tallo_real_g"]]

        if not cand:
            raise KeyError("fact_peso_tallo_real_grado_dia no tiene columna de peso. "
                           "Espero: peso_tallo_real / peso_tallo_g_real / peso_tallo_g")
        col = cand[0]
        out = real[["fecha", "bloque_base", "grado", col]].rename(columns={col: "peso_tallo_real_g"})
        out["peso_tallo_real_g"] = pd.to_numeric(out["peso_tallo_real_g"], errors="coerce")
        return out

    # Opción B: kg_real / tallos_real
    if not IN_REAL_KG.exists() or not IN_REAL_TALLOS.exists():
        raise FileNotFoundError(
            "No encuentro reales para peso/tallo. Necesito IN_REAL_PT o (IN_REAL_KG + IN_REAL_TALLOS). "
            f"Faltan: {IN_REAL_PT} o {IN_REAL_KG} / {IN_REAL_TALLOS}"
        )

    kg = read_parquet(IN_REAL_KG).copy()
    tl = read_parquet(IN_REAL_TALLOS).copy()

    kg["fecha"] = _to_date(kg["fecha"])
    tl["fecha"] = _to_date(tl["fecha"])

    kg = _resolve_block_base(kg)
    tl = _resolve_block_base(tl)

    if "grado" in kg.columns:
        kg["grado"] = _canon_str(kg["grado"])
    if "grado" in tl.columns:
        tl["grado"] = _canon_str(tl["grado"])

    # detectar columna kg real
    cand_kg = [c for c in ["kg_real", "kg", "kg_real_grado_dia"] if c in kg.columns]
    if not cand_kg:
        raise KeyError("fact_kg_real_grado_dia no tiene kg. Espero: kg_real / kg / kg_real_grado_dia")
    col_kg = cand_kg[0]

    kg2 = kg.groupby(["fecha", "bloque_base", "grado"], as_index=False).agg(kg_real=(col_kg, "sum"))
    tl2 = tl.groupby(["fecha", "bloque_base", "grado"], as_index=False).agg(tallos_real=("tallos_real", "sum"))

    m = kg2.merge(tl2, on=["fecha", "bloque_base", "grado"], how="inner")
    m["kg_real"] = pd.to_numeric(m["kg_real"], errors="coerce").fillna(0.0)
    m["tallos_real"] = pd.to_numeric(m["tallos_real"], errors="coerce").fillna(0.0)

    # g/tallo
    m["peso_tallo_real_g"] = np.where(m["tallos_real"] > 0, (m["kg_real"] * 1000.0) / m["tallos_real"], np.nan)
    return m[["fecha", "bloque_base", "grado", "peso_tallo_real_g"]]


def main() -> None:
    grid = read_parquet(IN_GRID).copy()
    clima = read_parquet(IN_CLIMA).copy()

    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    if "variedad_canon" in grid.columns:
        grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    # ML1 peso/tallo
    ml1_path = _pick_ml1_file()
    ml1 = read_parquet(ml1_path).copy()

    ml1["bloque_base"] = _canon_str(ml1["bloque_base"]) if "bloque_base" in ml1.columns else ml1.assign(bloque_base=None)["bloque_base"]
    if "grado" in ml1.columns:
        ml1["grado"] = _canon_str(ml1["grado"])

    # detectar columna peso ml1
    cand_ml1 = [c for c in ["peso_tallo_ml1_g", "peso_tallo_ml1", "peso_tallo_pred_ml1_g", "peso_tallo_pred_ml1"] if c in ml1.columns]
    if not cand_ml1:
        # si no existe, usa peso_tallo_pred (legacy)
        cand_ml1 = [c for c in ["peso_tallo_pred", "peso_tallo_g_pred"] if c in ml1.columns]
    if not cand_ml1:
        raise KeyError(
            "No encuentro columna de peso/tallo ML1 en pred_peso_tallo_grado_ml1*. "
            "Espero: peso_tallo_ml1_g / peso_tallo_ml1 / peso_tallo_pred_ml1_g / peso_tallo_pred_ml1 (o peso_tallo_pred)."
        )
    col_ml1 = cand_ml1[0]

    # Unir universo con ML1 por (bloque_base, grado). Si ML1 tiene variedad, se agrega también.
    join_keys = ["bloque_base", "grado"]
    if "variedad_canon" in ml1.columns and "variedad_canon" in grid.columns:
        ml1["variedad_canon"] = _canon_str(ml1["variedad_canon"])
        join_keys = ["bloque_base", "variedad_canon", "grado"]

    base_cols = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "rel_pos_final", "day_in_harvest_final", "n_harvest_days_final",
                 "dow", "month", "weekofyear", "area", "tipo_sp", "estado"]
    base_cols = [c for c in base_cols if c in grid.columns]
    # Para peso/tallo, necesitamos grade. Lo tomamos desde ML1 y replicamos por grado (merge many-to-many controlado)
    df = grid[base_cols].drop_duplicates(["ciclo_id", "fecha"]).copy()

    # expand grades via ML1 universe (distinct grados por join_keys sin fecha)
    ml1_small = ml1[join_keys + [col_ml1]].drop_duplicates(join_keys).copy()
    ml1_small = ml1_small.rename(columns={col_ml1: "peso_tallo_ml1_g"})
    ml1_small["peso_tallo_ml1_g"] = pd.to_numeric(ml1_small["peso_tallo_ml1_g"], errors="coerce")

    df = df.merge(ml1_small, on=[k for k in join_keys if k in df.columns] , how="left")

    # si join_keys incluye grado pero df no lo tiene, lo agrega desde ml1_small en el merge
    if "grado" in df.columns:
        df["grado"] = _canon_str(df["grado"])

    # reales peso/tallo
    real = _load_real_peso_tallo()
    real["fecha"] = _to_date(real["fecha"])
    real["bloque_base"] = _canon_str(real["bloque_base"])
    if "grado" in real.columns:
        real["grado"] = _canon_str(real["grado"])

    df = df.merge(real, on=["fecha", "bloque_base", "grado"], how="left")

    # clima
    clima["fecha"] = _to_date(clima["fecha"])
    clima["bloque_base"] = _canon_str(clima["bloque_base"])
    clima_cols = [
        "fecha", "bloque_base",
        "gdc_dia",
        "rainfall_mm_dia", "en_lluvia_dia",
        "temp_avg_dia", "solar_energy_j_m2_dia",
        "wind_speed_avg_dia", "wind_run_dia",
    ]
    clima_cols = [c for c in clima_cols if c in clima.columns]
    df = df.merge(clima[clima_cols].copy(), on=["fecha", "bloque_base"], how="left")

    # Limpieza + target
    df["peso_tallo_ml1_g"] = pd.to_numeric(df["peso_tallo_ml1_g"], errors="coerce")
    df["peso_tallo_real_g"] = pd.to_numeric(df["peso_tallo_real_g"], errors="coerce")

    # Filtrar filas con señal suficiente
    df = df[df["peso_tallo_ml1_g"].notna()].copy()

    ratio = (df["peso_tallo_real_g"] + EPS) / (df["peso_tallo_ml1_g"] + EPS)
    df["log_error_peso"] = np.log(ratio).clip(-CLIP_LOG_ERR, CLIP_LOG_ERR)
    df["peso_tallo_ratio"] = np.exp(df["log_error_peso"])

    # evitar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, OUT_DS)

    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(f"     rows={len(df):,} cycles={df['ciclo_id'].nunique():,} "
          f"fecha_range=[{df['fecha'].min().date()}..{df['fecha'].max().date()}]")
    print(f"     ml1_file_used={ml1_path.name}  ml1_col_used={col_ml1}")
    print(f"     real_source={'fact_peso_tallo_real_grado_dia' if IN_REAL_PT.exists() else 'kg_real/tallos_real derived'}")


if __name__ == "__main__":
    main()
