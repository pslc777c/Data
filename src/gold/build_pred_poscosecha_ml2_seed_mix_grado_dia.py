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
SILVER = DATA / "silver"
GOLD = DATA / "gold"

# INPUTS
IN_CAJAS_GRADO = GOLD / "pred_cajas_grado_dia_ml2_full.parquet"
IN_MIX = SILVER / "dim_mix_proceso_semana.parquet"

# OUTPUTS (ML2 seed)
OUT_GD_BD = GOLD / "pred_poscosecha_ml2_seed_grado_dia_bloque_destino.parquet"
OUT_DD = GOLD / "pred_poscosecha_ml2_seed_dia_destino.parquet"
OUT_DT = GOLD / "pred_poscosecha_ml2_seed_dia_total.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _safe_float(x, default: float) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _semana_ventas_from_fecha_cosecha(fecha: pd.Series) -> pd.Series:
    """
    CONSISTENTE con tu lógica de ventas (Fecha_Clasificacion = Fecha - 2; semana_ventas(x): (x+2)->%U).
    Para fecha de cosecha (equivale a Fecha_Clasificacion):
      Semana_Ventas = semana_ventas(fecha)
    """
    d = pd.to_datetime(fecha, errors="coerce") + pd.Timedelta(days=2)
    yy = (d.dt.year.astype("Int64") % 100).astype("Int64")
    ww = d.dt.strftime("%U").astype(int)
    ww = np.where(ww == 0, 1, ww)
    return yy.astype(str).str.zfill(2) + pd.Series(ww).astype(str).str.zfill(2)


def _collapse_sum(df: pd.DataFrame, keys: list[str], val_cols: list[str], name: str) -> pd.DataFrame:
    dup = int(df.duplicated(subset=keys).sum())
    if dup > 0:
        agg = {c: "sum" for c in val_cols if c in df.columns}
        extra = [c for c in df.columns if c not in keys and c not in agg]
        for c in extra:
            agg[c] = "first"
        out = df.groupby(keys, dropna=False, as_index=False).agg(agg)
        print(f"[WARN] {name}: duplicados por {keys} -> colapsado sum -> rows={len(out):,}")
        return out
    return df


def _resolve_cajas_col(df: pd.DataFrame) -> str:
    # prioridad: columna final ML2
    candidates = [
        "cajas_ml2_grado_dia",
        "cajas_final_grado_dia",
        "cajas_pred_ml2_grado_dia",
        "cajas_grado_dia_ml2",
        # fallback: si tu builder dejó el mismo nombre ml1 (no ideal, pero posible)
        "cajas_ml1_grado_dia",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "No encuentro columna de cajas en pred_cajas_grado_dia_ml2_full.parquet. "
        f"Probé: {candidates}. Disponibles={list(df.columns)}"
    )


def main(as_of_date: str | None = None) -> None:
    created_at = pd.Timestamp.utcnow()

    # as_of_date default = hoy-1 (operativo)
    if as_of_date is None:
        as_of = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    else:
        as_of = pd.to_datetime(as_of_date, errors="coerce").normalize()
        if pd.isna(as_of):
            raise ValueError(f"as_of_date inválida: {as_of_date}")

    # -------------------------
    # 1) Load cajas ML2 (grado/día) -> supply
    # -------------------------
    cajas = read_parquet(IN_CAJAS_GRADO).copy()
    cajas.columns = [str(c).strip() for c in cajas.columns]

    need = ["fecha", "bloque_base", "variedad_canon", "grado"]
    _require(cajas, need, "pred_cajas_grado_dia_ml2_full")

    cajas["fecha"] = _to_date(cajas["fecha"])
    cajas["bloque_base"] = _canon_int(cajas["bloque_base"])
    cajas["grado"] = _canon_int(cajas["grado"])
    cajas["variedad_canon"] = _canon_str(cajas["variedad_canon"])

    cajas_col = _resolve_cajas_col(cajas)
    cajas["cajas_campo_ml2_grado_dia"] = pd.to_numeric(cajas[cajas_col], errors="coerce").fillna(0.0)

    # filtro operativo hoy-1
    cajas = cajas.loc[cajas["fecha"].notna()].copy()

    key_supply = ["fecha", "bloque_base", "variedad_canon", "grado"]
    cajas = _collapse_sum(cajas[key_supply + ["cajas_campo_ml2_grado_dia"]], key_supply, ["cajas_campo_ml2_grado_dia"], "cajas_ml2")

    # -------------------------
    # 2) Load mix por Semana_Ventas
    # -------------------------
    mix = read_parquet(IN_MIX).copy()
    mix.columns = [str(c).strip() for c in mix.columns]
    _require(mix, ["Semana_Ventas", "W_Blanco", "W_Arcoiris", "W_Tinturado"], "dim_mix_proceso_semana")

    mix["Semana_Ventas"] = mix["Semana_Ventas"].astype(str).str.strip()
    for c in ["W_Blanco", "W_Arcoiris", "W_Tinturado"]:
        mix[c] = pd.to_numeric(mix[c], errors="coerce")

    # DEFAULT obligatorio
    mix_def = mix[mix["Semana_Ventas"].eq("DEFAULT")].copy()
    if mix_def.empty:
        raise ValueError("dim_mix_proceso_semana no trae Semana_Ventas='DEFAULT' (fallback obligatorio).")
    r = mix_def.iloc[0]
    def_w = {
        "W_Blanco": _safe_float(r["W_Blanco"], 0.80),
        "W_Arcoiris": _safe_float(r["W_Arcoiris"], 0.10),
        "W_Tinturado": _safe_float(r["W_Tinturado"], 0.10),
    }
    sdef = def_w["W_Blanco"] + def_w["W_Arcoiris"] + def_w["W_Tinturado"]
    def_w = {k: (v / sdef if sdef > 0 else v) for k, v in def_w.items()}

    DESTS = [
        ("BLANCO", "W_Blanco"),
        ("ARCOIRIS", "W_Arcoiris"),
        ("TINTURADO", "W_Tinturado"),
    ]

    # -------------------------
    # 3) Semana_Ventas + join mix (fallback DEFAULT)
    # -------------------------
    supply = cajas.copy()
    supply["Semana_Ventas"] = _semana_ventas_from_fecha_cosecha(supply["fecha"])

    mix_take = mix[mix["Semana_Ventas"].ne("DEFAULT")].drop_duplicates(subset=["Semana_Ventas"], keep="last").copy()
    supply = supply.merge(mix_take, on="Semana_Ventas", how="left")

    for _, wcol in DESTS:
        supply[wcol] = supply[wcol].fillna(def_w[wcol])

    # renormalize defensivo
    ws = supply[[w for _, w in DESTS]].sum(axis=1).replace(0, np.nan)
    for _, wcol in DESTS:
        supply[wcol] = np.where(ws.notna(), supply[wcol] / ws, def_w[wcol])

    # -------------------------
    # 4) Split por destino (mass-balance exact)
    # -------------------------
    chunks = []
    for dest, wcol in DESTS:
        sub = supply[key_supply + ["Semana_Ventas", "cajas_campo_ml2_grado_dia", wcol]].copy()
        sub = sub.rename(columns={wcol: "w_dest"})
        sub["destino"] = dest
        sub["cajas_split_grado_dia"] = sub["cajas_campo_ml2_grado_dia"].astype(float) * sub["w_dest"].astype(float)
        chunks.append(sub)

    split = pd.concat(chunks, ignore_index=True)
    split["destino"] = _canon_str(split["destino"])

    chk = (
        split.groupby(key_supply, dropna=False, as_index=False)
            .agg(sum_split=("cajas_split_grado_dia", "sum"))
            .merge(supply[key_supply + ["cajas_campo_ml2_grado_dia"]], on=key_supply, how="left", validate="1:1")
    )
    chk["abs_diff"] = (chk["sum_split"] - chk["cajas_campo_ml2_grado_dia"]).abs()
    max_abs = float(chk["abs_diff"].max()) if len(chk) else 0.0
    print(f"[CHECK] split mass-balance max_abs_diff={max_abs:.12f}")
    if max_abs > 1e-9:
        raise ValueError("[FATAL] split por destino no conserva cajas (mass-balance roto).")

    split["as_of_date"] = as_of
    split["cajas_base_col_used"] = cajas_col
    split["created_at"] = created_at

    # -------------------------
    # Outputs
    # -------------------------
    out_gd_bd = split[
        [
            "fecha",
            "bloque_base",
            "variedad_canon",
            "grado",
            "Semana_Ventas",
            "destino",
            "cajas_campo_ml2_grado_dia",
            "cajas_split_grado_dia",
            "as_of_date",
            "cajas_base_col_used",
            "created_at",
        ]
    ].copy()

    GOLD.mkdir(parents=True, exist_ok=True)
    write_parquet(out_gd_bd, OUT_GD_BD)
    print(f"[OK] Wrote: {OUT_GD_BD} rows={len(out_gd_bd):,} as_of_date={as_of.date()}")

    out_dd = (
        out_gd_bd.groupby(["fecha", "destino"], dropna=False, as_index=False)
            .agg(
                cajas_campo_ml2_dia=("cajas_campo_ml2_grado_dia", "sum"),
                cajas_split_dia=("cajas_split_grado_dia", "sum"),
            )
    )
    out_dd["as_of_date"] = as_of
    out_dd["created_at"] = created_at
    write_parquet(out_dd, OUT_DD)
    print(f"[OK] Wrote: {OUT_DD} rows={len(out_dd):,}")

    out_dt = (
        out_dd.groupby(["fecha"], dropna=False, as_index=False)
            .agg(
                cajas_campo_ml2_dia=("cajas_campo_ml2_dia", "sum"),
                cajas_split_dia=("cajas_split_dia", "sum"),
            )
    )
    out_dt["as_of_date"] = as_of
    out_dt["created_at"] = created_at
    write_parquet(out_dt, OUT_DT)
    print(f"[OK] Wrote: {OUT_DT} rows={len(out_dt):,}")


if __name__ == "__main__":
    main()
