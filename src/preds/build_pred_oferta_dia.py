from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import read_parquet, write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    cfg = load_settings()

    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))
    preds_dir.mkdir(parents=True, exist_ok=True)

    maestro_path = silver_dir / "fact_ciclo_maestro.parquet"
    win_path = silver_dir / "milestone_window_ciclo_final.parquet"

    if not maestro_path.exists():
        raise FileNotFoundError(f"No existe: {maestro_path}")
    if not win_path.exists():
        raise FileNotFoundError(f"No existe: {win_path}. Ejecuta primero build_milestone_window_ciclo_final_with_inference.")

    c = read_parquet(maestro_path).copy()
    w = read_parquet(win_path).copy()

    # Params
    use_stage = str(cfg.get("pred_oferta", {}).get("use_stage", "HARVEST")).upper()
    if use_stage not in {"HARVEST", "HARVEST_POST"}:
        raise ValueError("pred_oferta.use_stage debe ser 'HARVEST' o 'HARVEST_POST'.")

    # ---- Maestro
    c.columns = [str(x).strip() for x in c.columns]
    c["ciclo_id"] = c["ciclo_id"].astype(str)

    # variedad + map
    if "variedad" not in c.columns:
        c["variedad"] = "UNKNOWN"
    c["variedad"] = _canon_str(c["variedad"])

    var_map = (cfg.get("mappings", {}).get("variedad_map", {}) or {})
    var_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in var_map.items()}
    c["variedad"] = c["variedad"].map(lambda x: var_map.get(x, x))

    for col in ["tipo_sp", "area", "estado"]:
        if col not in c.columns:
            c[col] = "UNKNOWN"
        c[col] = _canon_str(c[col])

    # tallos_proy
    if "tallos_proy" not in c.columns:
        raise ValueError("fact_ciclo_maestro no tiene 'tallos_proy' (requerido).")
    c["tallos_proy"] = pd.to_numeric(c["tallos_proy"], errors="coerce").fillna(0.0)

    # bloque y canónico bloque_base
    if "bloque_base" not in c.columns:
        raise ValueError("fact_ciclo_maestro debe tener 'bloque_base' (llave canónica).")
    c["bloque_base"] = pd.to_numeric(c["bloque_base"], errors="coerce").astype("Int64")

    if "bloque" in c.columns:
        c["bloque"] = pd.to_numeric(c["bloque"], errors="coerce").astype("Int64")
    else:
        c["bloque"] = c["bloque_base"]

    # ---- Windows
    w["ciclo_id"] = w["ciclo_id"].astype(str)
    w["stage"] = _canon_str(w["stage"])
    w["start_date"] = _to_date(w["start_date"])
    w["end_date"] = _to_date(w["end_date"])

    w = w[w["start_date"].notna() & w["end_date"].notna()].copy()
    w = w[w["start_date"] <= w["end_date"]].copy()

    # Tomar harvest_start/end por ciclo (desde ventana HARVEST)
    hw = w[w["stage"].eq("HARVEST")][["ciclo_id", "start_date", "end_date"]].copy()
    hw = hw.rename(columns={"start_date": "harvest_start", "end_date": "harvest_end_eff"})
    hw = hw.drop_duplicates("ciclo_id")

    # Expandir días desde windows (NO OUT)
    parts = []
    for r in w.itertuples(index=False):
        if pd.isna(r.start_date) or pd.isna(r.end_date):
            continue
        dates = pd.date_range(r.start_date, r.end_date, freq="D")
        parts.append(
            pd.DataFrame({"ciclo_id": r.ciclo_id, "fecha": dates, "stage": r.stage})
        )

    if not parts:
        raise ValueError("No hay ventanas para expandir (w vacío).")

    grid = pd.concat(parts, ignore_index=True)

    # Join maestro + harvest meta
    out = (
        grid.merge(c[["ciclo_id", "bloque", "bloque_base", "variedad", "tipo_sp", "area", "estado", "tallos_proy"]],
                  on="ciclo_id", how="left")
            .merge(hw, on="ciclo_id", how="left")
    )

    # Compat: bloque_padre = bloque_base
    out["bloque_padre"] = out["bloque_base"]

    # n_harvest_days
    out["n_harvest_days"] = (out["harvest_end_eff"] - out["harvest_start"]).dt.days + 1
    out["n_harvest_days"] = pd.to_numeric(out["n_harvest_days"], errors="coerce").astype("Int64")

    # tallos_dia (solo si hay harvest)
    out["tallos_dia"] = np.where(
        out["n_harvest_days"].notna() & (out["n_harvest_days"].astype(float) > 0),
        out["tallos_proy"].astype(float) / out["n_harvest_days"].astype(float),
        0.0,
    )

    # máscara stage
    st = out["stage"].astype(str).str.upper()
    if use_stage == "HARVEST":
        mask_stage = st.eq("HARVEST")
    else:
        mask_stage = st.isin(["HARVEST", "POST"])

    # oferta: solo dentro harvest window y stage permitido
    in_window = out["harvest_start"].notna() & (out["fecha"] >= out["harvest_start"]) & (out["fecha"] <= out["harvest_end_eff"])
    out["tallos_pred"] = np.where(mask_stage & in_window, out["tallos_dia"], 0.0)

    out_pred = out[[
        "ciclo_id", "fecha",
        "bloque", "bloque_padre", "variedad", "tipo_sp", "area", "estado",
        "stage",
        "harvest_start", "harvest_end_eff", "n_harvest_days",
        "tallos_proy",
        "tallos_pred",
    ]].copy()

    out_pred["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = preds_dir / "pred_oferta_dia.parquet"
    write_parquet(out_pred, out_path)

    print(f"OK: pred_oferta_dia={len(out_pred):,} filas -> {out_path}")
    print("Stage counts:\n", out_pred["stage"].value_counts(dropna=False).to_string())
    # KPI correcto: ceros SOLO en HARVEST
    harv = out_pred[out_pred["stage"].astype(str).str.upper().eq("HARVEST")]
    if len(harv) > 0:
        print("ratio tallos_pred==0 en HARVEST:", round((harv["tallos_pred"].fillna(0).eq(0)).mean() * 100, 2))
    print("ratio tallos_pred==0 total:", round((out_pred["tallos_pred"].fillna(0).eq(0)).mean() * 100, 2))


if __name__ == "__main__":
    main()
