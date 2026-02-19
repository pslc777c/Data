# src/silver/build_patch_ciclo_maestro_from_pred_tallos.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_dt_norm(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_num(s: pd.Series) -> pd.Series:
    ss = s.copy()
    if ss.dtype == object or pd.api.types.is_string_dtype(ss):
        ss = (
            ss.astype("string")
              .str.replace(" ", "", regex=False)
              .str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(ss, errors="coerce")


def _build_fin_mask(df: pd.DataFrame, fin_col: str, fin_value: int) -> pd.Series:
    """
    - Si fin_col es datetime -> mask = notna()
    - Si fin_col es num/bool -> mask = (== fin_value)
    - Si es object -> intenta num; si no, intenta datetime; fallback strings
    """
    if fin_col not in df.columns:
        raise ValueError(f"No existe columna '{fin_col}' en fact_ciclo_maestro.")

    s = df[fin_col]

    if pd.api.types.is_datetime64_any_dtype(s):
        return s.notna()

    sn = _to_num(s)
    if sn.notna().sum() > 0:
        return sn.fillna(0).astype("Int64").eq(int(fin_value))

    sd = pd.to_datetime(s, errors="coerce")
    if sd.notna().sum() > 0:
        return sd.notna()

    st = s.astype("string").str.strip().str.lower()
    return st.isin([str(fin_value), "true", "si", "sí", "yes", "y"])


def main() -> None:
    cfg = load_settings()

    # dirs
    data_root = Path(cfg["paths"].get("data_root", "data"))
    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir = Path(cfg["paths"].get("preds", "data/preds"))

    # resolvemos rutas relativas al repo root actual (CWD = repo root usualmente)
    silver_dir = Path(silver_dir)
    preds_dir = Path(preds_dir)

    scfg = (cfg.get("silver", {}) or {}).get("patch_ciclo_maestro_from_pred_tallos", {}) or {}
    enabled = bool(scfg.get("enabled", True))
    if not enabled:
        print("[INFO] patch_ciclo_maestro_from_pred_tallos: disabled")
        return

    in_ciclo_rel = scfg.get("in_ciclo", "fact_ciclo_maestro.parquet")
    in_pred_rel = scfg.get("in_pred_tallos", "../preds/pred_tallos_ciclo.parquet")
    out_report_rel = scfg.get("out_report", "fact_ciclo_maestro_patch_pred_tallos_report.parquet")

    ciclo_id_col = scfg.get("ciclo_id_col", "ciclo_id")
    fin_col = scfg.get("fin_col", "fecha_fin_cosecha")
    fin_value = int(scfg.get("fin_value", 1))
    tallos_new_col = scfg.get("tallos_new_col", "tallos_proy_n")

    require_positive = bool(scfg.get("require_positive", True))

    in_ciclo = (silver_dir / in_ciclo_rel).resolve()
    # in_pred puede venir como "../preds/..." relativo a silver_dir
    in_pred = (silver_dir / in_pred_rel).resolve()
    out_report = (silver_dir / out_report_rel).resolve()

    if not in_ciclo.exists():
        raise FileNotFoundError(f"No existe: {in_ciclo}")

    # =========================
    # CONDICIÓN CLAVE: si preds no existe -> NO HACER NADA
    # =========================
    if not in_pred.exists():
        print(f"[INFO] No existe pred_tallos_ciclo -> SKIP. ({in_pred})")
        # igual escribimos reporte “skipped” para trazabilidad y para que el step tenga output
        rep = pd.DataFrame([{
            "patched_at": datetime.now().isoformat(timespec="seconds"),
            "status": "skipped_missing_pred",
            "in_ciclo": str(in_ciclo),
            "in_pred": str(in_pred),
            "rows_ciclo": None,
            "rows_pred": None,
            "rows_matched": None,
            "rows_replaced": None,
        }])
        out_report.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(rep, out_report)
        print(f"[INFO] report: {out_report}")
        return

    # =========================
    # LECTURAS
    # =========================
    ciclo = _norm_cols(pd.read_parquet(in_ciclo))
    pred = _norm_cols(pd.read_parquet(in_pred))

    # validación columnas
    if ciclo_id_col not in ciclo.columns:
        raise ValueError(f"fact_ciclo_maestro: falta columna '{ciclo_id_col}'")
    if "tallos_proy" not in ciclo.columns:
        raise ValueError("fact_ciclo_maestro: falta columna 'tallos_proy'")
    if fin_col not in ciclo.columns:
        raise ValueError(f"fact_ciclo_maestro: falta columna '{fin_col}'")

    if ciclo_id_col not in pred.columns:
        raise ValueError(f"pred_tallos_ciclo: falta columna '{ciclo_id_col}'")
    if tallos_new_col not in pred.columns:
        raise ValueError(f"pred_tallos_ciclo: falta columna '{tallos_new_col}'")

    # normalizar tipos
    ciclo[ciclo_id_col] = ciclo[ciclo_id_col].astype("string").str.strip()
    pred[ciclo_id_col] = pred[ciclo_id_col].astype("string").str.strip()

    ciclo["tallos_proy"] = _to_num(ciclo["tallos_proy"])
    pred[tallos_new_col] = _to_num(pred[tallos_new_col])

    # por seguridad: 1 ciclo_id -> 1 valor (tomamos max no nulo)
    pred_slim = pred[[ciclo_id_col, tallos_new_col]].copy()
    pred_slim = (
        pred_slim.groupby(ciclo_id_col, as_index=False)
                 .agg(**{tallos_new_col: (tallos_new_col, "max")})
    )

    # merge
    merged = ciclo.merge(pred_slim, on=ciclo_id_col, how="left")

    fin_mask = _build_fin_mask(merged, fin_col=fin_col, fin_value=fin_value)
    # fin_mask = True  -> tiene fecha_fin_cosecha / fin=1 (ciclo cerrado)
    # open_mask = True -> NO tiene fecha_fin_cosecha / fin!=1 (ciclo abierto)
    open_mask = ~fin_mask

    has_new = merged[tallos_new_col].notna()
    if require_positive:
        has_new = has_new & (merged[tallos_new_col] > 0)

    # ✅ Reemplazar solo si está ABIERTO (sin fecha_fin_cosecha) y hay tallos_proy_n
    mask_replace = has_new & open_mask


    rows_ciclo = int(len(merged))
    rows_pred = int(len(pred_slim))
    rows_matched = int(merged[tallos_new_col].notna().sum())
    rows_replaced = int(mask_replace.sum())

    if rows_replaced > 0:
        merged.loc[mask_replace, "tallos_proy"] = merged.loc[mask_replace, tallos_new_col]

    # limpiar col auxiliar
    merged = merged.drop(columns=[tallos_new_col])

    # overwrite maestro
    write_parquet(merged, in_ciclo)

    # reporte
    rep = pd.DataFrame([{
        "patched_at": datetime.now().isoformat(timespec="seconds"),
        "status": "patched",
        "rows_ciclo": rows_ciclo,
        "rows_pred": rows_pred,
        "rows_matched_new": rows_matched,
        "rows_replaced": rows_replaced,
        "require_positive": require_positive,
        "fin_col": fin_col,
        "fin_value": fin_value,
        "tallos_new_col": tallos_new_col,
        "in_ciclo": str(in_ciclo),
        "in_pred": str(in_pred),
        "out_report": str(out_report),
    }])

    out_report.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(rep, out_report)

    print(f"OK patch ciclo_maestro from preds: replaced={rows_replaced} matched_new={rows_matched} rows={rows_ciclo}")
    print(f" - updated: {in_ciclo}")
    print(f" - report : {out_report}")


if __name__ == "__main__":
    main()
