from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from common.io import read_parquet, write_parquet


IN_B2_MIX = Path("data/gold/features_mix_pred_dia_destino.parquet")   # ya tiene b2_*
IN_B2A = Path("data/gold/features_b2a_mix_pred_dia_destino.parquet")  # b2a_* + tot__
OUT = Path("data/gold/features_mix_pred_dia_destino.parquet")         # overwrite


def _canon_destino(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _pick_best_gi_low_bin(out: pd.DataFrame) -> tuple[str | None, float, float]:
    cand = [c for c in out.columns if c.startswith("b2a_share_gi_")]
    cand_num = [c for c in cand if pd.api.types.is_numeric_dtype(out[c])]
    if not cand_num:
        return None, 0.0, 0.0

    cov = out[cand_num].notna().mean()
    cov = cov[cov >= 0.05]
    if cov.empty:
        cov_all = out[cand_num].notna().mean()
        best = cov_all.sort_values(ascending=False).index[0]
        return best, float(cov_all[best]), float(out[best].var(skipna=True) or 0.0)

    var = out[cov.index].var(skipna=True)
    score = cov * (var.fillna(0.0))
    best = score.sort_values(ascending=False).index[0]
    return best, float(cov[best]), float(var[best] if best in var.index else 0.0)


def main() -> None:
    b2 = read_parquet(IN_B2_MIX).copy()
    b2.columns = [str(c).strip() for c in b2.columns]
    b2["fecha_post"] = pd.to_datetime(b2["fecha_post"], errors="coerce").dt.normalize()
    b2["destino"] = _canon_destino(b2["destino"])

    b2a = read_parquet(IN_B2A).copy()
    b2a.columns = [str(c).strip() for c in b2a.columns]
    b2a["fecha_post"] = pd.to_datetime(b2a["fecha_post"], errors="coerce").dt.normalize()
    b2a["destino"] = _canon_destino(b2a["destino"])

    # ---------------------------------------------------------
    # FIX: eliminar overlaps para evitar MergeError por duplicados
    # ---------------------------------------------------------
    keys = {"fecha_post", "destino"}
    overlap = sorted((set(b2.columns) & set(b2a.columns)) - keys)
    if overlap:
        print(f"[INFO] Dropeando {len(overlap)} columnas overlap en LEFT (ya venían de corridas previas).")
        b2 = b2.drop(columns=overlap)

    # Merge seguro (sin sufijos)
    out = b2.merge(b2a, on=["fecha_post", "destino"], how="left", validate="many_to_one")

    # --- interacción automática
    if "b2_share_56_75" not in out.columns:
        out["b2_share_56_75"] = np.nan

    gi_col, gi_cov, gi_var = _pick_best_gi_low_bin(out)
    if gi_col is None:
        print("[WARN] No encontré b2a_share_gi_* numérica en PRED; int=0.")
        out["int_largo_bajo_gi"] = 0.0
        out["int_largo_bajo_gi_gi_col"] = ""
    else:
        out["int_largo_bajo_gi"] = (
            pd.to_numeric(out["b2_share_56_75"], errors="coerce").fillna(0.0)
            * pd.to_numeric(out[gi_col], errors="coerce").fillna(0.0)
        )
        out["int_largo_bajo_gi_gi_col"] = gi_col  # auditoría (no usar en train)
        print(f"[INFO] PRED GI bin elegido: {gi_col} | coverage={gi_cov:.2%} var={gi_var:.6f}")

    write_parquet(out, OUT)
    print(f"[OK] {OUT} | rows={len(out):,} | cols={len(out.columns)}")
    print("[INFO] fechas:", out["fecha_post"].min(), out["fecha_post"].max())
    print("[INFO] destinos:", sorted(out["destino"].dropna().unique().tolist()))


if __name__ == "__main__":
    main()
