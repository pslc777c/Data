from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from common.io import read_parquet, write_parquet


B2 = Path("data/gold/features_b2_real_grado_dia_destino.parquet")
B2A = Path("data/gold/features_b2a_real_mix_dia_destino.parquet")
OUT = Path("data/gold/features_mix_real_dia_destino.parquet")


def _canon_destino(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _best_offset(b2: pd.DataFrame, b2a: pd.DataFrame, offsets=(-2, -1, 0, 1, 2)) -> int:
    """Elige offset para maximizar match (fecha_post, destino)."""
    best_off = 0
    best_m = -1
    b2_keys = set(zip(b2["fecha_post"].tolist(), b2["destino"].tolist()))

    for off in offsets:
        f = b2a["fecha_post"] + pd.Timedelta(days=off)
        keys = set(zip(f.tolist(), b2a["destino"].tolist()))
        m = len(b2_keys & keys)
        if m > best_m:
            best_m = m
            best_off = off
    return int(best_off)


def _pick_best_gi_low_bin(out: pd.DataFrame) -> tuple[str | None, float, float]:
    """
    Busca automáticamente la mejor columna b2a_share_gi_* disponible:
    - Prioriza alta cobertura (>= 5%).
    - Luego mayor varianza (para que no sea casi constante).
    Retorna (colname, coverage, variance).
    """
    cand = [c for c in out.columns if c.startswith("b2a_share_gi_")]

    # Sólo numéricas
    cand_num = []
    for c in cand:
        if pd.api.types.is_numeric_dtype(out[c]):
            cand_num.append(c)

    if not cand_num:
        return None, 0.0, 0.0

    cov = out[cand_num].notna().mean()
    cov = cov[cov >= 0.05]  # mínimo 5%
    if cov.empty:
        # si nada cumple 5%, toma la de mayor cobertura
        cov_all = out[cand_num].notna().mean()
        best = cov_all.sort_values(ascending=False).index[0]
        return best, float(cov_all[best]), float(out[best].var(skipna=True) or 0.0)

    # filtro por varianza
    var = out[cov.index].var(skipna=True)
    # score = coverage * variance (simple y estable)
    score = cov * (var.fillna(0.0))
    best = score.sort_values(ascending=False).index[0]

    return best, float(cov[best]), float(var[best] if best in var.index else 0.0)


def main() -> None:
    b2 = read_parquet(B2).copy()
    b2a = read_parquet(B2A).copy()

    for df in (b2, b2a):
        df.columns = [str(c).strip() for c in df.columns]
        df["fecha_post"] = pd.to_datetime(df["fecha_post"], errors="coerce").dt.normalize()
        df["destino"] = _canon_destino(df["destino"])

    b2 = b2.dropna(subset=["fecha_post", "destino"]).copy()
    b2a = b2a.dropna(subset=["fecha_post", "destino"]).copy()

    off = _best_offset(b2, b2a)
    if off != 0:
        print(f"[INFO] Aplicando offset B2A: {off:+d} día(s) (maximiza matches).")
        b2a["fecha_post"] = b2a["fecha_post"] + pd.Timedelta(days=off)
    else:
        print("[INFO] Offset óptimo = 0 días.")

    out = b2.merge(b2a, on=["fecha_post", "destino"], how="inner", validate="one_to_one")

    # --- interacción automática (mismatch físico-comercial)
    if "b2_share_56_75" not in out.columns:
        out["b2_share_56_75"] = np.nan

    gi_col, gi_cov, gi_var = _pick_best_gi_low_bin(out)
    if gi_col is None:
        print("[WARN] No encontré ninguna columna b2a_share_gi_* numérica para interacción.")
        out["int_largo_bajo_gi"] = 0.0
        out["int_largo_bajo_gi_gi_col"] = ""
    else:
        out["int_largo_bajo_gi"] = (
            pd.to_numeric(out["b2_share_56_75"], errors="coerce").fillna(0.0)
            * pd.to_numeric(out[gi_col], errors="coerce").fillna(0.0)
        )
        # Nota: esto es SOLO auditoría (string). No debe entrar al modelo.
        out["int_largo_bajo_gi_gi_col"] = gi_col
        print(f"[INFO] GI bin elegido para interacción: {gi_col} | coverage={gi_cov:.2%} var={gi_var:.6f}")

    write_parquet(out, OUT)
    print(f"[OK] {OUT} | rows={len(out):,} | fechas={out['fecha_post'].min()}..{out['fecha_post'].max()}")
    print("[INFO] destinos:", sorted(out["destino"].dropna().unique().tolist()))


if __name__ == "__main__":
    main()
