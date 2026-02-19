from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def main() -> None:
    cfg = load_settings()
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Fuente: fact real (mejor) o dim por fecha_cosecha si ya existe
    fact_path = silver_dir / "fact_hidratacion_real_post_grado_destino.parquet"
    dim_fc_path = silver_dir / "dim_hidratacion_fecha_cosecha_grado_destino.parquet"

    if fact_path.exists():
        df = pd.read_parquet(fact_path)
        df.columns = [str(c).strip() for c in df.columns]

        need = {"grado", "destino", "hidr_pct", "peso_base_g"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"fact_hidratacion_real_post_grado_destino sin columnas: {sorted(miss)}")

        df["grado"] = _canon_int(df["grado"])
        df["destino"] = df["destino"].astype(str).str.upper().str.strip()
        df["hidr_pct"] = pd.to_numeric(df["hidr_pct"], errors="coerce")
        df["peso_base_g"] = pd.to_numeric(df["peso_base_g"], errors="coerce").fillna(0.0)

        df = df[df["grado"].notna() & df["destino"].notna() & df["hidr_pct"].notna()].copy()
        if df.empty:
            raise ValueError("fact hidratación quedó vacío tras filtros.")

        # Peso como ponderador (si no, usa count)
        df["_w"] = df["peso_base_g"].clip(lower=0.0)

        # Cuantiles ponderados: aproximación robusta vía expansión por bins NO es viable.
        # Aquí usamos cuantiles simples + mediana ponderada aproximada por ranking con peso.
        # (Suficiente para baseline seed; ML vendrá después.)
        out_rows = []
        for (g, d), gdf in df.groupby(["grado", "destino"], dropna=False):
            x = gdf["hidr_pct"].to_numpy(dtype=float)
            w = gdf["_w"].to_numpy(dtype=float)
            if len(x) == 0:
                continue
            # fallback si pesos degeneran
            if np.nansum(w) <= 0:
                med = float(np.nanmedian(x))
            else:
                # mediana ponderada
                o = np.argsort(x)
                x2 = x[o]
                w2 = w[o]
                cw = np.cumsum(np.nan_to_num(w2, nan=0.0))
                cut = 0.5 * cw[-1]
                med = float(x2[np.searchsorted(cw, cut, side="left")])

            p25 = float(np.nanpercentile(x, 25))
            p75 = float(np.nanpercentile(x, 75))
            out_rows.append(
                {
                    "grado": int(g),
                    "destino": str(d),
                    "n": int(len(gdf)),
                    "peso_base_g_sum": float(np.nansum(w)),
                    "hidr_pct_med": med,
                    "hidr_pct_p25": p25,
                    "hidr_pct_p75": p75,
                    "factor_hidr_med": 1.0 + med,
                    "factor_hidr_p25": 1.0 + p25,
                    "factor_hidr_p75": 1.0 + p75,
                }
            )

        out = pd.DataFrame(out_rows)
    elif dim_fc_path.exists():
        # fallback si no tienes fact
        df = pd.read_parquet(dim_fc_path)
        df.columns = [str(c).strip() for c in df.columns]

        need = {"grado", "destino", "factor_hidr"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"dim_hidratacion_fecha_cosecha_grado_destino sin columnas: {sorted(miss)}")

        df["grado"] = _canon_int(df["grado"])
        df["destino"] = df["destino"].astype(str).str.upper().str.strip()
        df["factor_hidr"] = pd.to_numeric(df["factor_hidr"], errors="coerce")
        df = df[df["grado"].notna() & df["destino"].notna() & df["factor_hidr"].notna()].copy()
        df["hidr_pct"] = df["factor_hidr"] - 1.0

        out = (
            df.groupby(["grado","destino"], dropna=False)
              .agg(
                  n=("factor_hidr","size"),
                  factor_hidr_med=("factor_hidr","median"),
                  factor_hidr_p25=("factor_hidr", lambda x: float(np.nanpercentile(x, 25))),
                  factor_hidr_p75=("factor_hidr", lambda x: float(np.nanpercentile(x, 75))),
                  hidr_pct_med=("hidr_pct","median"),
              )
              .reset_index()
        )
    else:
        raise FileNotFoundError("No existe fact ni dim de hidratación para construir baseline.")

    out["created_at"] = datetime.now().isoformat(timespec="seconds")
    out_path = silver_dir / "dim_hidratacion_baseline_grado_destino.parquet"
    write_parquet(out, out_path)

    print(f"OK -> {out_path} | rows={len(out):,}")
    if "factor_hidr_med" in out.columns:
        print("factor_hidr_med describe:\n", pd.to_numeric(out["factor_hidr_med"], errors="coerce").describe().to_string())


if __name__ == "__main__":
    main()
