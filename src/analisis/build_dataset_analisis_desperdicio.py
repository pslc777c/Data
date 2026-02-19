from pathlib import Path
import pandas as pd
import numpy as np
from common.io import read_parquet, write_parquet


# =============================
# ROOT DINÁMICO (robusto)
# =============================
def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"

SILVER_BAL = DATA / "silver" / "balanzas"

B2_PATH = DATA / "silver" / "fact_hidratacion_real_post_grado_destino.parquet"
B2A_PATH = SILVER_BAL / "silver_b2a_sku_gradoideal_dia_destino_variedad.parquet"
MERMA_PATH = DATA / "silver" / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_PATH = DATA / "gold" / "dataset_analisis_desperdicio_dia_destino.parquet"


# =============================
# CONFIG
# =============================
GI_BIN_SIZE = 5  # <-- binning robusto


def main():

    # =====================================================
    # 1️⃣ B2 DISTRIBUCIÓN (Entrada proceso)
    # =====================================================
    b2 = read_parquet(B2_PATH).copy()

    b2["Fecha"] = pd.to_datetime(b2["fecha_post"]).dt.normalize()
    b2["Destino"] = b2["destino"].astype(str).str.upper().str.strip()
    b2["Grado"] = b2["grado"].astype(str).str.upper().str.strip()

    b2["Peso_B2_kg"] = pd.to_numeric(b2["peso_post_g"], errors="coerce") / 1000.0
    b2 = b2.loc[b2["Peso_B2_kg"] > 0].copy()

    total_b2 = (
        b2.groupby(["Fecha", "Destino"], as_index=False)
        .agg(total_b2_kg=("Peso_B2_kg", "sum"))
    )

    b2 = b2.merge(total_b2, on=["Fecha", "Destino"], how="left")
    b2["share_b2_grado"] = b2["Peso_B2_kg"] / b2["total_b2_kg"]

    b2_dist = (
        b2.groupby(["Fecha", "Destino", "Grado"], as_index=False)
        .agg(share_b2_grado=("share_b2_grado", "sum"))
    )

    b2_pivot = b2_dist.pivot_table(
        index=["Fecha", "Destino"],
        columns="Grado",
        values="share_b2_grado",
        fill_value=0,
    )

    # =====================================================
    # 2️⃣ B2A DISTRIBUCIÓN (Salida proceso)
    # =====================================================
    b2a = read_parquet(B2A_PATH).copy()

    b2a["Fecha"] = pd.to_datetime(b2a["Fecha"]).dt.normalize()
    b2a["Destino"] = b2a["Destino"].astype(str).str.upper().str.strip()

    # Solo filas válidas
    b2a = b2a.loc[
        b2a["SKU"].notna() &
        b2a["Grado_Ideal"].notna() &
        (b2a["Peso_Balanza_2A"] > 0)
    ].copy()

    # BINNING robusto de Grado_Ideal
    b2a["GI_bin"] = (b2a["Grado_Ideal"] / GI_BIN_SIZE).round() * GI_BIN_SIZE

    # Crear mix_key estable
    b2a["mix_key"] = (
        b2a["SKU"].astype(int).astype(str) +
        "_GI_" +
        b2a["GI_bin"].astype(int).astype(str)
    )

    total_b2a = (
        b2a.groupby(["Fecha", "Destino"], as_index=False)
        .agg(total_b2a_kg=("Peso_Balanza_2A", "sum"))
    )

    b2a = b2a.merge(total_b2a, on=["Fecha", "Destino"], how="left")
    b2a["share_b2a_mix"] = b2a["Peso_Balanza_2A"] / b2a["total_b2a_kg"]

    b2a_dist = (
        b2a.groupby(["Fecha", "Destino", "mix_key"], as_index=False)
        .agg(share_b2a_mix=("share_b2a_mix", "sum"))
    )

    b2a_pivot = b2a_dist.pivot_table(
        index=["Fecha", "Destino"],
        columns="mix_key",
        values="share_b2a_mix",
        fill_value=0,
    )

    # =====================================================
    # 3️⃣ JOIN DISTRIBUCIONES
    # =====================================================
    dataset = b2_pivot.join(b2a_pivot, how="left")
    dataset = dataset.fillna(0)
    dataset = dataset.reset_index()

    # =====================================================
    # 4️⃣ MERMAS
    # =====================================================
    merma = read_parquet(MERMA_PATH).copy()

    merma["Fecha"] = pd.to_datetime(merma["fecha_post"]).dt.normalize()
    merma["Destino"] = merma["destino"].astype(str).str.upper().str.strip()

    merma = merma[["Fecha", "Destino", "desp_pct", "ajuste"]]

    dataset = dataset.merge(merma, on=["Fecha", "Destino"], how="inner")

    # =====================================================
    # 5️⃣ OUTPUT
    # =====================================================
    write_parquet(dataset, OUT_PATH)
    print(f"[OK] Dataset maestro guardado en {OUT_PATH}")
    print("Shape final:", dataset.shape)


if __name__ == "__main__":
    main()
