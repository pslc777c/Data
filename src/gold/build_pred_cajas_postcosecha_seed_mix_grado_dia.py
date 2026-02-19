from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


# =============================================================================
# INPUTS
# =============================================================================
IN_CAJAS_GRADO = Path("data/gold/pred_cajas_grado_dia_ml1_full.parquet")
IN_MIX = Path("data/silver/dim_mix_proceso_semana.parquet")

# Hidratación: por fecha_cosecha (que en seed = fecha de cosecha proyectada)
IN_HIDR_FC = Path("data/silver/dim_hidratacion_fecha_cosecha_grado_destino.parquet")

# DH baseline por grado-destino
IN_DH = Path("data/silver/dim_dh_baseline_grado_destino.parquet")

# Merma + ajuste por fecha_post (día de proceso) y destino
IN_MERMA_AJUSTE = Path("data/silver/dim_mermas_ajuste_fecha_post_destino.parquet")

# =============================================================================
# OUTPUTS
# =============================================================================
OUT_GD_BD = Path("data/gold/pred_poscosecha_seed_grado_dia_bloque_destino.parquet")
OUT_DD = Path("data/gold/pred_poscosecha_seed_dia_destino.parquet")
OUT_DT = Path("data/gold/pred_poscosecha_seed_dia_total.parquet")


# =============================================================================
# HELPERS
# =============================================================================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


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
    Debe ser CONSISTENTE con ventas:
      ventas:
        Fecha_Clasificacion = Fecha - 2
        Semana_Ventas = semana_ventas(Fecha_Clasificacion)
        semana_ventas(x): (x + 2) -> %U
    Entonces para fecha de cosecha (equivale a Fecha_Clasificacion):
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
        dup_rate = float(dup / max(len(df), 1))
        print(f"[WARN] {name} tiene duplicados por {keys} -> dup_count={dup:,} dup_rate={dup_rate:.4%}")
        print("[WARN] ejemplos dup (top 10):")
        print(df.loc[df.duplicated(subset=keys, keep=False), keys].head(10).to_string(index=False))

        agg = {c: "sum" for c in val_cols if c in df.columns}
        # Mantener cualquier columna extra (first) si existe
        extra = [c for c in df.columns if c not in keys and c not in agg]
        for c in extra:
            agg[c] = "first"

        out = df.groupby(keys, dropna=False, as_index=False).agg(agg)
        print(f"[FIX] {name} colapsado por {keys} (sum) -> rows={len(out):,}")
        return out
    return df


def _collapse_median(df: pd.DataFrame, keys: list[str], cols: list[str], name: str) -> pd.DataFrame:
    agg = {}
    for c in cols:
        if c in df.columns:
            agg[c] = "median"
    # Mantener created_at si existe
    if "created_at" in df.columns:
        agg["created_at"] = "first"
    out = df.groupby(keys, dropna=False, as_index=False).agg(agg)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    created_at = pd.Timestamp.utcnow()

    # -------------------------
    # 1) Load cajas ML1 (grado/día)
    # -------------------------
    cajas = read_parquet(IN_CAJAS_GRADO).copy()
    cajas.columns = [str(c).strip() for c in cajas.columns]

    need_cajas = ["fecha", "bloque_base", "variedad_canon", "grado", "cajas_ml1_grado_dia"]
    _require(cajas, need_cajas, "pred_cajas_grado_dia_ml1_full")

    cajas["fecha"] = _to_date(cajas["fecha"])
    cajas["bloque_base"] = _canon_int(cajas["bloque_base"])
    cajas["grado"] = _canon_int(cajas["grado"])
    cajas["variedad_canon"] = _canon_str(cajas["variedad_canon"])
    cajas["cajas_ml1_grado_dia"] = pd.to_numeric(cajas["cajas_ml1_grado_dia"], errors="coerce").fillna(0.0)

    key_supply = ["fecha", "bloque_base", "variedad_canon", "grado"]
    cajas = _collapse_sum(cajas, key_supply, ["cajas_ml1_grado_dia"], "cajas")

    # -------------------------
    # 2) Load mix por Semana_Ventas
    # -------------------------
    mix = read_parquet(IN_MIX).copy()
    mix.columns = [str(c).strip() for c in mix.columns]

    _require(mix, ["Semana_Ventas", "W_Blanco", "W_Arcoiris", "W_Tinturado"], "dim_mix_proceso_semana")

    mix["Semana_Ventas"] = mix["Semana_Ventas"].astype(str).str.strip()
    for c in ["W_Blanco", "W_Arcoiris", "W_Tinturado"]:
        mix[c] = pd.to_numeric(mix[c], errors="coerce")

    # DEFAULT row
    mix_def = mix[mix["Semana_Ventas"].eq("DEFAULT")].copy()
    if mix_def.empty:
        raise ValueError("dim_mix_proceso_semana no trae Semana_Ventas='DEFAULT' (fallback obligatorio).")
    mix_def_row = mix_def.iloc[0]
    def_w = {
        "W_Blanco": _safe_float(mix_def_row["W_Blanco"], 0.80),
        "W_Arcoiris": _safe_float(mix_def_row["W_Arcoiris"], 0.10),
        "W_Tinturado": _safe_float(mix_def_row["W_Tinturado"], 0.10),
    }
    sdef = def_w["W_Blanco"] + def_w["W_Arcoiris"] + def_w["W_Tinturado"]
    if sdef > 0:
        def_w = {k: v / sdef for k, v in def_w.items()}
    else:
        def_w = {"W_Blanco": 0.80, "W_Arcoiris": 0.10, "W_Tinturado": 0.10}

    # mapeo destino (proceso)
    DESTS = [
        ("BLANCO", "W_Blanco"),
        ("ARCOIRIS", "W_Arcoiris"),
        ("TINTURADO", "W_Tinturado"),
    ]

    # -------------------------
    # 3) Attach Semana_Ventas to supply + join mix (fallback DEFAULT)
    # -------------------------
    supply = cajas.copy()
    supply["Semana_Ventas"] = _semana_ventas_from_fecha_cosecha(supply["fecha"])

    mix_take = mix[mix["Semana_Ventas"].ne("DEFAULT")].copy()
    mix_take = mix_take.drop_duplicates(subset=["Semana_Ventas"], keep="last")

    supply = supply.merge(mix_take, on="Semana_Ventas", how="left")

    # fill missing with DEFAULT medians
    for _, wcol in DESTS:
        supply[wcol] = supply[wcol].fillna(def_w[wcol])

    # renormalize weights per row (defensive)
    ws = supply[[w for _, w in DESTS]].sum(axis=1)
    ws = ws.replace(0, np.nan)
    for _, wcol in DESTS:
        supply[wcol] = np.where(ws.notna(), supply[wcol] / ws, def_w[wcol])

    cov_mix = float(supply["W_Blanco"].notna().mean())
    print(f"[MIX] coverage semana->mix (W_Blanco notna): {cov_mix:.4f}")

    # -------------------------
    # 4) Split supply by destino using weights (mass-balance exact)
    # -------------------------
    chunks = []
    for dest, wcol in DESTS:
        sub = supply[key_supply + ["Semana_Ventas", "cajas_ml1_grado_dia", wcol]].copy()
        sub = sub.rename(columns={wcol: "w_dest"})
        sub["destino"] = dest
        sub["cajas_split_grado_dia"] = sub["cajas_ml1_grado_dia"].astype(float) * sub["w_dest"].astype(float)
        chunks.append(sub)

    split = pd.concat(chunks, ignore_index=True)
    split["destino"] = _canon_str(split["destino"])

    # Mass-balance check: sum_split == cajas original
    chk = (
        split.groupby(key_supply, dropna=False, as_index=False)
             .agg(sum_split=("cajas_split_grado_dia", "sum"))
        .merge(
            supply[key_supply + ["cajas_ml1_grado_dia"]],
            on=key_supply,
            how="left",
            validate="1:1",
        )
    )
    chk["abs_diff"] = (chk["sum_split"] - chk["cajas_ml1_grado_dia"]).abs()
    max_abs = float(chk["abs_diff"].max()) if len(chk) else 0.0
    print(f"[CHECK] split mass-balance max_abs_diff={max_abs:.10f}")
    if max_abs > 1e-9:
        print("[DEBUG] ejemplos mismatch (top 20):")
        print(
            chk.sort_values("abs_diff", ascending=False)
               .head(20)
               .to_string(index=False)
        )
        raise ValueError("[FATAL] split por destino no conserva cajas.")

    # -------------------------
    # 5) Attach DH baseline (grado,destino) -> fecha_post_pred
    # -------------------------
    dh = read_parquet(IN_DH).copy()
    dh.columns = [str(c).strip() for c in dh.columns]
    _require(dh, ["grado", "destino", "dh_dias_med"], "dim_dh_baseline_grado_destino")

    dh["grado"] = _canon_int(dh["grado"])
    dh["destino"] = _canon_str(dh["destino"])
    dh["dh_dias_med"] = pd.to_numeric(dh["dh_dias_med"], errors="coerce")

    dh2 = dh.groupby(["grado", "destino"], dropna=False, as_index=False).agg(
        dh_dias_med=("dh_dias_med", "median")
    )

    split = split.merge(dh2, on=["grado", "destino"], how="left")

    dh_global = _safe_float(np.nanmedian(dh2["dh_dias_med"].values), default=7.0)
    split["dh_dias"] = (
        pd.to_numeric(split["dh_dias_med"], errors="coerce")
        .fillna(dh_global)
        .round()
        .astype("Int64")
        .clip(lower=0, upper=30)
    )
    split["fecha_post_pred"] = _to_date(split["fecha"]) + pd.to_timedelta(split["dh_dias"].astype(float), unit="D")

    # -------------------------
    # 6) Hidratación seed: prefer match por fecha_cosecha=fecha,grado,destino; fallback medianas
    # -------------------------
    hidr = read_parquet(IN_HIDR_FC).copy()
    hidr.columns = [str(c).strip() for c in hidr.columns]
    _require(hidr, ["fecha_cosecha", "grado", "destino", "factor_hidr"], "dim_hidratacion_fecha_cosecha_grado_destino")

    hidr["fecha_cosecha"] = _to_date(hidr["fecha_cosecha"])
    hidr["grado"] = _canon_int(hidr["grado"])
    hidr["destino"] = _canon_str(hidr["destino"])
    hidr["factor_hidr"] = pd.to_numeric(hidr["factor_hidr"], errors="coerce")

    # collapse to 1 row per (fecha_cosecha, grado, destino)
    hidr_fc = _collapse_median(hidr, ["fecha_cosecha", "grado", "destino"], ["factor_hidr"], "hidr_fc")

    # baseline by (grado,destino)
    hidr_gd = _collapse_median(hidr, ["grado", "destino"], ["factor_hidr"], "hidr_gd")

    # join fc first
    split = split.merge(
        hidr_fc.rename(columns={"factor_hidr": "factor_hidr_fc"}),
        left_on=["fecha", "grado", "destino"],
        right_on=["fecha_cosecha", "grado", "destino"],
        how="left",
    )

    # join gd baseline
    split = split.merge(
        hidr_gd.rename(columns={"factor_hidr": "factor_hidr_gd"}),
        on=["grado", "destino"],
        how="left",
    )

    hidr_global = _safe_float(np.nanmedian(hidr["factor_hidr"].values), default=1.45)

    split["factor_hidr_seed"] = (
        pd.to_numeric(split["factor_hidr_fc"], errors="coerce")
        .fillna(pd.to_numeric(split["factor_hidr_gd"], errors="coerce"))
        .fillna(hidr_global)
        .clip(lower=0.60, upper=3.00)
    )

    miss_hidr_fc = int(pd.to_numeric(split["factor_hidr_fc"], errors="coerce").isna().sum())
    print(f"[CHECK] miss_hidr_fc={miss_hidr_fc:,} (sin match por fecha_cosecha; se usó baseline grado-destino o global)")

    # -------------------------
    # 7) Merma + ajuste seed por fecha_post_pred + destino (fallback med destino -> med global)
    #     *** CAMBIO CLAVE: usar factor_ajuste y MULTIPLICAR (NO dividir por ajuste) ***
    # -------------------------
    mer = read_parquet(IN_MERMA_AJUSTE).copy()
    mer.columns = [str(c).strip() for c in mer.columns]
    _require(mer, ["fecha_post", "destino", "factor_desp", "factor_ajuste"], "dim_mermas_ajuste_fecha_post_destino")

    mer["fecha_post"] = _to_date(mer["fecha_post"])
    mer["destino"] = _canon_str(mer["destino"])
    mer["factor_desp"] = pd.to_numeric(mer["factor_desp"], errors="coerce")
    mer["factor_ajuste"] = pd.to_numeric(mer["factor_ajuste"], errors="coerce")

    mer2 = _collapse_median(mer, ["fecha_post", "destino"], ["factor_desp", "factor_ajuste"], "mer2")

    split = split.merge(
        mer2.rename(columns={"fecha_post": "fecha_post_key"}),
        left_on=["fecha_post_pred", "destino"],
        right_on=["fecha_post_key", "destino"],
        how="left",
    )

    med_dest = (
        mer2.groupby("destino", dropna=False, as_index=False)
            .agg(
                factor_desp_med=("factor_desp", "median"),
                factor_ajuste_med=("factor_ajuste", "median"),
            )
    )
    split = split.merge(med_dest, on="destino", how="left")

    fd_global = _safe_float(np.nanmedian(mer2["factor_desp"].values), default=0.70)
    fa_global = _safe_float(np.nanmedian(mer2["factor_ajuste"].values), default=1.00)

    split["factor_desp_seed"] = (
        pd.to_numeric(split["factor_desp"], errors="coerce")
        .fillna(pd.to_numeric(split["factor_desp_med"], errors="coerce"))
        .fillna(fd_global)
        .clip(lower=0.05, upper=1.00)
    )

    # baseline por medianas (destino -> global). NO usamos 1.0 salvo que sea imposible calcular mediana.
    split["factor_ajuste_seed"] = (
        pd.to_numeric(split["factor_ajuste"], errors="coerce")
        .fillna(pd.to_numeric(split["factor_ajuste_med"], errors="coerce"))
        .fillna(fa_global)
        .fillna(1.0)  # debería ser ~0% si mer2 tiene data; queda como último guardarraíl
        .clip(lower=0.50, upper=2.00)
    )

    # -------------------------
    # 8) Cajas poscosecha seed (macro)
    # -------------------------
    split["cajas_post_seed"] = (
        split["cajas_split_grado_dia"].astype(float)
        * split["factor_hidr_seed"].astype(float)
        * split["factor_desp_seed"].astype(float)
        * split["factor_ajuste_seed"].astype(float)  # <-- CAMBIO: MULTIPLICA
    )

    split["created_at"] = created_at

    # -------------------------
    # 9) Output 1: grado-dia-bloque-destino
    # -------------------------
    out_gd_bd = split[
        [
            "fecha",
            "fecha_post_pred",
            "bloque_base",
            "variedad_canon",
            "grado",
            "Semana_Ventas",
            "destino",
            "cajas_ml1_grado_dia",
            "cajas_split_grado_dia",
            "dh_dias",
            "factor_hidr_seed",
            "factor_desp_seed",
            "factor_ajuste_seed",
            "cajas_post_seed",
            "created_at",
        ]
    ].copy()

    out_gd_bd = out_gd_bd.sort_values(
        ["fecha", "bloque_base", "variedad_canon", "grado", "destino"]
    ).reset_index(drop=True)

    write_parquet(out_gd_bd, OUT_GD_BD)
    print(f"OK -> {OUT_GD_BD} | rows={len(out_gd_bd):,}")

    # -------------------------
    # 10) Output 2: día-destino (agregado)
    # -------------------------
    out_dd = (
        out_gd_bd.groupby(["fecha", "fecha_post_pred", "destino"], dropna=False, as_index=False)
                .agg(
                    cajas_ml1_dia=("cajas_ml1_grado_dia", "sum"),
                    cajas_split_dia=("cajas_split_grado_dia", "sum"),
                    cajas_post_seed=("cajas_post_seed", "sum"),
                )
    )
    out_dd["created_at"] = created_at
    out_dd = out_dd.sort_values(["fecha", "destino"]).reset_index(drop=True)

    write_parquet(out_dd, OUT_DD)
    print(f"OK -> {OUT_DD} | rows={len(out_dd):,}")

    # -------------------------
    # 11) Output 3: día-total (agregado)
    # -------------------------
    out_dt = (
        out_dd.groupby(["fecha", "fecha_post_pred"], dropna=False, as_index=False)
              .agg(
                  cajas_ml1_dia=("cajas_ml1_dia", "sum"),
                  cajas_split_dia=("cajas_split_dia", "sum"),
                  cajas_post_seed=("cajas_post_seed", "sum"),
              )
    )
    out_dt["created_at"] = created_at
    out_dt = out_dt.sort_values(["fecha"]).reset_index(drop=True)

    write_parquet(out_dt, OUT_DT)
    print(f"OK -> {OUT_DT} | rows={len(out_dt):,}")


if __name__ == "__main__":
    main()
