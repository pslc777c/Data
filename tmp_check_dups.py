import pandas as pd

p = r"data\preds\pred_oferta_grado.parquet"
df = pd.read_parquet(p)
df.columns = [str(c).strip() for c in df.columns]

keys = ["ciclo_id","fecha","bloque","bloque_padre","variedad_std","grado"]
keys = [k for k in keys if k in df.columns]

dup = df[df.duplicated(subset=keys, keep=False)].copy()
print("rows:", len(df))
print("dup_rows:", len(dup))
print("dup_groups:", dup.groupby(keys).size().shape[0])

print("\nTop dup group sizes:")
print(dup.groupby(keys).size().sort_values(ascending=False).head(10))

check_cols = [c for c in ["stage","tipo_sp","area","estado","variedad"] if c in df.columns]
print("\nMax nunique dentro de grupos duplicados:")
for c in check_cols:
    mx = dup.groupby(keys)[c].nunique(dropna=False).max()
    print(c, "->", int(mx))

if "tallos_pred_grado" in df.columns:
    mx = dup.groupby(keys)["tallos_pred_grado"].nunique(dropna=False).max()
    print("\ntallos_pred_grado nunique max dentro grupo:", int(mx))
