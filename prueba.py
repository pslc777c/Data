import pandas as pd
df=pd.read_parquet("data/gold/features_mix_pred_dia_destino.parquet")
print(df["int_largo_bajo_gi"].describe())
print(df["int_largo_bajo_gi_bin_used"].value_counts().head(10))