import pandas as pd
from src.features.feature_engineering import compute_features
from src.features.feature_store import upsert_features

df = compute_features(pd.read_csv("data/raw/creditcard.csv"))
upsert_features(df)
df.to_parquet("data/processed/features.parquet", index=False)
