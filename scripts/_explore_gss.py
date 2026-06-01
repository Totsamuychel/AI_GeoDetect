"""Temporary: explore Global Streetscapes metadata to find city names."""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from huggingface_hub import hf_hub_download
import pandas as pd

HF_TOKEN = os.environ.get("HF_TOKEN")

print("Downloading gadm.parquet...")
path = hf_hub_download(
    repo_id="NUS-UAL/global-streetscapes",
    filename="data/parquet/gadm.parquet",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_parquet(path)
print(f"Total records: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample rows:")
print(df.head(3).to_string())

# Find city column
city_cols = [c for c in df.columns if "city" in c.lower() or "adm" in c.lower() or "name" in c.lower()]
print(f"\nPotential city columns: {city_cols}")

for col in city_cols:
    target_keywords = ["kyiv", "kiev", "warsaw", "warszawa", "prague", "praha"]
    for kw in target_keywords:
        matches = df[df[col].astype(str).str.lower().str.contains(kw, na=False)]
        if len(matches) > 0:
            names = matches[col].unique()[:5].tolist()
            print(f"  [{col}] {kw}: {len(matches)} records, names={names}")
