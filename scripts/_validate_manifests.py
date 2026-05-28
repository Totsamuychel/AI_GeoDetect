import pandas as pd, sys
from pathlib import Path

root = Path("dataset")
ok = True
for split in ["train", "val", "test"]:
    df = pd.read_csv(root / "manifests" / f"{split}.csv")
    nan_lat  = int(df["lat"].isna().sum())
    nan_lon  = int(df["lon"].isna().sum())
    nan_city = int(df["city"].isna().sum())
    cities   = sorted(df["city"].dropna().unique())
    print(f"{split:5s}: {len(df):5d} rows | NaN lat={nan_lat} lon={nan_lon} city={nan_city} | cities={cities}")
    if nan_lat or nan_lon or nan_city:
        ok = False

# Spot-check 50 random file paths from all splits
import random, os
random.seed(42)
all_rows = []
for split in ["train","val","test"]:
    df = pd.read_csv(root / "manifests" / f"{split}.csv")
    all_rows.extend(df["filepath"].tolist())

sample = random.sample(all_rows, 50)
missing = [p for p in sample if not (root / p).exists()]
if missing:
    print(f"MISSING FILES ({len(missing)}/50):")
    for p in missing: print(f"  {p}")
    ok = False
else:
    print(f"File check: 50/50 random paths OK")

sys.exit(0 if ok else 1)
