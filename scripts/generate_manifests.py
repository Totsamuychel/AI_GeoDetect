import os
import sys
from pathlib import Path
import pandas as pd

# Add code directory to sys.path to import dataset module
sys.path.append(str(Path(__file__).resolve().parent.parent / "code"))
from dataset import _split_by_h3, _split_by_kmeans

def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_manifest_path = base_dir / "dataset" / "raw" / "osv5m" / "manifest.csv"
    train_out = base_dir / "dataset" / "manifests" / "train.csv"
    val_out = base_dir / "dataset" / "manifests" / "val.csv"
    
    print(f"Loading {raw_manifest_path}...")
    df = pd.read_csv(raw_manifest_path, low_memory=False)
    initial_len = len(df)
    
    required_cols = ['image_id', 'filepath', 'lat', 'lon', 'country', 'region', 'city', 'source', 'capture_date', 'quality_score']
    
    # Fill missing optional columns
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            
    # Filter out rows where lat, lon, or filepath are null
    df = df.dropna(subset=["lat", "lon", "filepath"])
    
    print("Generating H3 split (with fallback to kmeans)...")
    train_idx, val_idx, test_idx = _split_by_h3(df, train_frac=0.7, val_frac=0.15, h3_resolution=4, seed=42)
    
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]
    
    # Ensure correct column order
    train_df = train_df[required_cols]
    val_df = val_df[required_cols]
    test_df = test_df[required_cols]
    
    # Save the manifests (we do not overwrite test.csv)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print(f"Saved {train_out}")
    print(f"Saved {val_out}")
    
    # Statistics
    print("\n--- Statistics ---")
    print(f"Total rows in source manifest: {initial_len}")
    print(f"Rows in train: {len(train_df)}")
    print(f"Rows in val: {len(val_df)}")
    print(f"Rows in test (split size): {len(test_df)}")
    
    print(f"Unique countries in train: {train_df['country'].nunique(dropna=True)}")
    print(f"Unique countries in val: {val_df['country'].nunique(dropna=True)}")
    print(f"Unique cities in train: {train_df['city'].nunique(dropna=True)}")
    print(f"Unique cities in val: {val_df['city'].nunique(dropna=True)}")
    
    print("\nTop 10 cities in train:")
    print(train_df["city"].value_counts().head(10).to_string())
    
    print("\nTop 10 cities in val:")
    print(val_df["city"].value_counts().head(10).to_string())

if __name__ == "__main__":
    main()