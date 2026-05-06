import pandas as pd
from pathlib import Path

def fix_manifest(file_path):
    print(f"Fixing {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # 1. Fix Windows paths
    if 'filepath' in df.columns:
        df['filepath'] = df['filepath'].str.replace('\\', '/', regex=False)
    
    # 2. Fill city and region
    if 'country' in df.columns:
        if 'city' not in df.columns:
            df['city'] = df['country']
        else:
            df['city'] = df['city'].fillna(df['country'])
            # Also handle cases where city is empty string or "None"
            df.loc[df['city'].isna() | (df['city'] == ''), 'city'] = df['country']
            
        if 'region' not in df.columns:
            df['region'] = df['country']
        else:
            df['region'] = df['region'].fillna(df['country'])
            df.loc[df['region'].isna() | (df['region'] == ''), 'region'] = df['country']

    df.to_csv(file_path, index=False)
    
    # Verification stats
    print(f"Sample paths from {file_path.name}:")
    print(df['filepath'].head(3).tolist())
    print(f"City value counts (top 3):")
    print(df['city'].value_counts().head(3).to_string())
    print("-" * 20)

def main():
    base_dir = Path(__file__).resolve().parent.parent
    manifests_dir = base_dir / "dataset" / "manifests"
    
    for filename in ["train.csv", "val.csv", "test.csv"]:
        file_path = manifests_dir / filename
        if file_path.exists():
            fix_manifest(file_path)
        else:
            print(f"Skipping {filename} (not found)")

if __name__ == "__main__":
    main()