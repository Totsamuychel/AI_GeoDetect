You are working on the AI_GeoDetect project — a street-level image geolocation model.

## Task
Generate `train.csv` and `val.csv` manifests in `dataset/manifests/` by splitting the existing `dataset/raw/osv5m/manifest.csv` using H3-based geographic stratification (70% train / 15% val / 15% test, but test.csv already exists — only generate train and val).

## Context
- Existing file: `dataset/raw/osv5m/manifest.csv`
- Already exists: `dataset/manifests/test.csv`
- Must create: `dataset/manifests/train.csv` and `dataset/manifests/val.csv`
- The split logic already exists in `code/dataset.py` — use `_split_by_h3()` or `_split_by_kmeans()` as fallback
- Required CSV columns: `image_id`, `filepath`, `lat`, `lon`, `country`, `region`, `city`, `source`, `capture_date`, `quality_score`

## Requirements
1. Load `dataset/raw/osv5m/manifest.csv`
2. Apply H3 geographic split at resolution 4 (use `h3` library if available, fallback to k-means with n_clusters=200)
3. Use seed=42 for reproducibility
4. Split ratio: 70% train / 15% val / 15% test — BUT since test.csv already exists, regenerate the split consistently and save only train + val
5. Filter out rows where `lat`, `lon`, or `filepath` are null
6. Save results:
   - `dataset/manifests/train.csv`
   - `dataset/manifests/val.csv`
7. After saving, print statistics:
   - Total rows in source manifest
   - Rows in train / val / test
   - Number of unique countries and cities per split
   - Class distribution (top 10 cities by count)

## Constraints
- Use pandas, numpy, pathlib — no heavy dependencies beyond what's in requirements.txt
- If `h3` is not installed, automatically fall back to k-means
- Handle missing optional columns gracefully (fill with None)
- Do NOT overwrite `dataset/manifests/test.csv`
- Script must be runnable standalone: `python scripts/generate_manifests.py`

## File to create
Create a standalone script `scripts/generate_manifests.py` that performs all the above steps.
After running it, verify that both files exist and print a final confirmation with row counts.