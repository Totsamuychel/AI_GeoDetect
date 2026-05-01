You are working on the AI_GeoDetect project. Fix the `code/download_data.py` file to resolve a critical performance issue with OSV-5M dataset downloading.

## Problem
The current `_download_osv5m_hf()` function uses `streaming=True` with a linear scan of the entire ~5M record dataset to find country-filtered images. This causes multi-hour waits with no disk output and eventual crashes.

## Required Changes

### 1. Fix `_download_osv5m_hf()` — replace streaming scan with filtered batch load

Replace the current `load_dataset(streaming=True)` + manual for-loop scan with:

```python
dataset = load_dataset(
    OSV5M_HF_REPO,
    split="train",
    streaming=False,
    trust_remote_code=True,
)

filtered = dataset.filter(
    lambda x: str(x.get("country", "")).upper() in countries
              and float(x.get("quality_score", 1.0)) >= quality_threshold
)

if max_images_per_country:
    from collections import defaultdict
    counts = defaultdict(int)
    candidates = []
    for sample in filtered:
        c = str(sample.get("country", "")).upper()
        if counts[c] < max_images_per_country:
            candidates.append(sample)
            counts[c] += 1
        if all(counts[c] >= max_images_per_country for c in countries):
            break
else:
    candidates = list(filtered)
```

Remove the old manual `for sample in tqdm(dataset, ...)` loop entirely.

### 2. Add checkpoint/resume support in `_download_single()`

Before downloading, check if file already exists and skip:
```python
if out_path.exists() and out_path.stat().st_size > 1000:
    # File already downloaded, just return manifest row
    return { ... existing row data ... }
```

### 3. Add progress logging with ETA

After filtering, log how many candidates were found before starting downloads:
```python
logger.info(f"Filtered {len(candidates)} candidates for countries {countries}. Starting download...")
```

### 4. Fix `_download_osv5m_parquet()` — add correct parquet file path

The current hardcoded filename `data/train-00000-of-00001.parquet` may not exist. Update to try multiple shards:
```python
parquet_files = [
    f"data/train-{str(i).zfill(5)}-of-00100.parquet"
    for i in range(100)
]
# Download and concatenate first N shards only (e.g. 5) for speed
```

## Testing
After changes, verify with:
```bash
python code/download_data.py osv5m --countries UA --output dataset/raw/osv5m_test --max-images 10 --quality 0.3 --workers 4
```
Expected: completes in under 5 minutes with 10 images saved.

## Constraints
- Do not change function signatures or CLI argument names
- Keep all existing logging calls
- Preserve the `--no-hf` flag behavior (parquet fallback)
- Do not remove the `create_manifest` or `download_mapillary` functions