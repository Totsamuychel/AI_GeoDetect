"""
07_build_streetview_splits.py — Build leak-free train/val/test splits for the
Google Street View dataset (Kyiv + Warsaw + Prague + Budapest).

Photos live in data/images/<city>/<lat>_<lon>_h<heading>.jpg (4 headings per
panorama location). Rows are built directly from the files on disk (robust to
manifest/disk mismatches); lat/lon/heading are parsed from the filename.

Split = geo-block, stratified by city, grouped by H3 cell (res 9, ~0.3 km) so
that the 4 headings of one location — and any spatially adjacent locations —
never straddle two splits. This is the same leak-free scheme as the Mapillary
builder (scripts/06_build_mapillary_splits.py).

Output: dataset/manifests_sv/{train,val,test}.csv with columns compatible with
code/dataset.py (image_id, filepath, lat, lon, country, region, city, source,
capture_date, quality_score). filepath is relative to project root, so configs
use image_root="." .

Usage:
    python scripts/07_build_streetview_splits.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

IMAGES_ROOT = Path("data/images")
OUT_DIR = Path("dataset/manifests_sv")

# city -> ISO country code (one city per country, so city == class here)
CITY_COUNTRY = {
    "kyiv":     "UA",
    "warsaw":   "PL",
    "prague":   "CZ",
    "budapest": "HU",
}

# filename like 50.248573_30.375547_h0.jpg  (lat _ lon _ h<heading>)
FNAME_RE = re.compile(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_h(\d+)\.jpg$", re.IGNORECASE)

H3_RESOLUTION = 9
SEED = 42

REQUIRED_COLS = [
    "image_id", "filepath", "lat", "lon",
    "country", "region", "city", "source",
    "capture_date", "quality_score",
]


def build_rows() -> pd.DataFrame:
    rows = []
    for city, country in CITY_COUNTRY.items():
        city_dir = IMAGES_ROOT / city
        if not city_dir.is_dir():
            print(f"  [SKIP] {city}: directory not found ({city_dir})")
            continue
        n = 0
        for f in sorted(city_dir.glob("*.jpg")):
            m = FNAME_RE.match(f.name)
            if not m:
                continue
            lat, lon, heading = float(m.group(1)), float(m.group(2)), int(m.group(3))
            rows.append({
                "image_id": f"{city}_{m.group(1)}_{m.group(2)}_h{heading}",
                "filepath": f"data/images/{city}/{f.name}",
                "lat": lat,
                "lon": lon,
                "country": country,
                "region": "",
                "city": city,
                "source": "streetview",
                "capture_date": "",
                "quality_score": 1.0,
            })
            n += 1
        print(f"  {city}: {n:,} images ({country})")
    df = pd.DataFrame(rows, columns=REQUIRED_COLS)
    return df


def _make_geo_grouper():
    """Return fn (lat, lon) -> group key. Prefers H3, falls back to ~1km grid."""
    def _grid(lat, lon):
        return f"{round(float(lat), 2)}_{round(float(lon), 2)}"

    try:
        import h3
    except ImportError:
        print("  [INFO] h3 not installed — using ~1km lat/lon grid for grouping")
        return _grid

    if hasattr(h3, "latlng_to_cell"):
        cell = h3.latlng_to_cell
    elif hasattr(h3, "geo_to_h3"):
        cell = h3.geo_to_h3
    else:
        print("  [INFO] h3 API unexpected — using ~1km lat/lon grid")
        return _grid

    print(f"  [INFO] grouping by H3 cells (res={H3_RESOLUTION})")
    return lambda lat, lon: str(cell(float(lat), float(lon), H3_RESOLUTION))


def split_dataset(df, train_frac=0.70, val_frac=0.15, seed=SEED):
    """Geo-block split, stratified by city. Whole groups stay in one split."""
    grouper = _make_geo_grouper()
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for city, group in df.groupby("city"):
        group = group.copy()
        group["_grp"] = [grouper(la, lo) for la, lo in zip(group["lat"], group["lon"])]
        sizes = group.groupby("_grp").size()
        keys = list(sizes.index)
        order = rng.permutation(len(keys))

        n_total = len(group)
        n_train_target = int(round(n_total * train_frac))
        n_val_target = int(round(n_total * val_frac))

        train_keys, val_keys, test_keys = set(), set(), set()
        n_tr = n_va = 0
        for i in order:
            k = keys[i]
            sz = int(sizes[k])
            if n_tr < n_train_target:
                train_keys.add(k); n_tr += sz
            elif n_va < n_val_target:
                val_keys.add(k); n_va += sz
            else:
                test_keys.add(k)

        def _ensure(target_set, donor_set):
            if target_set or not donor_set:
                return
            smallest = min(donor_set, key=lambda kk: int(sizes[kk]))
            donor_set.discard(smallest)
            target_set.add(smallest)

        _ensure(val_keys, train_keys)
        _ensure(test_keys, train_keys)

        gk = group["_grp"]
        train_parts.append(group[gk.isin(train_keys)].drop(columns="_grp"))
        val_parts.append(group[gk.isin(val_keys)].drop(columns="_grp"))
        test_parts.append(group[gk.isin(test_keys)].drop(columns="_grp"))
        print(f"  {city}: groups={len(keys)} | "
              f"train={len(train_parts[-1])} val={len(val_parts[-1])} test={len(test_parts[-1])}")

    def _concat(parts):
        return (pd.concat(parts, ignore_index=True)
                .sample(frac=1, random_state=seed).reset_index(drop=True))

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def _self_check(train, val, test):
    s_tr, s_va, s_te = set(train.image_id), set(val.image_id), set(test.image_id)
    assert not (s_tr & s_va), f"train∩val leak: {len(s_tr & s_va)}"
    assert not (s_tr & s_te), f"train∩test leak: {len(s_tr & s_te)}"
    assert not (s_va & s_te), f"val∩test leak: {len(s_va & s_te)}"
    cities = set(train.city)
    for name, part in (("val", val), ("test", test)):
        missing = cities - set(part.city)
        assert not missing, f"{name} missing cities: {missing}"
    print("  [OK] no image_id leakage; all cities present in every split")


def main():
    print("=" * 60)
    print("Building leak-free Street View dataset splits")
    print("=" * 60)

    print("\n>>> Scanning images on disk...")
    merged = build_rows()
    if merged.empty:
        print("[ERROR] No images found under data/images/. Aborting.")
        return
    print(f"\nTotal: {len(merged):,} images across {merged['city'].nunique()} cities")

    print("\n>>> Creating geo-block stratified splits (70/15/15)...")
    train, val, test = split_dataset(merged)

    print("\n>>> Self-check...")
    _self_check(train, val, test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8")
    val.to_csv(OUT_DIR / "val.csv", index=False, encoding="utf-8")
    test.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8")

    print("\n>>> Results:")
    print(f"  train: {len(train):,}  →  {OUT_DIR / 'train.csv'}")
    print(f"  val:   {len(val):,}  →  {OUT_DIR / 'val.csv'}")
    print(f"  test:  {len(test):,}  →  {OUT_DIR / 'test.csv'}")
    print("\n>>> City distribution (train / val / test):")
    for c in sorted(set(train.city)):
        print(f"  {c}: {int((train.city == c).sum())} / "
              f"{int((val.city == c).sum())} / {int((test.city == c).sum())}")

    print("\n[OK] Done. Next: train with configs/*_v2.yaml")


if __name__ == "__main__":
    main()
