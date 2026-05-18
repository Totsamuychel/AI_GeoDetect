"""
06_build_mapillary_splits.py — Merge Mapillary city manifests and create
leak-free train/val/test splits.

Reads per-city manifests from dataset/raw/mapillary/<city>/manifest.csv, fixes
metadata (country, city, filepath), then splits 70/15/15 **stratified by city**
and **grouped by geo-block** so that spatially adjacent frames (consecutive
Mapillary sequence shots are near-duplicates) never land in two different
splits. Without this, random splits leak almost-identical frames train<->test
and inflate accuracy — invalid for a thesis.

Grouping key per photo:
  - H3 cell at resolution 9 (~0.1 km^2, ~0.3 km across) if the `h3` package is
    available (works with both h3>=4 `latlng_to_cell` and h3<4 `geo_to_h3`).
    At res 9 consecutive near-duplicate sequence frames (meters apart) share a
    cell while the city still has enough cells for a smooth 70/15/15 split;
  - otherwise a dependency-free ~1 km lat/lon grid (round to 2 decimals).

Whole groups are assigned to a single split; every city is guaranteed to be
present in train/val/test.

Usage:
    python scripts/06_build_mapillary_splits.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CITIES = {
    "warsaw":   {"country": "PL", "city": "warsaw"},
    "prague":   {"country": "CZ", "city": "prague"},
    "budapest": {"country": "HU", "city": "budapest"},
}

REQUIRED_COLS = [
    "image_id", "filepath", "lat", "lon",
    "country", "region", "city", "source",
    "capture_date", "quality_score",
]

H3_RESOLUTION = 9
SEED = 42


def _make_geo_grouper():
    """Return a vectorizable fn (lat, lon) -> group key. Prefers H3."""
    def _grid(lat, lon):
        return f"{round(float(lat), 2)}_{round(float(lon), 2)}"

    try:
        import h3
    except ImportError:
        print("  [INFO] h3 not installed — using ~1km lat/lon grid for grouping")
        return _grid

    if hasattr(h3, "latlng_to_cell"):          # h3 >= 4.0
        cell = h3.latlng_to_cell
    elif hasattr(h3, "geo_to_h3"):             # h3 < 4.0
        cell = h3.geo_to_h3
    else:
        print("  [INFO] h3 API unexpected — using ~1km lat/lon grid")
        return _grid

    print(f"  [INFO] grouping by H3 cells (res={H3_RESOLUTION})")
    return lambda lat, lon: str(cell(float(lat), float(lon), H3_RESOLUTION))


def load_city_manifest(city_name: str, meta: dict) -> pd.DataFrame:
    path = Path(f"dataset/raw/mapillary/{city_name}/manifest.csv")
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    print(f"  {city_name}: {len(df):,} rows loaded")

    # Per-city manifests have a wrong/missing country and no region/quality.
    df["country"] = meta["country"]
    df["city"]    = meta["city"]
    df["region"]  = ""
    df["quality_score"] = 1.0
    if "source" not in df.columns:
        df["source"] = "mapillary"

    # Normalize filepath: forward slashes, relative to dataset/ dir.
    df["filepath"] = (
        df["filepath"].astype(str)
        .str.replace("\\", "/", regex=False)
        .str.replace("dataset/", "", regex=False)
        .str.lstrip("/")
    )

    sample = Path("dataset") / df["filepath"].iloc[0]
    if not sample.exists():
        print(f"  [WARN] Sample file not found: {sample}")

    return df[REQUIRED_COLS]


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Geo-block split, stratified by city. Whole groups stay in one split."""
    grouper = _make_geo_grouper()
    rng = np.random.default_rng(seed)

    train_parts, val_parts, test_parts = [], [], []

    for city, group in df.groupby("city"):
        group = group.copy()
        group["_grp"] = [
            grouper(la, lo) for la, lo in zip(group["lat"], group["lon"])
        ]
        # Map non-hashable-friendly keys to ids, count images per group.
        sizes = group.groupby("_grp").size()        # str key -> count
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

        # Guarantee every split is non-empty for this city by moving the
        # smallest group(s) over if greedy assignment starved val/test.
        def _ensure(target_set, donor_set):
            if target_set:
                return
            if not donor_set:
                return
            smallest = min(donor_set, key=lambda kk: int(sizes[kk]))
            donor_set.discard(smallest)
            target_set.add(smallest)

        _ensure(val_keys, train_keys)
        _ensure(test_keys, train_keys)

        gk = group["_grp"]
        tr = group[gk.isin(train_keys)].drop(columns="_grp")
        va = group[gk.isin(val_keys)].drop(columns="_grp")
        te = group[gk.isin(test_keys)].drop(columns="_grp")

        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)
        print(
            f"  {city}: groups={len(keys)} | "
            f"train={len(tr)} val={len(va)} test={len(te)}"
        )

    def _concat(parts):
        return (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def _self_check(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    s_tr, s_va, s_te = (set(train.image_id), set(val.image_id), set(test.image_id))
    assert not (s_tr & s_va), f"train∩val leak: {len(s_tr & s_va)}"
    assert not (s_tr & s_te), f"train∩test leak: {len(s_tr & s_te)}"
    assert not (s_va & s_te), f"val∩test leak: {len(s_va & s_te)}"
    cities = set(train.city)
    for name, part in (("val", val), ("test", test)):
        missing = cities - set(part.city)
        assert not missing, f"{name} missing cities: {missing}"
    print("  [OK] no image_id leakage; all cities present in every split")


def main() -> None:
    print("=" * 60)
    print("Building leak-free Mapillary dataset splits")
    print("=" * 60)

    print("\n>>> Loading city manifests...")
    dfs = []
    for city_name, meta in CITIES.items():
        try:
            dfs.append(load_city_manifest(city_name, meta))
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if not dfs:
        print("[ERROR] No manifests found.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(merged):,} images across {merged['city'].nunique()} cities")

    merged_path = Path("dataset/raw/mapillary/manifest.csv")
    merged.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"[OK] Merged manifest: {merged_path}")

    print("\n>>> Creating geo-block stratified splits (70/15/15)...")
    train, val, test = split_dataset(merged)

    print("\n>>> Self-check...")
    _self_check(train, val, test)

    out_dir = Path("dataset/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    val.to_csv(out_dir / "val.csv",   index=False, encoding="utf-8")
    test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    print(f"\n>>> Results:")
    print(f"  train: {len(train):,}  →  dataset/manifests/train.csv")
    print(f"  val:   {len(val):,}  →  dataset/manifests/val.csv")
    print(f"  test:  {len(test):,}  →  dataset/manifests/test.csv")
    print("\n>>> City distribution (train / val / test):")
    for c in sorted(set(train.city)):
        print(
            f"  {c}: {int((train.city == c).sum())} / "
            f"{int((val.city == c).sum())} / {int((test.city == c).sum())}"
        )

    print("\n[OK] Done. Next: python code/train.py --config configs/baseline.yaml")


if __name__ == "__main__":
    main()
