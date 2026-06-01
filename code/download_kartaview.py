"""
download_kartaview.py — Download street photos from KartaView (no API key needed).

KartaView API v2: https://api.kartaview.org/2.0/
Strategy:
  1. POST /2.0/sequence/ with bbox → list of sequences
  2. POST /2.0/photo/  with sequence_id → list of photos with URLs
  3. Download file_url_proc_960 (960px processed image)

Usage:
    python code/download_kartaview.py --name kyiv --max-images 2500
    python code/download_kartaview.py --name warsaw --bbox 52.10 20.85 52.35 21.15
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

API_BASE = "https://api.kartaview.org/2.0"

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "kyiv":    (50.35, 30.25, 50.55, 30.75),
    "warsaw":  (52.10, 20.85, 52.35, 21.15),
    "prague":  (49.95, 14.25, 50.15, 14.65),
    "budapest":(47.43, 18.95, 47.58, 19.15),
}


def fetch_sequences(bbox: tuple[float, float, float, float], max_seqs: int = 2000) -> list[dict]:
    """Fetch all sequence IDs inside the bbox."""
    lat_min, lon_min, lat_max, lon_max = bbox
    # KartaView: bbTopLeft = max_lat,min_lon; bbBottomRight = min_lat,max_lon
    bb_top_left    = f"{lat_max},{lon_min}"
    bb_bottom_right = f"{lat_min},{lon_max}"

    sequences = []
    page = 1

    log.info(f"Fetching sequences for bbox ({lat_min},{lon_min}) - ({lat_max},{lon_max})...")

    while len(sequences) < max_seqs:
        try:
            resp = requests.post(
                f"{API_BASE}/sequence/",
                data={
                    "bbTopLeft":     bb_top_left,
                    "bbBottomRight": bb_bottom_right,
                    "page":          page,
                    "ipp":           100,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("currentPageItems", [])
            if not items:
                break

            sequences.extend(items)
            total = data.get("totalFilteredItems", {})
            if isinstance(total, dict):
                total = int(total.get("value", 0))
            else:
                total = int(total or 0)

            log.info(f"  Page {page}: +{len(items)} sequences (total available: {total})")

            if len(sequences) >= total or len(items) < 100:
                break

            page += 1
            time.sleep(0.2)

        except Exception as e:
            log.warning(f"Error fetching sequences page {page}: {e}")
            break

    log.info(f"Total sequences found: {len(sequences)}")
    return sequences


def fetch_photos_for_sequence(seq_id: int | str, max_photos: int = 500) -> list[dict]:
    """Fetch photo metadata for a single sequence."""
    photos = []
    page = 1

    while len(photos) < max_photos:
        try:
            resp = requests.post(
                f"{API_BASE}/photo/",
                data={
                    "sequence_id": seq_id,
                    "page": page,
                    "ipp": 100,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("currentPageItems", [])
            if not items:
                break

            photos.extend(items)

            if len(items) < 100:
                break

            page += 1
            time.sleep(0.1)

        except Exception:
            break

    return photos


def collect_photos(
    bbox: tuple[float, float, float, float],
    max_images: int = 2500,
    num_workers: int = 8,
) -> list[dict]:
    """Collect photo metadata from all sequences in bbox."""
    sequences = fetch_sequences(bbox, max_seqs=500)

    if not sequences:
        log.error("No sequences found in this bbox.")
        return []

    all_photos: list[dict] = []

    log.info(f"Fetching photos from {len(sequences)} sequences (parallel)...")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(fetch_photos_for_sequence, seq.get("id"), 200): seq
            for seq in sequences
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sequences"):
            try:
                photos = future.result()
                all_photos.extend(photos)
                if len(all_photos) >= max_images * 2:
                    break
            except Exception:
                pass

    log.info(f"Total photos collected: {len(all_photos)}")

    # Deduplicate by id
    seen = set()
    unique = []
    for p in all_photos:
        pid = p.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)

    log.info(f"Unique photos: {len(unique)}")
    return unique[:max_images]


def download_photo(photo: dict, images_dir: Path) -> Optional[dict]:
    """Download a single photo. Returns manifest row or None."""
    photo_id = str(photo.get("id", ""))
    lat = photo.get("lat") or photo.get("latitude")
    lon = photo.get("lng") or photo.get("longitude")

    if not photo_id or lat is None or lon is None:
        return None

    out_path = images_dir / f"{photo_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return _make_row(photo_id, out_path, float(lat), float(lon), photo)

    # Try different URL fields in order of quality
    url = (
        photo.get("file_url_proc_1920")
        or photo.get("file_url_proc_960")
        or photo.get("file_url_large_thumb")
        or photo.get("file_url_thumb")
    )
    if not url:
        return None

    # Prepend domain if relative URL
    if url.startswith("/"):
        url = f"https://openstreetcam.org{url}"

    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=16384):
                f.write(chunk)

        if out_path.stat().st_size < 500:
            out_path.unlink(missing_ok=True)
            return None

        return _make_row(photo_id, out_path, float(lat), float(lon), photo)

    except Exception as e:
        log.debug(f"Error downloading {photo_id}: {e}")
        return None


def _make_row(photo_id: str, path: Path, lat: float, lon: float, meta: dict) -> dict:
    date_added = str(meta.get("date_added", ""))[:10]
    return {
        "image_id":     photo_id,
        "filepath":     f"raw/mapillary/kyiv/images/{path.name}",
        "lat":          round(lat, 6),
        "lon":          round(lon, 6),
        "country":      "UA",
        "region":       "",
        "city":         "kyiv",
        "source":       "kartaview",
        "capture_date": date_added,
        "quality_score": 1.0,
    }


def download_city(
    city_name: str,
    bbox: tuple[float, float, float, float],
    output_root: Path,
    max_images: int = 2500,
    num_workers: int = 16,
) -> int:
    city_dir = output_root / city_name
    images_dir = city_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"  {city_name.upper()} — KartaView download")
    log.info(f"{'='*60}")

    photos = collect_photos(bbox, max_images=max_images, num_workers=num_workers)
    if not photos:
        log.warning(f"No photos found for {city_name}")
        return 0

    log.info(f"Downloading {len(photos)} photos...")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(download_photo, p, images_dir): p for p in photos}
        pbar = tqdm(as_completed(futures), total=len(futures), desc=city_name, unit="img")
        for future in pbar:
            row = future.result()
            if row:
                rows.append(row)
            pbar.set_postfix(ok=len(rows))

    if rows:
        # Fix filepath to be city-agnostic
        for r in rows:
            img_name = Path(r["filepath"]).name
            r["filepath"] = f"raw/mapillary/{city_name}/images/{img_name}"
            r["city"] = city_name

        manifest_path = city_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Manifest saved: {manifest_path}")

    log.info(f"[OK] {city_name}: {len(rows)} photos -> {city_dir}")
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Download KartaView street photos")
    p.add_argument("--name", type=str, default="kyiv",
                   help="City name (kyiv/warsaw/prague or custom with --bbox)")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
                   help="Bounding box override")
    p.add_argument("--max-images", type=int, default=2500)
    p.add_argument("--output", type=str, default="dataset/raw/mapillary")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.name in CITY_BBOXES:
        bbox = CITY_BBOXES[args.name]
    else:
        p.error(f"Unknown city '{args.name}'. Use --bbox or one of: {list(CITY_BBOXES)}")

    output_root = Path(args.output)
    count = download_city(
        city_name=args.name,
        bbox=bbox,
        output_root=output_root,
        max_images=args.max_images,
        num_workers=args.workers,
    )

    log.info(f"\n{'='*60}")
    log.info(f"DONE: {count} photos downloaded to {output_root / args.name}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
