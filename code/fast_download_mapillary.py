
"""
fast_download_mapillary.py — Быстрое скачивание уличных фото из Mapillary.

Разбивает большой bbox на тайлы 0.01°, получает метаданные через Graph API,
качает фото параллельно (16 воркеров).  Поддерживает resume.

Запуск:
    # Целевые города проекта (Київ, Варшава, Прага, Будапешт)
    python code/fast_download_mapillary.py --preset target-cities --max-per-city 5000

    # Один город по bbox
    python code/fast_download_mapillary.py \
        --bbox 50.35 30.25 50.55 30.75 \
        --name kyiv \
        --max-images 2000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Bbox-ы целевых городов проекта (Київ, Варшава, Прага, Будапешт)
# Format: (min_lat, min_lon, max_lat, max_lon)
# ──────────────────────────────────────────────────────────────────────────────

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "kyiv":        (50.35, 30.25, 50.55, 30.75),
    "warsaw":      (52.10, 20.85, 52.35, 21.15),
    "prague":      (49.95, 14.25, 50.15, 14.65),
    "budapest":    (47.43, 18.95, 47.58, 19.15),
}

PRESETS = {
    "target-cities": ["kyiv", "warsaw", "prague", "budapest"],
}

MAPILLARY_GRAPH_URL = "https://graph.mapillary.com"

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Ищет ключ в env под любым из распространённых имён."""
    for var in ("MAPILLARY_API_KEY", "MAPILLARY_ACCESS_TOKEN", "MLY_TOKEN"):
        key = os.environ.get(var, "")
        if key:
            return key
    raise RuntimeError(
        "Mapillary API ключ не найден! Укажите MAPILLARY_API_KEY в .env"
    )


def split_bbox(
    bbox: tuple[float, float, float, float],
    step: float = 0.01,
) -> list[tuple[float, float, float, float]]:
    """Разбивает большой bbox на тайлы ≤ step° (Mapillary лимит: 0.01°)."""
    min_lat, min_lon, max_lat, max_lon = bbox
    tiles = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            tiles.append((
                round(lat, 6),
                round(lon, 6),
                round(min(lat + step, max_lat), 6),
                round(min(lon + step, max_lon), 6),
            ))
            lon += step
        lat += step
    return tiles


def fetch_image_metas(
    bbox: tuple[float, float, float, float],
    api_key: str,
    max_images: int = 2000,
    image_size: str = "thumb_2048_url",
) -> list[dict]:
    """
    Получает метаданные фото из Mapillary Graph API для bbox.
    Автоматически разбивает на тайлы при большом bbox.
    """
    tiles = split_bbox(bbox)
    log.info(f"Bbox разбит на {len(tiles)} тайлов")

    fields = "id,geometry,captured_at,thumb_256_url,thumb_1024_url,thumb_2048_url"
    all_metas: list[dict] = []

    for tile_idx, tile in enumerate(tiles):
        if len(all_metas) >= max_images:
            break

        min_lat, min_lon, max_lat, max_lon = tile
        url = f"{MAPILLARY_GRAPH_URL}/images"
        params = {
            "access_token": api_key,
            "fields": fields,
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "limit": 200,
        }

        page = 0
        while url and len(all_metas) < max_images:
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    log.warning("Rate limit! Ждём 10 сек...")
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                data = resp.json()

                batch = data.get("data", [])
                all_metas.extend(batch)

                # Pagination
                paging = data.get("paging", {})
                url = paging.get("next")
                params = {}  # next URL уже содержит параметры

                page += 1
                time.sleep(0.15)  # Rate limiting

            except requests.RequestException as e:
                log.warning(f"Ошибка запроса тайла {tile_idx}: {e}")
                break

        if (tile_idx + 1) % 50 == 0:
            log.info(f"  Тайл {tile_idx+1}/{len(tiles)}, мета: {len(all_metas)}")

    all_metas = all_metas[:max_images]
    log.info(f"Получено {len(all_metas)} метаданных")
    if all_metas:
        log.info(f"Sample meta: {all_metas[0]}")
    return all_metas


def download_single_image(
    meta: dict,
    images_dir: Path,
    preferred_size: str = "thumb_2048_url",
) -> Optional[dict]:
    """Скачивает одно фото. Возвращает строку для маніфесту или None."""
    img_id = str(meta.get("id", ""))
    geometry = meta.get("geometry", {})
    coords = geometry.get("coordinates", [0, 0])
    lon, lat = float(coords[0]), float(coords[1])

    out_path = images_dir / f"{img_id}.jpg"

    # Resume: пропускаем уже скачанные
    if out_path.exists() and out_path.stat().st_size > 1000:
        return _make_row(img_id, out_path, lat, lon, meta)

    # Выбираем лучший доступный URL
    url = (
        meta.get(preferred_size)
        or meta.get("thumb_1024_url")
        or meta.get("thumb_256_url")
    )
    if not url:
        return None

    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out_path), "wb") as f:
            for chunk in resp.iter_content(chunk_size=16384):
                f.write(chunk)

        if out_path.stat().st_size < 500:
            out_path.unlink(missing_ok=True)
            return None

        return _make_row(img_id, out_path, lat, lon, meta)

    except Exception as e:
        log.error(f"Ошибка скачивания {img_id}: {e}")
        return None


def _make_row(img_id, out_path, lat, lon, meta) -> dict:
    captured_at = meta.get("captured_at", "")
    capture_date = ""
    if captured_at:
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(int(captured_at) / 1000)
            capture_date = dt.strftime("%Y-%m-%d")
        except Exception:
            capture_date = str(captured_at)

    return {
        "image_id": img_id,
        "filepath": str(out_path),
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "country": "UA",
        "source": "mapillary",
        "capture_date": capture_date,
    }


def download_city(
    city_name: str,
    bbox: tuple[float, float, float, float],
    api_key: str,
    output_root: Path,
    max_images: int = 1000,
    num_workers: int = 16,
) -> int:
    """Скачивает фото для одного города. Возвращает кол-во скачанных."""
    city_dir = output_root / city_name
    images_dir = city_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = city_dir / "manifest.csv"

    log.info(f"\n{'='*60}")
    log.info(f"🏙️  {city_name.upper()} — bbox: {bbox}")
    log.info(f"{'='*60}")

    # 1. Получаем метаданные
    metas = fetch_image_metas(bbox, api_key, max_images=max_images)
    if not metas:
        log.warning(f"Нет фото для {city_name}")
        return 0

    # 2. Параллельное скачивание
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(download_single_image, m, images_dir): m
            for m in metas
        }
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"📥 {city_name}",
            unit="img",
        )
        for future in pbar:
            row = future.result()
            if row:
                rows.append(row)
            pbar.set_postfix(ok=len(rows))

    # 3. Сохраняем маніфест
    if rows:
        with open(str(manifest_path), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    log.info(f"✅ {city_name}: {len(rows)} фото скачано → {city_dir}")
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Быстрое скачивание уличных фото из Mapillary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--preset", choices=list(PRESETS.keys()),
        help="Preset набор городов (target-cities = Київ, Варшава, Прага, Будапешт)",
    )
    p.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        help="Bbox для одного города",
    )
    p.add_argument("--name", type=str, default="custom",
                   help="Имя города (при --bbox)")
    p.add_argument("--max-per-city", type=int, default=1000,
                   help="Макс. фото на город")
    p.add_argument("--max-images", type=int, default=None,
                   help="Макс. фото (при --bbox)")
    p.add_argument("--output", type=str, default="dataset/raw/mapillary",
                   help="Корневая директория для сохранения")
    p.add_argument("--workers", type=int, default=16,
                   help="Количество параллельных потоков")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = get_api_key()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    total = 0

    if args.preset:
        cities = PRESETS[args.preset]
        log.info(f"Preset '{args.preset}': {len(cities)} городов")

        for city in cities:
            bbox = CITY_BBOXES[city]
            count = download_city(
                city_name=city,
                bbox=bbox,
                api_key=api_key,
                output_root=output_root,
                max_images=args.max_per_city,
                num_workers=args.workers,
            )
            total += count

    elif args.bbox:
        bbox = tuple(args.bbox)
        max_img = args.max_images or args.max_per_city
        total = download_city(
            city_name=args.name,
            bbox=bbox,
            api_key=api_key,
            output_root=output_root,
            max_images=max_img,
            num_workers=args.workers,
        )

    else:
        log.error("Укажи --preset или --bbox! Пример:")
        log.error("  python code/fast_download_mapillary.py --preset target-cities")
        sys.exit(1)

    log.info(f"\n{'='*60}")
    log.info(f"🎉 ИТОГО: {total} фото скачано в {output_root}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
