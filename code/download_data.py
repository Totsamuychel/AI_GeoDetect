"""
download_data.py — Завантаження даних із OSV-5M та Mapillary API.

Функції:
- download_osv5m_subset:  завантаження та фільтрація OSV-5M за кодами країн
- download_mapillary:     завантаження зображень Mapillary через ZenSVI
- create_manifest:        створення CSV-маніфесту з локальних зображень

Запуск:
    # Завантаження OSV-5M для України
    python download_data.py osv5m --countries UA --output data/ukraine

    # Завантаження Mapillary для Києва
    python download_data.py mapillary --bbox 50.2 30.2 50.6 30.8 \
                                       --api-key YOUR_KEY \
                                       --output data/kyiv

    # Створення маніфесту з директорії
    python download_data.py manifest --image-dir data/ukraine \
                                      --output data/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Завантажуємо змінні оточення (.env)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Константи
# ──────────────────────────────────────────────────────────────────────────────

OSV5M_BASE_URL = "https://huggingface.co/datasets/osv5m/osv5m"
OSV5M_HF_REPO  = "osv5m/osv5m"

MAPILLARY_API_URL  = "https://graph.mapillary.com"
MAPILLARY_TILE_URL = "https://tiles.mapillary.com"

# Коди країн ISO 3166-1 alpha-2
UKRAINE_CODE = "UA"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"}


# ──────────────────────────────────────────────────────────────────────────────
# 1. OSV-5M
# ──────────────────────────────────────────────────────────────────────────────

def download_osv5m_subset(
    countries: list[str],
    output_dir: Union[str, Path],
    max_images_per_country: Optional[int] = None,
    quality_threshold: float = 0.3,
    image_size: int = 512,
    use_hf_datasets: bool = True,
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Завантажує підмножину OSV-5M, відфільтровану за кодами країн.

    OSV-5M — відкритий датасет вуличних зображень із ~5M фото.
    Хостується на HuggingFace: https://huggingface.co/datasets/osv5m/osv5m

    Аргументи:
        countries:                 Список кодів країн ISO (наприклад, ['UA', 'PL']).
        output_dir:                Директорія для збереження зображень та маніфесту.
        max_images_per_country:    Максимальна кількість зображень на країну (None = всі).
        quality_threshold:         Мінімальний поріг якості (0.0–1.0).
        image_size:                Бажаний розмір зображення (пікселів).
        use_hf_datasets:           Якщо True, використовує HuggingFace datasets API.
        num_workers:               Кількість паралельних завантажень.

    Повертає:
        DataFrame із маніфестом завантажених зображень.

    Примітки:
        - OSV-5M вимагає підтвердження умов використання (CC-BY-4.0).
        - Потребує: pip install datasets huggingface-hub
        - При use_hf_datasets=False використовується пряме завантаження parquet.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    countries = [c.upper() for c in countries]
    logger.info(f"Завантаження OSV-5M для країн: {countries}")

    manifest_rows: list[dict] = []

    if use_hf_datasets:
        manifest_rows = _download_osv5m_hf(
            countries=countries,
            images_dir=images_dir,
            max_images_per_country=max_images_per_country,
            quality_threshold=quality_threshold,
            num_workers=num_workers,
        )
    else:
        manifest_rows = _download_osv5m_parquet(
            countries=countries,
            images_dir=images_dir,
            max_images_per_country=max_images_per_country,
            quality_threshold=quality_threshold,
        )

    df = pd.DataFrame(manifest_rows)
    if len(df) == 0:
        logger.warning("Не завантажено жодного зображення!")
        return df

    manifest_path = output_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    logger.info(f"Маніфест збережено: {manifest_path} ({len(df)} рядків)")

    return df


def _download_osv5m_hf(
    countries: list[str],
    images_dir: Path,
    max_images_per_country: Optional[int],
    quality_threshold: float,
    num_workers: int,
) -> list[dict]:
    """Завантаження через метадані та ZIP-шарди з видаленням для економії місця."""
    import os
    import pandas as pd
    from huggingface_hub import hf_hub_download
    import requests
    import zipfile
    import tempfile
    import shutil

    # 1. Isolate Cache Directory
    cache_dir = Path("data") / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    
    logger.info("Завантаження метаданих OSV-5M (train.csv)...")
    try:
        csv_path = hf_hub_download(
            repo_id=OSV5M_HF_REPO,
            filename="train.csv",
            repo_type="dataset",
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN")
        )
    except Exception as e:
        logger.error(f"Помилка завантаження метаданих: {e}")
        raise

    # 2. Local Filtering
    logger.info("Фільтрація метаданих (Pandas)...")
    df = pd.read_csv(csv_path, dtype={"id": str, "country": str})
    
    # Зберігаємо оригінальний індекс рядка для обчислення номеру ZIP-шарду (по 50000 фото в шарді)
    df['row_index'] = df.index
    
    # Фільтрація по країнах
    df = df[df["country"].str.upper().isin(countries)]

    # Обмеження кількості
    if max_images_per_country:
        df = df.groupby("country").head(max_images_per_country).reset_index(drop=True)

    target_ids = set(df["id"].astype(str).tolist())
    logger.info(f"Відфільтровано {len(target_ids)} зображень для країн: {countries}")

    if not target_ids:
        return []

    # 3. Idempotency (перевірка існуючих)
    needed_ids = set()
    needed_shards = set()
    manifest_rows = []
    
    for _, row in df.iterrows():
        img_id = str(row["id"])
        country = str(row["country"]).upper()
        country_dir = images_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)
        out_path = country_dir / f"{img_id}.jpg"
        
        lat = float(row.get("latitude", 0.0))
        lon = float(row.get("longitude", 0.0))
        
        manifest_row = {
            "image_id": img_id,
            "filepath": str(out_path.relative_to(images_dir.parent)),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "country": country,
            "region": str(row.get("region", "")),
            "city": str(row.get("city", "")),
            "source": "osv5m",
            "capture_date": str(row.get("captured_at", "")),
            "quality_score": 1.0,
        }
        manifest_rows.append(manifest_row)
        
        if out_path.exists() and out_path.stat().st_size > 1000:
            pass # Вже є
        else:
            needed_ids.add(img_id)
            # Розрахунок шарду: 50000 фотографій на 1 zip файл
            shard_idx = row['row_index'] // 50000
            needed_shards.add(shard_idx)

    if not needed_ids:
        logger.info("Усі потрібні зображення вже завантажено!")
        return manifest_rows

    logger.info(f"Потрібно завантажити {len(needed_ids)} зображень. Вони знаходяться у шардах: {sorted(list(needed_shards))}")

    # 4 & 5. Targeted Download & Disk Space Management
    base_url = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/images/train/{:02d}.zip"
    
    for shard_idx in sorted(list(needed_shards)):
        if not needed_ids:
            break
            
        url = base_url.format(shard_idx)
        logger.info(f"Завантаження цільового шарду {shard_idx:02d}.zip...")
        
        tmp_zip_path = None
        try:
            fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp_zip_path, "wb") as f:
                    for chunk in tqdm(r.iter_content(chunk_size=1024*1024), desc=f"Shard {shard_idx:02d}", leave=False):
                        f.write(chunk)
            
            extracted_in_this_shard = 0
            with zipfile.ZipFile(tmp_zip_path, "r") as zf:
                for file_info in zf.infolist():
                    filename = file_info.filename
                    if filename.endswith('/'): continue
                    
                    basename = os.path.basename(filename)
                    file_id, _ = os.path.splitext(basename)
                    
                    if file_id in needed_ids:
                        row_match = df[df["id"].astype(str) == file_id].iloc[0]
                        country = str(row_match["country"]).upper()
                        out_path = images_dir / country / basename
                        
                        with zf.open(file_info) as source, open(out_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                            
                        needed_ids.remove(file_id)
                        extracted_in_this_shard += 1
                        
            if extracted_in_this_shard > 0:
                logger.info(f"Витягнуто {extracted_in_this_shard} фото з шарду {shard_idx:02d}. Залишилось: {len(needed_ids)}")
            
        except Exception as e:
            logger.warning(f"Помилка обробки шарду {shard_idx}: {e}")
        finally:
            if tmp_zip_path and os.path.exists(tmp_zip_path):
                os.remove(tmp_zip_path)

    final_manifest = []
    for row in manifest_rows:
        expected_path = images_dir.parent / row["filepath"]
        if expected_path.exists() and expected_path.stat().st_size > 1000:
            final_manifest.append(row)

    return final_manifest


def _download_osv5m_parquet(
    countries: list[str],
    images_dir: Path,
    max_images_per_country: Optional[int],
    quality_threshold: float,
) -> list[dict]:
    """
    Parquet-файлів більше немає в репозиторії OSV-5M.
    Ця функція тепер викликає _download_osv5m_hf (metadata-first).
    """
    logger.warning("Parquet-файли більше не підтримуються OSV-5M. Використовуємо metadata-first підхід...")
    return _download_osv5m_hf(
        countries=countries,
        images_dir=images_dir,
        max_images_per_country=max_images_per_country,
        quality_threshold=quality_threshold,
        num_workers=1
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Mapillary
# ──────────────────────────────────────────────────────────────────────────────

def download_mapillary(
    bbox: tuple[float, float, float, float],
    api_key: str,
    output_dir: Union[str, Path],
    max_images: int = 1000,
    image_size: str = "thumb_2048_url",
    use_zensvi: bool = True,
    organization_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Завантажує вуличні зображення з Mapillary API.

    Аргументи:
        bbox:            Обмежувальна рамка (min_lat, min_lon, max_lat, max_lon).
        api_key:         Mapillary Access Token (з developer.mapillary.com).
        output_dir:      Директорія для збереження зображень.
        max_images:      Максимальна кількість зображень.
        image_size:      Розмір зображень ('thumb_256_url', 'thumb_1024_url',
                         'thumb_2048_url', 'thumb_original_url').
        use_zensvi:      Якщо True, використовує ZenSVI (простіший API).
        organization_id: Фільтр за організацією Mapillary.
        start_date:      Початкова дата (ISO формат YYYY-MM-DD).
        end_date:        Кінцева дата.
        num_workers:     Кількість паралельних завантажень.

    Повертає:
        DataFrame із маніфестом завантажених зображень.

    Примітки:
        - Потребує Mapillary Access Token: https://www.mapillary.com/dashboard/developers
        - ZenSVI: pip install zensvi (рекомендовано)
        - Пряме API: потребує обробки pagination та tile-based запитів
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    min_lat, min_lon, max_lat, max_lon = bbox
    logger.info(
        f"Завантаження Mapillary: bbox=({min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f}), "
        f"max={max_images}"
    )

    if use_zensvi:
        return _download_mapillary_zensvi(
            bbox=bbox,
            api_key=api_key,
            output_dir=output_dir,
            images_dir=images_dir,
            max_images=max_images,
            image_size=image_size,
        )
    else:
        return _download_mapillary_direct(
            bbox=bbox,
            api_key=api_key,
            images_dir=images_dir,
            output_dir=output_dir,
            max_images=max_images,
            image_size=image_size,
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            num_workers=num_workers,
        )


def _download_mapillary_zensvi(
    bbox: tuple,
    api_key: str,
    output_dir: Path,
    images_dir: Path,
    max_images: int,
    image_size: str,
) -> pd.DataFrame:
    """Завантаження через ZenSVI (обгортка навколо Mapillary API)."""
    try:
        from zensvi.download import MLYDownloader
    except ImportError:
        raise ImportError(
            "Встановіть ZenSVI: pip install zensvi\n"
            "Або використайте use_zensvi=False для прямого API."
        )

    min_lat, min_lon, max_lat, max_lon = bbox

    logger.info("Ініціалізація ZenSVI Downloader...")
    downloader = MLYDownloader(
        mly_api_key=api_key,
        log_path=str(output_dir / "zensvi.log"),
    )

    downloader.download_svi(
        dir_output=str(images_dir),
        lat=min_lat + (max_lat - min_lat) / 2,
        lon=min_lon + (max_lon - min_lon) / 2,
        input_shp_file=None,  # Bbox замість shapefile
        bbox=[min_lon, min_lat, max_lon, max_lat],
        image_type="all",
        start_date=None,
        end_date=None,
        option="mly1_public",
        resolution=17,
        limit=max_images,
    )

    # Читаємо маніфест, що генерує ZenSVI
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
    else:
        # Генеруємо маніфест з завантажених файлів
        df = create_manifest(str(images_dir), str(manifest_path))

    return df


def _download_mapillary_direct(
    bbox: tuple,
    api_key: str,
    images_dir: Path,
    output_dir: Path,
    max_images: int,
    image_size: str,
    organization_id: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    num_workers: int,
) -> pd.DataFrame:
    """Пряме завантаження через Mapillary Graph API."""
    min_lat, min_lon, max_lat, max_lon = bbox

    headers = {"Authorization": f"OAuth {api_key}"}

    # Параметри запиту зображень
    fields = [
        "id", "geometry", "captured_at",
        "thumb_256_url", "thumb_1024_url", "thumb_2048_url",
        "creator", "organization",
    ]

    params: dict = {
        "access_token": api_key,
        "fields":       ",".join(fields),
        "bbox":         f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "limit":        min(200, max_images),
    }

    if organization_id:
        params["organization_id"] = organization_id
    if start_date:
        params["start_captured_at"] = start_date
    if end_date:
        params["end_captured_at"] = end_date

    all_images: list[dict] = []
    next_url: Optional[str] = f"{MAPILLARY_API_URL}/images"

    logger.info("Отримання метаданих зображень із Mapillary API...")

    while next_url and len(all_images) < max_images:
        try:
            resp = requests.get(next_url, params=params if "images" in next_url else {}, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            images_batch = data.get("data", [])
            all_images.extend(images_batch)
            logger.info(f"  Отримано {len(all_images)}/{max_images} метаданих...")

            # Pagination
            paging = data.get("paging", {})
            next_url = paging.get("next")
            params = {}  # Очищаємо params для наступних запитів (URL вже містить їх)

            time.sleep(0.2)  # Rate limiting

        except requests.RequestException as e:
            logger.error(f"Помилка запиту Mapillary: {e}")
            break

    all_images = all_images[:max_images]
    logger.info(f"Всього метаданих: {len(all_images)}. Завантаження зображень...")

    # Паралельне завантаження зображень
    manifest_rows: list[dict] = []

    def _download_img(img_meta: dict) -> Optional[dict]:
        img_id = img_meta.get("id", "")
        geometry = img_meta.get("geometry", {})
        coords = geometry.get("coordinates", [0, 0])
        lon, lat = float(coords[0]), float(coords[1])

        url = img_meta.get(image_size) or img_meta.get("thumb_1024_url") or \
              img_meta.get("thumb_256_url")
        if not url:
            return None

        out_path = images_dir / f"{img_id}.jpg"
        if not out_path.exists():
            success = _download_image_url(url, out_path)
            if not success:
                return None

        captured_at = img_meta.get("captured_at", "")
        capture_date = ""
        if captured_at:
            from datetime import datetime
            try:
                dt = datetime.fromtimestamp(int(captured_at) / 1000)
                capture_date = dt.strftime("%Y-%m-%d")
            except Exception:
                capture_date = str(captured_at)

        return {
            "image_id":     img_id,
            "filepath":     str(out_path.relative_to(images_dir.parent)),
            "lat":          round(lat, 6),
            "lon":          round(lon, 6),
            "country":      "UA",     # Можна визначити через reverse geocoding
            "region":       "",
            "city":         "",
            "source":       "mapillary",
            "capture_date": capture_date,
            "quality_score": 1.0,
        }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_download_img, m) for m in all_images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Завантаження"):
            row = future.result()
            if row:
                manifest_rows.append(row)

    df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    logger.info(f"Завантажено {len(df)} зображень. Маніфест: {manifest_path}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Генерація маніфесту
# ──────────────────────────────────────────────────────────────────────────────

def create_manifest(
    image_dir: Union[str, Path],
    output_csv: Union[str, Path],
    recursive: bool = True,
    default_country: str = "UA",
    extract_gps_exif: bool = True,
    quality_estimator: Optional[callable] = None,
) -> pd.DataFrame:
    """
    Створює CSV-маніфест із директорії зображень.

    Намагається зчитати GPS-координати з EXIF-метаданих.
    Якщо EXIF відсутній — записує (0.0, 0.0) як placeholder.

    Аргументи:
        image_dir:         Шлях до директорії із зображеннями.
        output_csv:        Шлях для збереження CSV-маніфесту.
        recursive:         Якщо True, обходить піддиректорії.
        default_country:   Код країни за замовчуванням.
        extract_gps_exif:  Якщо True, намагається зчитати GPS із EXIF.
        quality_estimator: Функція(PIL.Image) → float для оцінки якості.

    Повертає:
        DataFrame із маніфестом.
    """
    image_dir  = Path(image_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Директорія не знайдена: {image_dir}")

    # Збір файлів
    if recursive:
        all_files = [
            p for ext in IMAGE_EXTENSIONS
            for p in image_dir.rglob(f"*{ext}")
        ]
    else:
        all_files = [
            p for ext in IMAGE_EXTENSIONS
            for p in image_dir.glob(f"*{ext}")
        ]

    # Видаляємо дублікати
    all_files = sorted(set(all_files))
    logger.info(f"Знайдено {len(all_files)} зображень у {image_dir}")

    rows = []
    for img_path in tqdm(all_files, desc="Створення маніфесту"):
        img_id = img_path.stem
        rel_path = str(img_path.relative_to(image_dir.parent))

        lat, lon = 0.0, 0.0
        capture_date = ""
        quality_score = 1.0

        # Спроба зчитати EXIF
        if extract_gps_exif:
            lat, lon, capture_date = _extract_exif_gps(img_path)

        # Назва міста із структури директорій (image_dir/country/city/img.jpg)
        parts = img_path.relative_to(image_dir).parts
        country = default_country
        city    = ""
        region  = ""

        if len(parts) >= 3:
            country = parts[0] if len(parts[0]) == 2 else default_country
            region  = parts[1] if len(parts) >= 3 else ""
            city    = parts[-2] if len(parts) >= 2 else ""
        elif len(parts) == 2:
            city = parts[0]

        # Оцінка якості
        if quality_estimator is not None:
            try:
                from PIL import Image as PILImage
                with PILImage.open(str(img_path)) as img:
                    quality_score = float(quality_estimator(img))
            except Exception:
                quality_score = 1.0

        rows.append({
            "image_id":     img_id,
            "filepath":     rel_path,
            "lat":          round(lat, 6),
            "lon":          round(lon, 6),
            "country":      country,
            "region":       region,
            "city":         city,
            "source":       _detect_source(img_id),
            "capture_date": capture_date,
            "quality_score": round(quality_score, 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(str(output_csv), index=False, encoding="utf-8")
    logger.info(
        f"Маніфест збережено: {output_csv} "
        f"({len(df)} зображень, {df['city'].nunique()} міст)"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Допоміжні функції
# ──────────────────────────────────────────────────────────────────────────────

def _extract_exif_gps(
    img_path: Path,
) -> tuple[float, float, str]:
    """
    Зчитує GPS-координати та дату з EXIF-метаданих зображення.

    Аргументи:
        img_path: Шлях до файлу зображення.

    Повертає:
        Кортеж (lat, lon, capture_date). При відсутності EXIF → (0.0, 0.0, "").
    """
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS, GPSTAGS

        with PILImage.open(str(img_path)) as img:
            exif_data = img._getexif()
            if exif_data is None:
                return 0.0, 0.0, ""

            named_exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
            capture_date = str(named_exif.get("DateTimeOriginal", ""))

            gps_info = named_exif.get("GPSInfo", {})
            if not gps_info:
                return 0.0, 0.0, capture_date

            gps_named = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

            def _dms_to_decimal(dms, ref) -> float:
                d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
                decimal = d + m / 60.0 + s / 3600.0
                return -decimal if ref in ("S", "W") else decimal

            lat = _dms_to_decimal(
                gps_named.get("GPSLatitude", (0, 0, 0)),
                gps_named.get("GPSLatitudeRef", "N"),
            )
            lon = _dms_to_decimal(
                gps_named.get("GPSLongitude", (0, 0, 0)),
                gps_named.get("GPSLongitudeRef", "E"),
            )
            return round(lat, 6), round(lon, 6), capture_date

    except Exception:
        return 0.0, 0.0, ""


def _detect_source(img_id: str) -> str:
    """Визначає джерело за форматом image_id."""
    if len(img_id) > 15 and img_id.isdigit():
        return "mapillary"
    elif img_id.startswith("img_") or "_" in img_id:
        return "osv5m"
    return "unknown"


def _download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Завантажує файл за URL."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(str(dest), "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except Exception as e:
        logger.debug(f"Помилка завантаження {url}: {e}")
        return False


def _download_image_url(url: str, dest: Path, timeout: int = 30) -> bool:
    """Завантажує зображення за URL та зберігає як JPEG."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        from PIL import Image as PILImage
        import io

        img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(dest), "JPEG", quality=90)
        return True

    except Exception as e:
        logger.debug(f"Помилка завантаження зображення {url}: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Статистика датасету
# ──────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(manifest_path: Union[str, Path]) -> None:
    """
    Виводить статистику датасету із CSV-маніфесту.

    Аргументи:
        manifest_path: Шлях до CSV-маніфесту.
    """
    df = pd.read_csv(manifest_path)

    print(f"\n{'='*50}")
    print(f"СТАТИСТИКА ДАТАСЕТУ: {Path(manifest_path).name}")
    print(f"{'='*50}")
    print(f"Всього зображень:  {len(df):,}")

    if "country" in df.columns:
        print(f"\nРозподіл за країнами:")
        for country, count in df["country"].value_counts().head(10).items():
            print(f"  {country}: {count:,}")

    if "city" in df.columns:
        print(f"\nКількість міст: {df['city'].nunique()}")
        print(f"Топ-10 міст:")
        for city, count in df["city"].value_counts().head(10).items():
            print(f"  {city}: {count:,}")

    if "quality_score" in df.columns:
        q = df["quality_score"]
        print(f"\nЯкість зображень:")
        print(f"  Середня: {q.mean():.3f}")
        print(f"  Медіана: {q.median():.3f}")
        print(f"  ≥ 0.7:   {(q >= 0.7).sum():,} ({(q >= 0.7).mean()*100:.1f}%)")

    if "source" in df.columns:
        print(f"\nДжерела:")
        for src, count in df["source"].value_counts().items():
            print(f"  {src}: {count:,}")

    print(f"{'='*50}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Завантаження даних для геолокаційного проєкту",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── OSV-5M ───────────────────────────────────────────────────────────────
    osv_parser = subparsers.add_parser("osv5m", help="Завантажити OSV-5M")
    osv_parser.add_argument("--countries",   nargs="+", default=["UA"],
                            help="Коди країн ISO (наприклад: UA PL)")
    osv_parser.add_argument("--output",      type=str, default="data/osv5m",
                            help="Директорія для збереження")
    osv_parser.add_argument("--max-images",  type=int, default=None,
                            help="Максимум зображень на країну")
    osv_parser.add_argument("--quality",     type=float, default=0.3,
                            help="Мінімальний поріг якості")
    osv_parser.add_argument("--no-hf",       action="store_true",
                            help="Використовувати parquet замість HF datasets")
    osv_parser.add_argument("--workers",     type=int, default=8)

    # ── Mapillary ─────────────────────────────────────────────────────────────
    mly_parser = subparsers.add_parser("mapillary", help="Завантажити з Mapillary")
    mly_parser.add_argument("--bbox",        nargs=4, type=float,
                            metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
                            required=True,
                            help="Обмежувальна рамка (наприклад: 50.2 30.2 50.6 30.8)")
    mly_parser.add_argument("--api-key",     type=str,
                            default=os.environ.get("MAPILLARY_ACCESS_TOKEN", ""),
                            help="Mapillary API ключ (або $MAPILLARY_ACCESS_TOKEN)")
    mly_parser.add_argument("--output",      type=str, default="data/mapillary",
                            help="Директорія для збереження")
    mly_parser.add_argument("--max-images",  type=int, default=1000)
    mly_parser.add_argument("--no-zensvi",   action="store_true",
                            help="Пряме Mapillary API замість ZenSVI")
    mly_parser.add_argument("--workers",     type=int, default=8)
    mly_parser.add_argument("--start-date",  type=str, default=None,
                            help="Початкова дата (YYYY-MM-DD)")
    mly_parser.add_argument("--end-date",    type=str, default=None)

    # ── Manifest ──────────────────────────────────────────────────────────────
    mf_parser = subparsers.add_parser("manifest", help="Створити маніфест")
    mf_parser.add_argument("--image-dir",   type=str, required=True,
                           help="Директорія із зображеннями")
    mf_parser.add_argument("--output",      type=str, default="data/manifest.csv",
                           help="Шлях для збереження CSV")
    mf_parser.add_argument("--no-recursive", action="store_true",
                           help="Не обходити піддиректорії")
    mf_parser.add_argument("--country",     type=str, default="UA")
    mf_parser.add_argument("--stats",       action="store_true",
                           help="Вивести статистику після створення")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "osv5m":
        df = download_osv5m_subset(
            countries=args.countries,
            output_dir=args.output,
            max_images_per_country=args.max_images,
            quality_threshold=args.quality,
            use_hf_datasets=not args.no_hf,
            num_workers=args.workers,
        )
        print(f"\nОSV-5M: завантажено {len(df)} зображень у {args.output}")

    elif args.command == "mapillary":
        if not args.api_key:
            logger.error(
                "Mapillary API ключ не задано. "
                "Використайте --api-key або змінну середовища MAPILLARY_ACCESS_TOKEN"
            )
            return

        df = download_mapillary(
            bbox=tuple(args.bbox),
            api_key=args.api_key,
            output_dir=args.output,
            max_images=args.max_images,
            use_zensvi=not args.no_zensvi,
            start_date=args.start_date,
            end_date=args.end_date,
            num_workers=args.workers,
        )
        print(f"\nMapillary: завантажено {len(df)} зображень у {args.output}")

    elif args.command == "manifest":
        df = create_manifest(
            image_dir=args.image_dir,
            output_csv=args.output,
            recursive=not args.no_recursive,
            default_country=args.country,
        )
        print(f"\nМаніфест: {len(df)} зображень → {args.output}")
        if args.stats:
            print_dataset_stats(args.output)


if __name__ == "__main__":
    main()
