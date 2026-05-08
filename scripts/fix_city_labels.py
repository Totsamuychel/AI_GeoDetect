"""
fix_city_labels.py — Исправление меток городов в манифестах через reverse geocoding.

Использует Nominatim API (OpenStreetMap) для определения города по координатам.
Includes rate limiting and caching to avoid excessive API calls.

Запуск:
    python scripts/fix_city_labels.py --input dataset/raw/osv5m/manifest.csv \
                                       --output dataset/raw/osv5m/manifest_fixed.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Целевые города для диплома
TARGET_CITIES = {
    "UA": ["Kyiv", "Lviv", "Odesa", "Kharkiv", "Dnipro"],
    "PL": ["Warsaw", "Kraków", "Gdańsk"],
    "CZ": ["Prague"],
    "HU": ["Budapest"],
    "RO": ["Bucharest"],
}

# Кэш для координат -> город (чтобы не спамить API)
GEOCODE_CACHE: dict[tuple[float, float], dict] = {}


def reverse_geocode(
    lat: float,
    lon: float,
    geolocator: Nominatim,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Определяет город, регион и страну по координатам через Nominatim API.

    Args:
        lat: Широта
        lon: Долгота
        geolocator: Nominatim geocoder instance
        max_retries: Максимальное количество попыток при ошибках

    Returns:
        Dict с полями {city, region, country} или None при ошибке
    """
    # Округляем до 2 знаков для кэширования (≈1.1 км точность)
    cache_key = (round(lat, 2), round(lon, 2))

    if cache_key in GEOCODE_CACHE:
        return GEOCODE_CACHE[cache_key]

    for attempt in range(max_retries):
        try:
            location = geolocator.reverse(
                f"{lat}, {lon}",
                language="en",
                exactly_one=True,
                timeout=10,
            )

            if location is None:
                GEOCODE_CACHE[cache_key] = None
                return None

            address = location.raw.get("address", {})

            # Извлекаем город (приоритет: city > town > village > municipality)
            city = (
                address.get("city") or
                address.get("town") or
                address.get("village") or
                address.get("municipality") or
                address.get("county") or
                ""
            )

            # Извлекаем регион (state/province)
            region = (
                address.get("state") or
                address.get("province") or
                address.get("region") or
                ""
            )

            # Код страны
            country = address.get("country_code", "").upper()

            result = {
                "city": city,
                "region": region,
                "country": country,
            }

            GEOCODE_CACHE[cache_key] = result
            return result

        except GeocoderTimedOut:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            logger.warning(f"Timeout for ({lat}, {lon}) after {max_retries} attempts")
            return None

        except GeocoderServiceError as e:
            logger.warning(f"Geocoder error for ({lat}, {lon}): {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error for ({lat}, {lon}): {e}")
            return None

    return None


def normalize_city_name(city: str, country: str) -> str:
    """
    Нормализует название города к стандартным формам.

    Примеры:
        - Kijów, Київ → Kyiv
        - Warszawa → Warsaw
        - Praha → Prague
    """
    # Словарь нормализации (локальное название → английское)
    CITY_MAPPING = {
        # Ukraine
        "Київ": "Kyiv",
        "Kijów": "Kyiv",
        "Kiev": "Kyiv",
        "Львів": "Lviv",
        "Lwów": "Lviv",
        "Одеса": "Odesa",
        "Odessa": "Odesa",
        "Харків": "Kharkiv",
        "Kharkov": "Kharkiv",
        "Дніпро": "Dnipro",
        "Dnipropetrovsk": "Dnipro",

        # Poland
        "Warszawa": "Warsaw",
        "Warschau": "Warsaw",
        "Kraków": "Kraków",
        "Cracow": "Kraków",
        "Gdańsk": "Gdańsk",
        "Danzig": "Gdańsk",

        # Czech Republic
        "Praha": "Prague",
        "Prag": "Prague",

        # Hungary
        "Budapest": "Budapest",

        # Romania
        "București": "Bucharest",
        "Bukarest": "Bucharest",
    }

    return CITY_MAPPING.get(city, city)


def filter_target_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Фильтрует датафрейм, оставляя только целевые города.

    Args:
        df: DataFrame с колонками [country, city, ...]

    Returns:
        Отфильтрованный DataFrame
    """
    mask = pd.Series([False] * len(df))

    for country, cities in TARGET_CITIES.items():
        country_mask = df["country"] == country
        city_mask = df["city"].isin(cities)
        mask |= (country_mask & city_mask)

    filtered_df = df[mask].copy()

    logger.info(f"Filtered {len(filtered_df)}/{len(df)} images matching target cities")
    logger.info("Distribution by city:")
    for city, count in filtered_df["city"].value_counts().items():
        logger.info(f"  {city}: {count:,}")

    return filtered_df


def fix_city_labels(
    input_csv: str | Path,
    output_csv: str | Path,
    user_agent: str = "AI_GeoDetect_Diploma_Project",
    rate_limit_delay: float = 1.0,
    filter_targets: bool = True,
    save_cache: bool = True,
) -> pd.DataFrame:
    """
    Исправляет метки городов в манифесте через reverse geocoding.

    Args:
        input_csv: Путь к входному CSV манифесту
        output_csv: Путь для сохранения исправленного манифеста
        user_agent: User-agent для Nominatim API
        rate_limit_delay: Задержка между запросами (секунды)
        filter_targets: Если True, оставляет только целевые города
        save_cache: Сохранить кэш geocoding для повторного использования

    Returns:
        Исправленный DataFrame
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    logger.info(f"Loading manifest: {input_csv}")
    df = pd.read_csv(input_csv)

    logger.info(f"Original manifest: {len(df)} images, {df['city'].nunique()} unique cities")
    logger.info(f"Current city distribution:\n{df['city'].value_counts().head(10)}")

    # Инициализация Nominatim geocoder
    geolocator = Nominatim(user_agent=user_agent)

    # Загрузка кэша если существует
    cache_file = Path("dataset/.cache/geocode_cache.csv")
    if cache_file.exists():
        logger.info(f"Loading geocode cache from {cache_file}")
        cache_df = pd.read_csv(cache_file)
        for _, row in cache_df.iterrows():
            key = (round(row["lat"], 2), round(row["lon"], 2))
            GEOCODE_CACHE[key] = {
                "city": row["city"],
                "region": row["region"],
                "country": row["country"],
            }
        logger.info(f"Loaded {len(GEOCODE_CACHE)} cached geocode results")

    # Обработка каждой строки
    updated_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reverse geocoding"):
        lat = float(row["lat"])
        lon = float(row["lon"])

        # Пропускаем невалидные координаты
        if lat == 0.0 and lon == 0.0:
            updated_rows.append(row)
            continue

        # Reverse geocoding
        location_info = reverse_geocode(lat, lon, geolocator)

        if location_info:
            # Нормализация названия города
            city = normalize_city_name(
                location_info["city"],
                location_info["country"]
            )

            row["city"] = city
            row["region"] = location_info["region"]
            row["country"] = location_info["country"]

        updated_rows.append(row)

        # Rate limiting (Nominatim requires 1 req/sec)
        time.sleep(rate_limit_delay)

    # Создание обновленного DataFrame
    updated_df = pd.DataFrame(updated_rows)

    logger.info(f"\nUpdated city distribution:")
    for city, count in updated_df["city"].value_counts().head(20).items():
        logger.info(f"  {city}: {count:,}")

    # Фильтрация по целевым городам
    if filter_targets:
        logger.info("\nFiltering for target cities...")
        updated_df = filter_target_cities(updated_df)

    # Сохранение результата
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"\nSaved fixed manifest: {output_csv} ({len(updated_df)} images)")

    # Сохранение кэша
    if save_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = [
            {
                "lat": key[0],
                "lon": key[1],
                "city": value["city"],
                "region": value["region"],
                "country": value["country"],
            }
            for key, value in GEOCODE_CACHE.items()
            if value is not None
        ]
        cache_df = pd.DataFrame(cache_data)
        cache_df.to_csv(cache_file, index=False)
        logger.info(f"Saved geocode cache: {cache_file} ({len(cache_df)} entries)")

    return updated_df


def main():
    parser = argparse.ArgumentParser(
        description="Fix city labels in manifest using reverse geocoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="dataset/raw/osv5m/manifest.csv",
        help="Input manifest CSV file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/raw/osv5m/manifest_fixed.csv",
        help="Output manifest CSV file"
    )

    parser.add_argument(
        "--user-agent",
        type=str,
        default="AI_GeoDetect_Diploma_Project",
        help="User agent for Nominatim API"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests (seconds)"
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter for target cities"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't save geocoding cache"
    )

    args = parser.parse_args()

    df = fix_city_labels(
        input_csv=args.input,
        output_csv=args.output,
        user_agent=args.user_agent,
        rate_limit_delay=args.delay,
        filter_targets=not args.no_filter,
        save_cache=not args.no_cache,
    )

    logger.info(f"\n✅ Done! Fixed {len(df)} images with proper city labels.")


if __name__ == "__main__":
    main()
