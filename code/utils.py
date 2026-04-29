"""
utils.py — Допоміжні утиліти для геолокаційного проєкту.

Містить:
- Зворотне геокодування через Nominatim (geopy)
- Завантаження GeoJSON-меж адмінрегіонів
- Визначення регіону за координатами (geopandas sjoin)
- seed_everything для відтворюваності експериментів
- Інші корисні утиліти
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Відтворюваність
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """
    Встановлює фіксований seed для всіх джерел випадковості.

    Фіксує: Python random, NumPy, PyTorch (CPU та GPU).
    Також вмикає детерміністичні алгоритми CUDA за наявності.

    Аргументи:
        seed: Значення seed (за замовчуванням 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Детерміністичні операції CUDA (може уповільнити роботу)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Seed встановлено: {seed}")


# ──────────────────────────────────────────────────────────────────────────────
# Зворотне геокодування
# ──────────────────────────────────────────────────────────────────────────────

def reverse_geocode(
    lat: float,
    lon: float,
    language: str = "uk",
    timeout: float = 5.0,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> dict[str, Optional[str]]:
    """
    Визначає адресу за географічними координатами через Nominatim (OpenStreetMap).

    Аргументи:
        lat:         Широта точки.
        lon:         Довгота точки.
        language:    Мова відповіді ('uk', 'en', тощо).
        timeout:     Таймаут запиту в секундах.
        retry_count: Кількість повторних спроб при помилці.
        retry_delay: Затримка між повторними спробами (секунди).

    Повертає:
        Словник із полями:
        {
            "country": str | None,
            "country_code": str | None,
            "state": str | None,
            "city": str | None,
            "district": str | None,
            "road": str | None,
            "display_name": str | None,
        }

    Примітки:
        - Nominatim має ліміт 1 запит/секунду. Використовуйте кеш для великих обсягів.
        - За умовами використання Nominatim слід вказувати правильний User-Agent.
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    except ImportError:
        raise ImportError("Встановіть geopy: pip install geopy")

    geocoder = Nominatim(
        user_agent="ua_geolocation_diploma/1.0 (contact: research@example.com)",
        timeout=timeout,
    )

    empty_result: dict[str, Optional[str]] = {
        "country": None,
        "country_code": None,
        "state": None,
        "city": None,
        "district": None,
        "road": None,
        "display_name": None,
    }

    for attempt in range(retry_count):
        try:
            location = geocoder.reverse(
                f"{lat}, {lon}",
                language=language,
                exactly_one=True,
            )
            if location is None:
                return empty_result

            addr = location.raw.get("address", {})
            return {
                "country":      addr.get("country"),
                "country_code": addr.get("country_code", "").upper() or None,
                "state":        addr.get("state") or addr.get("region"),
                "city":         (
                    addr.get("city")
                    or addr.get("town")
                    or addr.get("village")
                    or addr.get("municipality")
                ),
                "district":     addr.get("city_district") or addr.get("suburb"),
                "road":         addr.get("road"),
                "display_name": location.address,
            }

        except GeocoderTimedOut:
            logger.warning(f"Nominatim timeout (спроба {attempt + 1}/{retry_count})")
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
        except GeocoderServiceError as e:
            logger.error(f"Nominatim сервісна помилка: {e}")
            break
        except Exception as e:
            logger.error(f"Неочікувана помилка геокодування: {e}")
            break

    return empty_result


def reverse_geocode_batch(
    coords: list[tuple[float, float]],
    delay: float = 1.1,
    language: str = "uk",
) -> list[dict[str, Optional[str]]]:
    """
    Пакетне зворотне геокодування зі збереженням ліміту Nominatim (1 req/s).

    Аргументи:
        coords:   Список пар (lat, lon).
        delay:    Затримка між запитами (секунди).
        language: Мова відповіді.

    Повертає:
        Список словників із адресами (той же порядок, що й coords).
    """
    results = []
    for i, (lat, lon) in enumerate(coords):
        result = reverse_geocode(lat, lon, language=language)
        results.append(result)
        if i < len(coords) - 1:
            time.sleep(delay)
        if (i + 1) % 10 == 0:
            logger.info(f"Геокодування: {i + 1}/{len(coords)}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON та адміністративні межі
# ──────────────────────────────────────────────────────────────────────────────

def load_geojson_boundaries(path: Union[str, Path]) -> "geopandas.GeoDataFrame":
    """
    Завантажує GeoJSON-файл із адміністративними межами у GeoDataFrame.

    Аргументи:
        path: Шлях до GeoJSON-файлу (наприклад, межі областей України).

    Повертає:
        geopandas.GeoDataFrame із геометрією та атрибутами регіонів.

    Примітки:
        - GeoJSON-файли меж України можна знайти на geoBoundaries.org
        - Очікується, що файл має поле 'geometry' типу Polygon/MultiPolygon
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("Встановіть geopandas: pip install geopandas")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON-файл не знайдено: {path}")

    gdf = gpd.read_file(str(path))

    # Переводимо у стандартну систему координат WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    logger.info(f"Завантажено {len(gdf)} регіонів із {path.name}")
    return gdf


def assign_region(
    lat: float,
    lon: float,
    gdf: "geopandas.GeoDataFrame",
    region_col: str = "name",
) -> Optional[str]:
    """
    Визначає назву адміністративного регіону за координатами через sjoin.

    Аргументи:
        lat:        Широта точки.
        lon:        Довгота точки.
        gdf:        GeoDataFrame із межами регіонів (EPSG:4326).
        region_col: Назва колонки з іменами регіонів у gdf.

    Повертає:
        Назву регіону або None, якщо точка поза межами.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise ImportError("Встановіть geopandas та shapely: pip install geopandas shapely")

    point_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(lon, lat)]},
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(point_gdf, gdf[[region_col, "geometry"]], how="left", predicate="within")

    if joined.empty or joined[region_col].isna().all():
        # Fallback: знаходимо найближчий полігон
        distances = gdf.geometry.distance(Point(lon, lat))
        nearest_idx = distances.idxmin()
        return str(gdf.loc[nearest_idx, region_col])

    return str(joined[region_col].iloc[0])


def assign_regions_batch(
    coords_df: "pandas.DataFrame",
    gdf: "geopandas.GeoDataFrame",
    lat_col: str = "lat",
    lon_col: str = "lon",
    region_col: str = "name",
    output_col: str = "region",
) -> "pandas.DataFrame":
    """
    Пакетне визначення регіонів для DataFrame із координатами.

    Аргументи:
        coords_df:  DataFrame із координатами.
        gdf:        GeoDataFrame із межами регіонів.
        lat_col:    Назва колонки широти.
        lon_col:    Назва колонки довготи.
        region_col: Назва колонки регіонів у gdf.
        output_col: Назва нової колонки у coords_df для регіонів.

    Повертає:
        coords_df із доданою колонкою регіонів.
    """
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point
    except ImportError:
        raise ImportError("Встановіть geopandas: pip install geopandas")

    geometry = [Point(row[lon_col], row[lat_col]) for _, row in coords_df.iterrows()]
    points_gdf = gpd.GeoDataFrame(coords_df.copy(), geometry=geometry, crs="EPSG:4326")

    joined = gpd.sjoin(
        points_gdf,
        gdf[[region_col, "geometry"]],
        how="left",
        predicate="within",
    )

    coords_df = coords_df.copy()
    coords_df[output_col] = joined[region_col].values
    return coords_df


# ──────────────────────────────────────────────────────────────────────────────
# Утиліти для роботи з координатами
# ──────────────────────────────────────────────────────────────────────────────

def coords_to_xyz(lat: float, lon: float) -> tuple[float, float, float]:
    """
    Перетворює сферичні координати (lat, lon) на 3D декартові (x, y, z).

    Корисно для кластеризації координат без розриву на лінії дат зміни.

    Аргументи:
        lat: Широта (градуси).
        lon: Довгота (градуси).

    Повертає:
        Кортеж (x, y, z) на одиничній сфері.
    """
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return float(x), float(y), float(z)


def xyz_to_coords(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Зворотне перетворення із 3D декартових координат у (lat, lon).

    Аргументи:
        x, y, z: Декартові координати на одиничній сфері.

    Повертає:
        Кортеж (lat, lon) у градусах.
    """
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return float(lat), float(lon)


def encode_coords_fourier(
    lat: Union[float, np.ndarray, torch.Tensor],
    lon: Union[float, np.ndarray, torch.Tensor],
    num_frequencies: int = 64,
) -> np.ndarray:
    """
    Кодує координати за допомогою Fourier Features для нейронних мереж.

    Перетворює (lat, lon) у вектор розміром 2 * num_frequencies * 2
    для подачі до MLP-кодувальника місцезнаходження.

    Аргументи:
        lat:             Широта або масив широт.
        lon:             Довгота або масив довгот.
        num_frequencies: Кількість частот для кодування.

    Повертає:
        numpy.ndarray форми (..., 4 * num_frequencies).
    """
    lat_arr = np.asarray(lat, dtype=np.float32)
    lon_arr = np.asarray(lon, dtype=np.float32)

    # Нормалізація до [-1, 1]
    lat_norm = lat_arr / 90.0
    lon_norm = lon_arr / 180.0

    coords = np.stack([lat_norm, lon_norm], axis=-1)  # (..., 2)

    # Матриця частот
    freqs = np.arange(1, num_frequencies + 1, dtype=np.float32)  # (F,)
    # Outer product: (..., 2, F)
    angles = coords[..., np.newaxis] * freqs * np.pi  # broadcasting

    sin_feats = np.sin(angles)
    cos_feats = np.cos(angles)

    # Конкатенація: (..., 4*F)
    encoded = np.concatenate([
        sin_feats.reshape(*coords.shape[:-1], -1),
        cos_feats.reshape(*coords.shape[:-1], -1),
    ], axis=-1)

    return encoded


# ──────────────────────────────────────────────────────────────────────────────
# Утиліти для роботи з файлами конфігурації
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path]) -> dict:
    """
    Завантажує конфігураційний файл у форматі YAML або JSON.

    Аргументи:
        config_path: Шлях до файлу конфігурації.

    Повертає:
        Словник із параметрами конфігурації.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфіг не знайдено: {config_path}")

    if config_path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("Встановіть PyYAML: pip install PyYAML")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    elif config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(f"Непідтримуваний формат конфігу: {config_path.suffix}")


def save_config(config: dict, path: Union[str, Path]) -> None:
    """
    Зберігає словник конфігурації у YAML або JSON файл.

    Аргументи:
        config: Словник конфігурації.
        path:   Шлях для збереження.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("Встановіть PyYAML: pip install PyYAML")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Повертає найкращий доступний пристрій для PyTorch.

    Аргументи:
        prefer_cuda: Якщо True, надає перевагу CUDA перед CPU.

    Повертає:
        torch.device ('cuda', 'mps' або 'cpu').
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Використовується GPU: {gpu_name}")
    elif prefer_cuda and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Використовується Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Використовується CPU")
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Підраховує кількість параметрів моделі.

    Аргументи:
        model:          PyTorch модель.
        trainable_only: Якщо True, рахує лише параметри з requires_grad=True.

    Повертає:
        Кількість параметрів.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_param_count(n: int) -> str:
    """Форматує кількість параметрів у зручний вигляд (M, K)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ──────────────────────────────────────────────────────────────────────────────
# Тест при прямому запуску
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Тест утиліт ===")

    # Seed
    seed_everything(42)
    print("seed_everything(42) — OK")

    # Перетворення координат
    x, y, z = coords_to_xyz(50.45, 30.52)
    lat, lon = xyz_to_coords(x, y, z)
    assert abs(lat - 50.45) < 1e-5 and abs(lon - 30.52) < 1e-5
    print(f"coords_to_xyz / xyz_to_coords — OK: ({lat:.2f}, {lon:.2f})")

    # Fourier encoding
    encoded = encode_coords_fourier(50.45, 30.52, num_frequencies=32)
    assert encoded.shape == (128,)
    print(f"encode_coords_fourier — OK: shape={encoded.shape}")

    # Пристрій
    device = get_device()
    print(f"Пристрій: {device}")

    # Зворотне геокодування (тест API-запиту)
    print("Тест reverse_geocode (Київ)...")
    result = reverse_geocode(50.4501, 30.5234)
    print(f"  Результат: {result}")

    print("Всі тести пройдено успішно!")
