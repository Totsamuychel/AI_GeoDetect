"""
02_filter_osv5m.py — Фильтрация метаданных OSV-5M по полигонам городов.

Скачивает метаданные OSV-5M из HuggingFace и фильтрует по точным границам городов,
исключая трассы, пригороды и сельские территории.

Требования:
    - Наличие файлов полигонов в dataset/raw/boundaries/
    - HF_TOKEN в переменных окружения для доступа к OSV-5M

Запуск:
    python scripts/02_filter_osv5m.py
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path
from shapely.geometry import Point, shape
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Константы
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "osv5m/osv5m"
CITIES = ["warsaw", "prague", "budapest"]

def load_city_polygons():
    """
    Загружает полигоны городов из GeoJSON файлов.

    Returns:
        Dict[str, Polygon]: Словарь {city_name: shapely.Polygon}
    """
    city_polygons = {}
    boundaries_dir = Path("dataset/raw/boundaries")

    print(">>> Загрузка полигонов городов...")
    for city in CITIES:
        geojson_path = boundaries_dir / f"{city}.geojson"

        if not geojson_path.exists():
            raise FileNotFoundError(
                f"Полигон для {city} не найден: {geojson_path}\n"
                f"Запустите сначала: python scripts/01_get_city_polygons.py"
            )

        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        # Берём первую геометрию из FeatureCollection
        geom = gj["features"][0]["geometry"]
        city_polygons[city] = shape(geom)
        print(f"   [OK] {city.capitalize()}: полигон загружен")

    return city_polygons


def download_osv5m_metadata():
    """
    Скачивает метаданные OSV-5M (train.csv) из HuggingFace.

    Returns:
        Path: Путь к скачанному CSV файлу
    """
    output_dir = Path("dataset/raw/osv5m")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📦 Скачивание метаданных OSV-5M...")

    try:
        # OSV-5M метаданные хранятся в train.csv (основной файл)
        csv_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="train.csv",
            repo_type="dataset",
            cache_dir=str(output_dir / ".cache"),
            token=HF_TOKEN
        )

        print(f"   [OK] Метаданные скачаны: {csv_path}")
        return Path(csv_path)

    except Exception as e:
        raise RuntimeError(
            f"Ошибка скачивания метаданных OSV-5M: {e}\n"
            f"Убедитесь, что HF_TOKEN установлен: export HF_TOKEN=your_token"
        )


def filter_by_polygons(df, city_polygons):
    """
    Фильтрует DataFrame по полигонам городов.

    Args:
        df: DataFrame с метаданными OSV-5M (должен содержать 'latitude', 'longitude')
        city_polygons: Dict[str, Polygon] - полигоны городов

    Returns:
        DataFrame: Отфильтрованный DataFrame с добавленной колонкой 'city'
    """
    frames = []

    print("\n🔍 Фильтрация по полигонам городов...")
    print(f"   Всего записей в OSV-5M: {len(df):,}\n")

    for city, polygon in city_polygons.items():
        print(f"   🏙️ {city.capitalize()}:")

        # Грубая предварительная фильтрация по bounding box (намного быстрее)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        bbox_filter = df[
            df["latitude"].between(bounds[1], bounds[3]) &
            df["longitude"].between(bounds[0], bounds[2])
        ].copy()

        print(f"      - После bbox-фильтрации: {len(bbox_filter):,}")

        # Точная фильтрация по полигону (медленнее, но точная)
        print(f"      - Проверка попадания в полигон...", end="", flush=True)

        mask = bbox_filter.apply(
            lambda row: polygon.contains(Point(row["longitude"], row["latitude"])),
            axis=1
        )

        city_data = bbox_filter[mask].copy()
        city_data["city"] = city
        city_data["source"] = "osv5m"
        frames.append(city_data)

        print(f" ✅ {len(city_data):,} фото внутри границ")

    # Объединяем результаты
    result = pd.concat(frames, ignore_index=True)

    # Удаляем дубликаты по ID
    initial_count = len(result)
    result = result.drop_duplicates(subset=["id"])
    duplicates = initial_count - len(result)

    if duplicates > 0:
        print(f"\n   🔄 Удалено дубликатов: {duplicates}")

    return result


def main():
    """
    Основная функция: загрузка полигонов, скачивание и фильтрация метаданных.
    """
    print("=" * 70)
    print("OSV-5M Metadata Filtering by City Boundaries")
    print("=" * 70 + "\n")

    # 1. Загружаем полигоны городов
    city_polygons = load_city_polygons()

    # 2. Скачиваем метаданные OSV-5M
    csv_path = download_osv5m_metadata()

    # 3. Читаем метаданные
    print("\n>>> Чтение метаданных OSV-5M...")
    df = pd.read_csv(csv_path, dtype={"id": str})

    # ВАЖНО: Сохраняем row_index для определения шардов
    df['row_index'] = df.index

    print(f"   [OK] Загружено {len(df):,} записей")

    # 4. Фильтруем по полигонам
    filtered = filter_by_polygons(df, city_polygons)

    # 5. Сохраняем результат
    output_path = Path("dataset/raw/osv5m/filtered_cities.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)

    # 6. Статистика
    print("\n" + "=" * 70)
    print(">>> ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"\nРаспределение по городам:")
    city_counts = filtered["city"].value_counts()
    for city, count in city_counts.items():
        print(f"   {city.capitalize():10s}: {count:,} фото")

    print(f"\n   {'ИТОГО':10s}: {len(filtered):,} фото")
    print(f"\n✅ Результаты сохранены: {output_path.absolute()}")
    print("\nСледующий шаг:")
    print("   python scripts/03_download_osv5m_images.py")


if __name__ == "__main__":
    main()
