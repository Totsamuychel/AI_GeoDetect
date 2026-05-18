# Work Notes - AI_GeoDetect Diploma Project

Контекст проекта
Проект: AI_GeoDetect — геолокация уличных фото нейросетью.
Стек: Python 3.10, PyTorch, OSV-5M (HuggingFace), Mapillary (ZenSVI), osmnx, H3.
Цель датасета: классификация по 3 городам (Kyiv / Warsaw / Prague).
Целевой размер: докачать уже к готовым фото ~2000–2500 фото на город, итого ~10 000 фото.
Критерий качества: только городская застройка, без трасс, пригородов, полей.


ЭТАП 1: Получить точные полигоны городов через OSM
Важно: используем реальные границы города (не bbox-прямоугольники) — это исключит пригороды и трассы за городом.

Создай файл scripts/01_get_city_polygons.py:

python
import osmnx as ox
import json, os

CITIES = {
    "kyiv":   "Kyiv, Ukraine",
    "warsaw": "Warsaw, Poland",
    "prague": "Prague, Czech Republic",
}

os.makedirs("dataset/raw/boundaries", exist_ok=True)

for city_key, city_name in CITIES.items():
    print(f"📍 Получаем полигон: {city_name}")
    gdf = ox.geocode_to_gdf(city_name)
    geojson = json.loads(gdf.to_json())
    with open(f"dataset/raw/boundaries/{city_key}.geojson", "w") as f:
        json.dump(geojson, f)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"   bbox: lat [{bounds[1]:.3f}, {bounds[3]:.3f}], lon [{bounds[0]:.3f}, {bounds[2]:.3f}]")
    print(f"   ✅ Сохранён: dataset/raw/boundaries/{city_key}.geojson")

print("\n✅ Все полигоны получены")
Запустить:

bash
python scripts/01_get_city_polygons.py
ЭТАП 2: Скачать метаданные OSV-5M и отфильтровать по полигонам
Создай файл scripts/02_filter_osv5m.py:

python
import pandas as pd
import json
from shapely.geometry import Point, shape
from huggingface_hub import hf_hub_download
import os

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "osv5m/osv5m"

# --- Загрузить полигоны ---
city_polygons = {}
for city in ["kyiv", "warsaw", "prague"]:
    with open(f"dataset/raw/boundaries/{city}.geojson") as f:
        gj = json.load(f)
    # Берём первую геометрию из FeatureCollection
    geom = gj["features"][0]["geometry"]
    city_polygons[city] = shape(geom)
    print(f"✅ Полигон {city} загружен")

# --- Скачать parquet метаданные OSV-5M ---
print("\n📦 Скачиваем метаданные OSV-5M...")
for split in ["train", "val", "test"]:
    path = f"dataset/raw/osv5m/{split}.parquet"
    if not os.path.exists(path):
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{split}.parquet",
            repo_type="dataset",
            local_dir="dataset/raw/osv5m",
            token=HF_TOKEN
        )
        print(f"  ✅ {split}.parquet скачан")
    else:
        print(f"  ⏭️  {split}.parquet уже есть")

# --- Фильтрация ---
frames = []

for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"dataset/raw/osv5m/{split}.parquet")
    print(f"\n📊 {split}: {len(df):,} строк")

    for city, polygon in city_polygons.items():
        # Грубая pre-фильтрация по bbox (быстро)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        pre = df[
            df["latitude"].between(bounds[1], bounds[3]) &
            df["longitude"].between(bounds[0], bounds[2])
        ].copy()
        print(f"  {city}: {len(pre)} после bbox-фильтра...", end="")

        # Точная фильтрация по полигону
        mask = pre.apply(
            lambda r: polygon.contains(Point(r["longitude"], r["latitude"])),
            axis=1
        )
        sub = pre[mask].copy()
        sub["city"] = city
        sub["source"] = "osv5m"
        frames.append(sub)
        print(f" → {len(sub)} внутри полигона ✅")

result = pd.concat(frames, ignore_index=True)
result = result.drop_duplicates(subset=["id"])
result.to_parquet("dataset/raw/osv5m/filtered_cities.parquet", index=False)

print(f"\n📊 Итого OSV-5M после фильтрации по полигонам:")
print(result["city"].value_counts())
print(f"\nВсего: {len(result):,} фото")
Запустить:

bash
python scripts/02_filter_osv5m.py
Ожидаемый результат:

text
kyiv:   ~800–1500 фото
warsaw: ~600–1200 фото
prague: ~500–1000 фото
ЭТАП 3: Скачать изображения OSV-5M (только нужные шарды)
Создай файл scripts/03_download_osv5m_images.py:

python
import pandas as pd
from huggingface_hub import hf_hub_download
import zipfile, os, shutil
from tqdm import tqdm

df = pd.read_parquet("dataset/raw/osv5m/filtered_cities.parquet")
HF_TOKEN = os.environ["HF_TOKEN"]

# Определяем нужные шарды (первые 2 символа ID)
needed_shards = set(df["id"].astype(str).str.zfill(8).str[:2].unique())
print(f"📦 Нужно шардов: {len(needed_shards)} → {sorted(needed_shards)}")

valid_ids = set(df["id"].astype(str).values)
downloaded = 0

for shard_id in tqdm(sorted(needed_shards), desc="Шарды"):
    try:
        zip_path = hf_hub_download(
            repo_id="osv5m/osv5m",
            filename=f"images/{shard_id}.zip",
            repo_type="dataset",
            local_dir="dataset/raw/osv5m",
            token=HF_TOKEN
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                img_id = os.path.splitext(os.path.basename(member))[0]