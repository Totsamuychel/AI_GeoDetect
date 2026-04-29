# 🗺️ Dataset Assembly Guide — Street-Level Geolocation (8 Cities)

> **Для Gemini CLI / OpenCode + Claude Opus 4.7**
> Инструкция по сборке датасета из ~35 000 уличных фото (Київ, Львів, Одеса, Харків, Дніпро, Варшава, Прага, Будапешт)
> Источники: **OSV-5M** (HuggingFace) + **Mapillary** (ZenSVI API)

---

## 📋 Требования

### Системные зависимости
```bash
# Python 3.10+
python --version

# Установка всех зависимостей одной командой
pip install huggingface_hub datasets zensvi h3 geopy imagehash             scipy pandas pillow tqdm geopandas shapely pyarrow             requests torch torchvision
```

### Нужные API-ключи
| Сервис | Где получить | Стоимость |
|---|---|---|
| **Mapillary** | [mapillary.com/dashboard/developers](https://www.mapillary.com/dashboard/developers) | Бесплатно |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Бесплатно |

### Экспортируй ключи
```bash
export MAPILLARY_API_KEY="MLY|xxxxxxxxxxxxxxx"
export HF_TOKEN="hf_xxxxxxxxxxxxxxx"
```

---

## 📁 Структура проекта

```
dataset/
├── raw/
│   ├── osv5m/           # Скачанные данные OSV-5M
│   └── mapillary/       # Скачанные данные Mapillary
├── processed/
│   └── images/          # Финальные отфильтрованные изображения
└── manifests/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

```bash
mkdir -p dataset/raw/osv5m dataset/raw/mapillary dataset/processed/images dataset/manifests
```

---

## 🗺️ Bounding Boxes городов (из диплома, таблица 2.2)

```python
# cities_bbox.py
CITIES_BBOX = {
    "kyiv":     {"lat_min": 50.30, "lat_max": 50.60, "lon_min": 30.20, "lon_max": 30.85, "country": "UA"},
    "lviv":     {"lat_min": 49.77, "lat_max": 49.90, "lon_min": 23.90, "lon_max": 24.15, "country": "UA"},
    "odesa":    {"lat_min": 46.35, "lat_max": 46.55, "lon_min": 30.60, "lon_max": 30.82, "country": "UA"},
    "kharkiv":  {"lat_min": 49.90, "lat_max": 50.10, "lon_min": 36.10, "lon_max": 36.42, "country": "UA"},
    "dnipro":   {"lat_min": 48.38, "lat_max": 48.55, "lon_min": 34.90, "lon_max": 35.20, "country": "UA"},
    "warsaw":   {"lat_min": 52.10, "lat_max": 52.37, "lon_min": 20.80, "lon_max": 21.20, "country": "PL"},
    "prague":   {"lat_min": 49.94, "lat_max": 50.18, "lon_min": 14.22, "lon_max": 14.71, "country": "CZ"},
    "budapest": {"lat_min": 47.35, "lat_max": 47.61, "lon_min": 18.90, "lon_max": 19.20, "country": "HU"},
}
```

---

## ЭТАП 1: OSV-5M (HuggingFace)

> OSV-5M — 5.1M геопривязанных фото из 225 стран, лицензия CC-BY-SA.

### 1.1 Скачать метаданные (Parquet)

```python
# 01_download_osv5m.py
from huggingface_hub import hf_hub_download
import pandas as pd
import os

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "osv5m/osv5m"

# Скачать только метаданные сначала (train/val/test parquet)
for split in ["train", "val", "test"]:
    hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{split}.parquet",
        repo_type="dataset",
        local_dir="dataset/raw/osv5m",
        token=HF_TOKEN
    )
    print(f"✅ {split}.parquet скачан")
```

### 1.2 Фильтрация по Bounding Box

```python
# 02_filter_osv5m.py
import pandas as pd
from cities_bbox import CITIES_BBOX

frames = []

for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"dataset/raw/osv5m/{split}.parquet")
    print(f"📦 {split}: {len(df):,} строк до фильтрации")

    for city, bbox in CITIES_BBOX.items():
        mask = (
            df["latitude"].between(bbox["lat_min"], bbox["lat_max"]) &
            df["longitude"].between(bbox["lon_min"], bbox["lon_max"])
        )
        sub = df[mask].copy()
        sub["city"] = city
        sub["country"] = bbox["country"]
        sub["source"] = "osv5m"
        sub["split_original"] = split
        frames.append(sub)
        print(f"  📍 {city}: {len(sub)} фото")

result = pd.concat(frames, ignore_index=True)
result = result.drop_duplicates(subset=["id"])
result.to_parquet("dataset/raw/osv5m/filtered_cities.parquet", index=False)
print(f"\n✅ Всего OSV-5M после фильтрации: {len(result):,}")
```

### 1.3 Скачать изображения OSV-5M

```python
# 03_download_osv5m_images.py
import pandas as pd
from huggingface_hub import hf_hub_download
import zipfile, os, shutil
from tqdm import tqdm

df = pd.read_parquet("dataset/raw/osv5m/filtered_cities.parquet")
# Определяем нужные zip-шарды (id // 1000 -> номер шарда)
needed_shards = set(df["id"].astype(str).str[:2].unique())

for shard_id in tqdm(needed_shards, desc="Скачиваем шарды"):
    zip_path = hf_hub_download(
        repo_id="osv5m/osv5m",
        filename=f"images/{shard_id.zfill(2)}.zip",
        repo_type="dataset",
        local_dir="dataset/raw/osv5m",
        token=os.environ["HF_TOKEN"]
    )
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            img_id = os.path.splitext(os.path.basename(member))[0]
            if img_id in df["id"].astype(str).values:
                z.extract(member, "dataset/raw/osv5m/images/")

print("✅ Изображения OSV-5M скачаны")
```

---

## ЭТАП 2: Mapillary через ZenSVI

> ZenSVI автоматически скачивает фото через Vector Tiles API Mapillary по GeoJSON границам города.

### 2.1 Получить GeoJSON границы городов (через Nominatim)

```python
# 04_get_city_boundaries.py
from geopy.geocoders import Nominatim
from geopy.extra.ratelimiter import RateLimiter
import json, os, time

geolocator = Nominatim(user_agent="dataset_builder_thesis")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

cities = {
    "kyiv": "Kyiv, Ukraine",
    "lviv": "Lviv, Ukraine",
    "odesa": "Odesa, Ukraine",
    "kharkiv": "Kharkiv, Ukraine",
    "dnipro": "Dnipro, Ukraine",
    "warsaw": "Warsaw, Poland",
    "prague": "Prague, Czech Republic",
    "budapest": "Budapest, Hungary",
}

os.makedirs("dataset/raw/boundaries", exist_ok=True)

for city_key, city_name in cities.items():
    location = geocode(city_name, geometry="geojson")
    if location:
        geojson = location.raw["geojson"]
        with open(f"dataset/raw/boundaries/{city_key}.geojson", "w") as f:
            json.dump(geojson, f)
        print(f"✅ {city_key}: GeoJSON сохранён")
    else:
        print(f"❌ {city_key}: не найден")
    time.sleep(1.5)
```

### 2.2 Скачать фото через ZenSVI

```python
# 05_download_mapillary.py
import os
from zensvi.download import MLYDownloader

MLY_API_KEY = os.environ["MAPILLARY_API_KEY"]

cities = ["kyiv", "lviv", "odesa", "kharkiv", "dnipro", "warsaw", "prague", "budapest"]

for city in cities:
    print(f"\n📸 Скачиваем {city}...")
    output_dir = f"dataset/raw/mapillary/{city}"
    os.makedirs(output_dir, exist_ok=True)

    downloader = MLYDownloader(mly_api_key=MLY_API_KEY)
    downloader.download_svi(
        dir_output=output_dir,
        input_shp_file=f"dataset/raw/boundaries/{city}.geojson",
        start_date="2019-01-01",     # Только 2019-2023 как в дипломе
        end_date="2023-12-31",
        is_pano=False,               # Только обычные фото, без панорам
        resolution=1024,
        limit=10000,                  # До 10k на город
        batch_size=1000,
        log_path=f"{output_dir}/download.log"
    )
    print(f"✅ {city}: скачано")
```

---

## ЭТАП 3: Дедупликация (KD-Tree, порог 50м)

> Удаляем фото Mapillary, у которых есть "близнец" в OSV-5M ближе 50 метров.

```python
# 06_deduplication.py
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import math

# Загрузка метаданных обоих источников
osv_df = pd.read_parquet("dataset/raw/osv5m/filtered_cities.parquet")

# Метаданные Mapillary (ZenSVI сохраняет в CSV)
import glob
mly_frames = []
for csv_file in glob.glob("dataset/raw/mapillary/*/*.csv"):
    city = csv_file.split("/")[-2]
    df = pd.read_csv(csv_file)
    df["city"] = city
    df["source"] = "mapillary"
    mly_frames.append(df)
mly_df = pd.concat(mly_frames, ignore_index=True)

# Конвертация координат в радианы для KD-Tree
def to_radians(df):
    return np.radians(df[["latitude", "longitude"]].values)

EARTH_RADIUS_M = 6371000
THRESHOLD_M = 50

osv_rad = to_radians(osv_df)
mly_rad = to_radians(mly_df)

tree = KDTree(osv_rad)
# Порог в радианах: 50m / R_earth
threshold_rad = THRESHOLD_M / EARTH_RADIUS_M

distances, _ = tree.query(mly_rad, k=1)
duplicates_mask = distances < threshold_rad

mly_unique = mly_df[~duplicates_mask].copy()
print(f"📊 Mapillary всего: {len(mly_df):,}")
print(f"🗑️  Удалено дубликатов: {duplicates_mask.sum():,}")
print(f"✅ Уникальных Mapillary: {len(mly_unique):,}")

mly_unique.to_parquet("dataset/raw/mapillary/deduplicated.parquet", index=False)
```

---

## ЭТАП 4: Контроль качества

### 4.1 Фильтр размытых фото (Laplacian Variance)

```python
# 07_quality_filter.py
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def is_blurry(image_path, threshold=100):
    """Возвращает True если фото размыто (laplacian variance < threshold)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < threshold

def is_too_small(image_path, min_size=500):
    """Возвращает True если фото меньше 500px"""
    img = cv2.imread(image_path)
    if img is None:
        return True
    h, w = img.shape[:2]
    return min(h, w) < min_size

# Применить к датасету
df = pd.read_parquet("dataset/combined_metadata.parquet")
valid_mask = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Проверка качества"):
    path = row["imagepath"]
    if not os.path.exists(path) or is_blurry(path) or is_too_small(path):
        valid_mask.append(False)
    else:
        valid_mask.append(True)

df_clean = df[valid_mask].copy()
print(f"✅ После фильтрации качества: {len(df_clean):,} / {len(df):,}")
df_clean.to_parquet("dataset/cleaned_metadata.parquet", index=False)
```

### 4.2 Размытие лиц (GDPR — deface)

```bash
# Установка deface
pip install deface

# Размытие лиц на всех изображениях
python -m deface dataset/processed/images/ --output dataset/processed/images/ --thresh 0.3
```

---

## ЭТАП 5: H3 Стратифицированное разбиение (70/15/15)

> Вместо случайного split — H3-гексагональное, исключает утечку данных между train/val/test.

```python
# 08_h3_split.py
import h3
import pandas as pd
import random

random.seed(42)

df = pd.read_parquet("dataset/cleaned_metadata.parquet")

H3_RESOLUTION = 7  # ~1.2 км² на гексагон

# 1. Добавляем H3-индекс для каждой точки
df["h3index"] = df.apply(
    lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RESOLUTION),
    axis=1
)

# 2. Уникальные гексагоны → перемешиваем
all_hexes = df["h3index"].unique().tolist()
random.shuffle(all_hexes)

# 3. Разбиваем гексагоны 70/15/15
n = len(all_hexes)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

train_hexes = set(all_hexes[:n_train])
val_hexes   = set(all_hexes[n_train:n_train + n_val])
test_hexes  = set(all_hexes[n_train + n_val:])

# 4. Назначаем split каждому изображению
def assign_split(h3idx):
    if h3idx in train_hexes: return "train"
    if h3idx in val_hexes:   return "val"
    return "test"

df["split"] = df["h3index"].apply(assign_split)

# 5. Сохраняем манифесты
for split in ["train", "val", "test"]:
    subset = df[df["split"] == split]
    subset.to_csv(f"dataset/manifests/{split}.csv", index=False)
    print(f"✅ {split}: {len(subset):,} изображений")

print(f"\n📊 Итого: {len(df):,} изображений")
print(df["split"].value_counts())
```

---

## ЭТАП 6: Финальная структура CSV

Каждый `train.csv` / `val.csv` / `test.csv` содержит колонки:

| Колонка | Тип | Описание |
|---|---|---|
| `imageid` | str | Уникальный ID фото |
| `source` | str | `osv5m` или `mapillary` |
| `latitude` | float64 | GPS широта |
| `longitude` | float64 | GPS долгота |
| `city` | str | Название города (нижний регистр) |
| `country` | str | ISO 3166-1 (UA/PL/CZ/HU) |
| `capturedate` | YYYY-MM-DD | Дата съёмки |
| `compassangle` | float | Угол компаса (0–360) |
| `split` | str | train / val / test |
| `h3index` | str | H3-индекс resolution 7 |
| `imagepath` | str | Относительный путь к файлу |

---

## ЭТАП 7: Проверка датасета

```python
# 09_validate_dataset.py
import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"dataset/manifests/{split}.csv")
    print(f"\n=== {split.upper()} ({len(df):,} изображений) ===")
    print(df["city"].value_counts())
    print(f"Источники: {df['source'].value_counts().to_dict()}")

    # Проверка баланса
    min_city = df["city"].value_counts().min()
    max_city = df["city"].value_counts().max()
    ir = max_city / min_city
    print(f"Imbalance Ratio: {ir:.2f} (норма < 3.0)")
```

**Ожидаемый результат:**
```
=== TRAIN (24 513 изображений) ===
kyiv        4550
warsaw      3570
prague      3360
...
Imbalance Ratio: 2.21 ✅
```

---

## 🚀 Быстрый запуск для Gemini CLI

```bash
# Сохрани файл как dataset_pipeline.md и запусти в Gemini CLI:
gemini "Прочитай файл dataset_pipeline.md и последовательно выполни
этапы 1-7 для сборки датасета. Начни с установки зависимостей."
```

## 🤖 Быстрый запуск для OpenCode

```bash
cd ~/your-project
opencode
# В TUI:
# /read dataset_pipeline.md
# Выполни этапы сборки датасета по инструкции
```

---

## ⚠️ Важные замечания

- **Mapillary**: при лимите в 10к запросов — ZenSVI автоматически чекпоинтится, запускай заново при ошибке
- **Время сборки**: OSV-5M фильтрация ~5 мин, Mapillary загрузка 1 города ~20–30 мин
- **Место на диске**: ~15–20 GB для 35k изображений 512px
- **deface (GDPR)**: запускай только если публикуешь датасет — замедляет обработку
- **OSV-5M шарды**: полный скачать ~800GB, фильтруй сначала через parquet метаданные!

---

*Датасет зібрано для бакалаврської дипломної роботи: "Геолокація зображень на основі нейромереж для міст України та Європи"*
*Stack: Python 3.10 · PyTorch 2.1 · HuggingFace · ZenSVI · H3 · OSV-5M · Mapillary API v4*
