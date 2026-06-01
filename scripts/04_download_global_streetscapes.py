"""
04_download_global_streetscapes.py — Скачивание из Global Streetscapes dataset.

Dataset: https://huggingface.co/datasets/NUS-UAL/global-streetscapes
Структура: метаданные в streetscapes.parquet, изображения по пути img/<uuid[0]>/<uuid>.jpeg

Цель: Скачать ~2500 фото для каждого доступного города:
    - Warsaw (Варшава, Польша)  — 23,668 записей
    - Prague (Прага, Чехия)     — 54,660 записей
    - Kyiv (Киев, Украина)      — отсутствует в датасете, пропускается

Требования:
    - HF_TOKEN в переменных окружения

Запуск:
    python scripts/04_download_global_streetscapes.py
"""

import os
import sys
import random
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "NUS-UAL/global-streetscapes"
TARGET_PER_CITY = 2500
CITY_NAMES = {
    "warsaw": "Warsaw",
    "prague": "Prague",
}
RANDOM_SEED = 42


def load_metadata() -> pd.DataFrame:
    """Скачивает и загружает streetscapes.parquet с координатами и uuid."""
    print(">>> Загрузка streetscapes.parquet...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="data/parquet/streetscapes.parquet",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    df = pd.read_parquet(path, columns=["uuid", "lat", "lon", "city", "country", "iso2"])
    print(f"   Всего записей: {len(df):,}")
    return df


def filter_cities(df: pd.DataFrame) -> dict:
    """Фильтрует записи по целевым городам, выбирает до TARGET_PER_CITY каждого."""
    result = {}
    rng = random.Random(RANDOM_SEED)

    for city_key, city_label in CITY_NAMES.items():
        subset = df[df["city"] == city_label].copy()
        print(f"   {city_label}: найдено {len(subset):,} записей")

        if len(subset) == 0:
            print(f"   [SKIP] {city_label} не найден в датасете")
            continue

        if len(subset) > TARGET_PER_CITY:
            subset = subset.sample(n=TARGET_PER_CITY, random_state=RANDOM_SEED)

        result[city_key] = subset
        print(f"   {city_label}: выбрано {len(subset):,}")

    return result


def img_path_from_uuid(uuid: str) -> str:
    """Формирует путь к изображению в HF-репо по uuid."""
    return f"img/{uuid[0]}/{uuid}.jpeg"


def download_images(city_data: dict) -> list:
    """Скачивает изображения из HF-репо для каждого города."""
    output_dir = Path("dataset/raw/global_streetscapes/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for city_key, subset in city_data.items():
        city_dir = output_dir / city_key
        city_dir.mkdir(exist_ok=True)

        print(f"\n>>> Скачивание {city_key} ({len(subset):,} фото)...")
        skipped = 0
        downloaded = 0
        errors = 0

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=city_key):
            uuid = str(row["uuid"])
            hf_path = img_path_from_uuid(uuid)
            out_path = city_dir / f"{uuid}.jpeg"

            if out_path.exists() and out_path.stat().st_size > 1000:
                skipped += 1
            else:
                try:
                    cached = hf_hub_download(
                        repo_id=REPO_ID,
                        filename=hf_path,
                        repo_type="dataset",
                        token=HF_TOKEN,
                    )
                    shutil.copy2(cached, out_path)
                    downloaded += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"\n   [WARN] {uuid}: {e}")
                    continue

            manifest_rows.append({
                "image_id": uuid,
                "filepath": f"raw/global_streetscapes/images/{city_key}/{uuid}.jpeg",
                "lat": round(float(row["lat"]), 6),
                "lon": round(float(row["lon"]), 6),
                "country": str(row.get("iso2", "")),
                "region": "",
                "city": city_key,
                "source": "global_streetscapes",
                "capture_date": "",
                "quality_score": 1.0,
            })

        print(f"   [OK] {city_key}: скачано={downloaded}, пропущено={skipped}, ошибок={errors}")

    return manifest_rows


def save_manifest(rows: list) -> Path:
    """Сохраняет манифест."""
    manifest_df = pd.DataFrame(rows)
    out = Path("dataset/raw/global_streetscapes/manifest.csv")
    manifest_df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n>>> Манифест: {len(manifest_df):,} записей")
    if "city" in manifest_df.columns:
        print(manifest_df["city"].value_counts().to_string())
    print(f"[OK] Сохранен: {out.absolute()}")
    return out


def main():
    print("=" * 70)
    print("Global Streetscapes Dataset Download")
    print("=" * 70 + "\n")

    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN не установлен!")
        print("  $env:HF_TOKEN = 'hf_...'")
        return

    df = load_metadata()
    city_data = filter_cities(df)

    if not city_data:
        print("[ERROR] Нет данных ни для одного города")
        return

    rows = download_images(city_data)
    save_manifest(rows)

    print("\n" + "=" * 70)
    print("[SUCCESS] Готово!")
    print("=" * 70)
    print("  Изображения: dataset/raw/global_streetscapes/images/")
    print("  Следующий шаг: python scripts/generate_manifests.py")


if __name__ == "__main__":
    main()
