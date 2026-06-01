"""
04_download_global_streetscapes_expanded.py — Скачивание расширенного набора из Global Streetscapes.

Цель: Скачать до 5000 фото для Warsaw, Prague и Budapest.
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
TARGET_PER_CITY = 5000  # Увеличено до 5000
CITY_NAMES = {
    "warsaw": "Warsaw",
    "prague": "Prague",
    "budapest": "Budapest", # Добавлен Будапешт
}
RANDOM_SEED = 42


def load_metadata() -> pd.DataFrame:
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
    result = {}
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
    return f"img/{uuid[0]}/{uuid}.jpeg"


def download_images(city_data: dict) -> list:
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
    manifest_df = pd.DataFrame(rows)
    out = Path("dataset/raw/global_streetscapes/manifest.csv")
    manifest_df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n>>> Манифест обновлен: {len(manifest_df):,} записей")
    return out


def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN не установлен!")
        return

    df = load_metadata()
    city_data = filter_cities(df)
    rows = download_images(city_data)
    save_manifest(rows)
    print("\n[SUCCESS] Фото из Global Streetscapes (Prague/Budapest/Warsaw) скачаны!")


if __name__ == "__main__":
    main()
