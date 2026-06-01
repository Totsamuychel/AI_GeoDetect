"""
03_download_osv5m_images.py — Скачивание изображений OSV-5M.

Скачивает только те изображения, которые прошли фильтрацию по полигонам городов.
Оптимизирует скачивание, загружая только нужные ZIP-шарды.

Требования:
    - Результаты 02_filter_osv5m.py (filtered_cities.parquet)
    - HF_TOKEN в переменных окружения

Особенности:
    - Идемпотентность: не скачивает уже существующие файлы
    - Организация: изображения сортируются по папкам городов
    - Генерация манифеста: создает итоговый CSV для обучения

Запуск:
    python scripts/03_download_osv5m_images.py
"""

import pandas as pd
import sys

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Константы
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "osv5m/osv5m"


def load_filtered_metadata():
    """
    Загружает отфильтрованные метаданные.

    Returns:
        DataFrame: Метаданные изображений для скачивания
    """
    parquet_path = Path("dataset/raw/osv5m/filtered_cities.parquet")

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Файл {parquet_path} не найден.\n"
            "Запустите сначала: python scripts/02_filter_osv5m.py"
        )

    df = pd.read_parquet(parquet_path)
    print(f">>> Загружено {len(df):,} записей из filtered_cities.parquet")

    return df


def determine_needed_shards(df):
    """
    Определяет какие ZIP-шарды нужно скачать.

    OSV-5M хранит изображения в ZIP-архивах по 50,000 изображений.
    Шарды нумеруются 00-99. Номер шарда: shard = row_index // 50000

    Args:
        df: DataFrame с колонкой 'row_index'

    Returns:
        set: Множество индексов шардов (0-99) для скачивания
    """
    if "row_index" not in df.columns:
        raise ValueError(
            "DataFrame не содержит 'row_index'. "
            "Перезапустите 02_filter_osv5m.py с исправленной версией скрипта."
        )

    # Вычисляем номера шардов: каждый шард содержит 50,000 изображений
    needed_shards = set((df["row_index"] // 50000).astype(int))

    print(f">>> Требуется шардов: {len(needed_shards)}")
    print(f"   Индексы: {sorted(needed_shards)}")

    # Проверка на разумность (в OSV-5M ~100 шардов, номера 0-99)
    if max(needed_shards) > 150:
        print(f"   [WARN] Обнаружены подозрительно большие номера шардов: {sorted(needed_shards)}")
        print(f"   [WARN] Возможно, row_index не сохранен корректно.")

    return needed_shards


def download_and_extract_images(df, needed_shards):
    """
    Скачивает ZIP-шарды и извлекает нужные изображения.

    Args:
        df: DataFrame с метаданными
        needed_shards: Множество индексов шардов для скачивания

    Returns:
        int: Количество успешно скачанных изображений
    """
    images_dir = Path("dataset/raw/osv5m/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    # Множество ID которые нам нужны
    valid_ids = set(df["id"].astype(str))

    # Словарь для быстрого поиска города по ID
    id_to_city = dict(zip(df["id"].astype(str), df["city"]))

    downloaded_count = 0
    skipped_count = 0

    print(f"\n>>> Скачивание изображений из {len(needed_shards)} шардов...\n")

    for shard_idx in tqdm(sorted(needed_shards), desc="Шарды"):
        shard_file = f"{shard_idx:02d}.zip"

        try:
            # Скачиваем ZIP-шард
            zip_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"images/train/{shard_file}",
                repo_type="dataset",
                cache_dir=str(images_dir.parent / ".cache"),
                token=HF_TOKEN
            )

            # Извлекаем только нужные изображения
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.endswith('/'):
                        continue  # Пропускаем директории

                    # Получаем ID из имени файла
                    filename = os.path.basename(member)
                    img_id, ext = os.path.splitext(filename)

                    if img_id not in valid_ids:
                        continue  # Это изображение нам не нужно

                    # Определяем город и путь сохранения
                    city = id_to_city.get(img_id, "unknown")
                    city_dir = images_dir / city
                    city_dir.mkdir(exist_ok=True)

                    output_path = city_dir / filename

                    # Проверяем, не скачано ли уже
                    if output_path.exists() and output_path.stat().st_size > 1000:
                        skipped_count += 1
                        continue

                    # Извлекаем изображение
                    with zf.open(member) as source:
                        with open(output_path, "wb") as target:
                            shutil.copyfileobj(source, target)

                    downloaded_count += 1

        except Exception as e:
            print(f"\n[WARN] Ошибка обработки шарда {shard_idx}: {e}")
            continue

    print(f"\n[OK] Скачано: {downloaded_count} новых изображений")
    print(f"[SKIP] Пропущено (уже существуют): {skipped_count}")

    return downloaded_count


def generate_manifest(df):
    """
    Генерирует итоговый CSV-манифест для обучения.

    Args:
        df: DataFrame с метаданными

    Returns:
        Path: Путь к сгенерированному манифесту
    """
    images_dir = Path("dataset/raw/osv5m/images")
    manifest_rows = []

    print("\n>>> Генерация манифеста...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка записей"):
        img_id = str(row["id"])
        city = row["city"]
        lat = row.get("latitude", 0.0)
        lon = row.get("longitude", 0.0)

        # Путь к изображению
        img_path = images_dir / city / f"{img_id}.jpg"

        # Проверяем существование файла
        if not img_path.exists():
            continue

        # Относительный путь от корня dataset
        relative_path = img_path.relative_to(Path("dataset"))

        manifest_rows.append({
            "image_id": img_id,
            "filepath": str(relative_path).replace("\\", "/"),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "country": row.get("country", ""),
            "region": "",
            "city": city,
            "source": "osv5m",
            "capture_date": row.get("captured_at", ""),
            "quality_score": 1.0,
        })

    # Создаем DataFrame
    manifest_df = pd.DataFrame(manifest_rows)

    # Сохраняем
    output_path = Path("dataset/raw/osv5m/manifest.csv")
    manifest_df.to_csv(output_path, index=False, encoding="utf-8")

    # Статистика
    print(f"\n>>> Статистика манифеста:")
    print(manifest_df["city"].value_counts().to_string())
    print(f"\nИтого изображений: {len(manifest_df):,}")
    print(f"[OK] Манифест сохранен: {output_path.absolute()}")

    return output_path


def main():
    """
    Основная функция: скачивание изображений и генерация манифеста.
    """
    print("=" * 70)
    print("OSV-5M Image Download")
    print("=" * 70 + "\n")

    # 1. Загружаем отфильтрованные метаданные
    df = load_filtered_metadata()

    # 2. Определяем нужные шарды
    needed_shards = determine_needed_shards(df)

    # 3. Скачиваем и извлекаем изображения
    downloaded = download_and_extract_images(df, needed_shards)

    # 4. Генерируем итоговый манифест
    manifest_path = generate_manifest(df)

    # 5. Итоговая информация
    print("\n" + "=" * 70)
    print("[OK] СКАЧИВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"\n Изображения: dataset/raw/osv5m/images/")
    print(f" Манифест: {manifest_path}")
    print("\nСледующие шаги:")
    print("   1. python scripts/generate_manifests.py  # Создать train/val/test splits")
    print("   2. python code/train.py --config configs/baseline.yaml")


if __name__ == "__main__":
    main()
