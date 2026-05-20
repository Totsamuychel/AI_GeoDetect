"""
05_merge_manifests.py — Объединение манифестов из разных источников.

Объединяет манифесты из:
    - OSV-5M (dataset/raw/osv5m/manifest.csv)
    - Global Streetscapes (dataset/raw/global_streetscapes/manifest.csv)
    - Mapillary (если есть)

Создает итоговый объединенный манифест для обучения.

Запуск:
    python scripts/05_merge_manifests.py
"""

import sys
from pathlib import Path
import pandas as pd

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_manifest(path: Path, source_name: str) -> pd.DataFrame:
    """
    Загружает манифест и выводит статистику.

    Args:
        path: Путь к CSV файлу
        source_name: Название источника для вывода

    Returns:
        DataFrame или None если файл не найден
    """
    if not path.exists():
        print(f"   [SKIP] {source_name}: файл не найден")
        return None

    df = pd.read_csv(path)
    print(f"   [OK] {source_name}: {len(df):,} изображений")

    if 'city' in df.columns:
        print(f"        По городам:")
        for city, count in df['city'].value_counts().items():
            print(f"          {city}: {count:,}")

    return df


def merge_manifests():
    """
    Объединяет манифесты из всех источников.
    """
    print("=" * 70)
    print("Merging Dataset Manifests")
    print("=" * 70 + "\n")

    manifests = []

    # 1. Global Streetscapes (основной источник)
    print(">>> Загрузка Global Streetscapes...")
    gss_path = Path("dataset/raw/global_streetscapes/manifest.csv")
    gss_df = load_manifest(gss_path, "Global Streetscapes")
    if gss_df is not None:
        manifests.append(gss_df)

    # 2. Mapillary (опционально, для дополнения)
    print("\n>>> Загрузка Mapillary...")
    mapillary_path = Path("dataset/raw/mapillary/manifest.csv")
    mapillary_df = load_manifest(mapillary_path, "Mapillary")
    if mapillary_df is not None:
        manifests.append(mapillary_df)

    # Проверка
    if not manifests:
        print("\n[ERROR] Не найдено ни одного манифеста!")
        print("\nЗапустите сначала:")
        print("   python scripts/04_download_global_streetscapes.py")
        return

    # Объединение
    print("\n>>> Объединение манифестов...")
    merged_df = pd.concat(manifests, ignore_index=True)

    # Удаление дубликатов по координатам (с точностью до 10 метров)
    print(f"\n   Всего записей до дедупликации: {len(merged_df):,}")

    # Округляем координаты до 4 знаков (~10 метров)
    merged_df['lat_rounded'] = merged_df['lat'].round(4)
    merged_df['lon_rounded'] = merged_df['lon'].round(4)

    # Удаляем дубликаты
    merged_df = merged_df.drop_duplicates(
        subset=['lat_rounded', 'lon_rounded', 'city'],
        keep='first'
    )

    # Удаляем вспомогательные колонки
    merged_df = merged_df.drop(columns=['lat_rounded', 'lon_rounded'])

    print(f"   После дедупликации: {len(merged_df):,}")

    # Статистика
    print("\n" + "=" * 70)
    print(">>> ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)

    print(f"\nВсего изображений: {len(merged_df):,}")

    print(f"\nПо городам:")
    for city, count in merged_df['city'].value_counts().items():
        print(f"   {city.capitalize():10s}: {count:,} фото")

    print(f"\nПо источникам:")
    for source, count in merged_df['source'].value_counts().items():
        print(f"   {source:20s}: {count:,} фото")

    # Сохранение
    output_path = Path("dataset/raw/merged_manifest.csv")
    merged_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n[OK] Объединенный манифест сохранен: {output_path.absolute()}")

    # Проверка достаточности данных
    print("\n" + "=" * 70)
    print(">>> ПРОВЕРКА ДАННЫХ")
    print("=" * 70)

    target_per_city = 2000
    for city in ['warsaw', 'prague', 'budapest']:
        count = len(merged_df[merged_df['city'] == city])
        status = "[OK]" if count >= target_per_city else "[WARN]"
        percentage = (count / target_per_city * 100) if target_per_city > 0 else 0
        print(f"{status} {city.capitalize():10s}: {count:,} / {target_per_city:,} ({percentage:.1f}%)")

    print("\nСледующие шаги:")
    print("   1. Если данных недостаточно, скачайте дополнительно:")
    print("      - Mapillary: python code/download_data.py mapillary ...")
    print("   2. Создайте train/val/test splits:")
    print("      python scripts/generate_manifests.py \\")
    print("          --input dataset/raw/merged_manifest.csv \\")
    print("          --output-dir dataset/manifests")


if __name__ == "__main__":
    merge_manifests()
