"""
download_dataset_simple.py — Упрощенный master-скрипт для скачивания датасета.

Скачивает данные ТОЛЬКО из Global Streetscapes (OSV-5M пропущен из-за плохого покрытия).

Этапы:
1. Получение полигонов городов (если еще не сделано)
2. Скачивание из Global Streetscapes
3. Создание итогового манифеста

Запуск:
    python scripts/download_dataset_simple.py [--skip-polygons]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_name, description):
    """
    Запускает Python скрипт и обрабатывает ошибки.

    Args:
        script_name: Имя скрипта для запуска
        description: Описание этапа
    """
    print("\n" + "=" * 70)
    print(f"ЭТАП: {description}")
    print("=" * 70 + "\n")

    script_path = Path("scripts") / script_name

    if not script_path.exists():
        print(f"[ERROR] Скрипт не найден: {script_path}")
        sys.exit(1)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            text=True
        )

        print(f"\n[OK] Этап завершен успешно: {description}")
        return result

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Ошибка на этапе: {description}")
        print(f"Код возврата: {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Упрощенный master-скрипт для скачивания датасета (только Global Streetscapes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

    # Полный процесс (все этапы)
    python scripts/download_dataset_simple.py

    # Пропустить получение полигонов (если уже есть)
    python scripts/download_dataset_simple.py --skip-polygons
        """
    )

    parser.add_argument(
        "--skip-polygons",
        action="store_true",
        help="Пропустить получение полигонов городов (если уже выполнено)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 15 + "Dataset Download Pipeline")
    print(" " * 20 + "(Global Streetscapes)")
    print("=" * 70 + "\n")

    print("Целевые города: Kyiv, Warsaw, Prague")
    print("Целевое количество: ~2500 фото на город")
    print("Источник: Global Streetscapes (NUS-UAL/global-streetscapes)")
    print("\nПримечание: OSV-5M пропущен из-за плохого покрытия для Восточной Европы\n")

    # Этап 1: Получение полигонов
    if not args.skip_polygons:
        run_step("01_get_city_polygons.py", "Получение полигонов городов")
    else:
        print("\n[SKIP] Пропускаем этап 1: полигоны уже получены")

    # Этап 2: Скачивание из Global Streetscapes
    run_step("04_download_global_streetscapes.py", "Скачивание Global Streetscapes")

    # Этап 3: Создание итогового манифеста (если нужно объединить с Mapillary)
    print("\n" + "=" * 70)
    print("ОПЦИОНАЛЬНО: Объединение с Mapillary")
    print("=" * 70 + "\n")

    mapillary_path = Path("dataset/raw/mapillary/manifest.csv")
    if mapillary_path.exists():
        print("Найден манифест Mapillary. Объединяем...")
        run_step("05_merge_manifests.py", "Объединение манифестов")
    else:
        print("Mapillary не найден. Используем только Global Streetscapes.")
        print("\nЕсли хотите добавить данные из Mapillary:")
        print("   python code/download_data.py mapillary ...")
        print("   python scripts/05_merge_manifests.py")

    # Финальное сообщение
    print("\n" + "=" * 70)
    print("[SUCCESS] ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ!")
    print("=" * 70)

    print("\nРезультаты:")
    print("   Изображения: dataset/raw/global_streetscapes/images/")
    print("   Манифест: dataset/raw/global_streetscapes/manifest.csv")

    if mapillary_path.exists():
        print("   Объединенный манифест: dataset/raw/merged_manifest.csv")

    print("\nСледующие шаги:")
    print("  1. Проверьте количество изображений по городам")
    print("  2. Создайте train/val/test splits:")
    print("     python scripts/generate_manifests.py \\")

    if mapillary_path.exists():
        print("         --input dataset/raw/merged_manifest.csv \\")
    else:
        print("         --input dataset/raw/global_streetscapes/manifest.csv \\")

    print("         --output-dir dataset/manifests")
    print("  3. Начните обучение:")
    print("     python code/train.py --config configs/baseline.yaml")


if __name__ == "__main__":
    main()
