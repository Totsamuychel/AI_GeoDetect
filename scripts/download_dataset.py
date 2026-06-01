"""
download_dataset.py — Master-скрипт для скачивания датасета OSV-5M.

Запускает весь процесс скачивания последовательно:
1. Получение полигонов городов через OSM
2. Фильтрация метаданных OSV-5M по полигонам
3. Скачивание изображений

Требования:
    - HF_TOKEN в переменных окружения
    - Установленные зависимости: osmnx, shapely, huggingface_hub

Запуск:
    python scripts/download_dataset.py [--skip-polygons] [--skip-filter]
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
        print(f"❌ Скрипт не найден: {script_path}")
        sys.exit(1)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            text=True
        )

        print(f"\n✅ Этап завершен успешно: {description}")
        return result

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка на этапе: {description}")
        print(f"Код возврата: {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Master-скрипт для скачивания датасета OSV-5M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

    # Полный процесс (все этапы)
    python scripts/download_dataset.py

    # Пропустить получение полигонов (если уже есть)
    python scripts/download_dataset.py --skip-polygons

    # Пропустить фильтрацию метаданных (если уже выполнена)
    python scripts/download_dataset.py --skip-polygons --skip-filter
        """
    )

    parser.add_argument(
        "--skip-polygons",
        action="store_true",
        help="Пропустить получение полигонов городов (если уже выполнено)"
    )

    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Пропустить фильтрацию метаданных (если уже выполнена)"
    )

    args = parser.parse_args()

    print("\n" + "🌍 " * 20)
    print(" " * 10 + "OSV-5M Dataset Download Pipeline")
    print("🌍 " * 20 + "\n")

    print("Целевые города: Kyiv, Warsaw, Prague")
    print("Целевое количество: ~2000-2500 фото на город")
    print("Фильтрация: только городская застройка (без трасс и пригородов)\n")

    # Этап 1: Получение полигонов
    if not args.skip_polygons:
        run_step("01_get_city_polygons.py", "Получение полигонов городов")
    else:
        print("\n⏭️ Пропускаем этап 1: полигоны уже получены")

    # Этап 2: Фильтрация метаданных
    if not args.skip_filter:
        run_step("02_filter_osv5m.py", "Фильтрация метаданных OSV-5M")
    else:
        print("\n⏭️ Пропускаем этап 2: метаданные уже отфильтрованы")

    # Этап 3: Скачивание изображений
    run_step("03_download_osv5m_images.py", "Скачивание изображений")

    # Финальное сообщение
    print("\n" + "=" * 70)
    print("🎉 ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ УСПЕШНО!")
    print("=" * 70)

    print("\nРезультаты:")
    print("  📁 Изображения: dataset/raw/osv5m/images/")
    print("  📄 Манифест: dataset/raw/osv5m/manifest.csv")
    print("  📍 Полигоны: dataset/raw/boundaries/")

    print("\nСледующие шаги:")
    print("  1. Проверьте количество изображений по городам")
    print("  2. Запустите: python scripts/generate_manifests.py")
    print("  3. Начните обучение: python code/train.py --config configs/baseline.yaml")


if __name__ == "__main__":
    main()
