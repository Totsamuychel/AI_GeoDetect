"""
01_get_city_polygons.py — Получение точных границ городов через OpenStreetMap.

Использует osmnx для получения реальных административных границ городов,
что позволяет исключить пригороды, трассы и сельские территории.

Запуск:
    python scripts/01_get_city_polygons.py
"""

import osmnx as ox
import json
import os
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

CITIES = {
    "kyiv":   "Kyiv, Ukraine",
    "warsaw": "Warsaw, Poland",
    "prague": "Prague, Czech Republic",
}

def get_city_polygons():
    """
    Получает полигоны городов через OpenStreetMap и сохраняет в GeoJSON.
    """
    output_dir = Path("dataset/raw/boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Получение полигонов городов через OpenStreetMap\n")

    for city_key, city_name in CITIES.items():
        print(f"[*] Обрабатываем: {city_name}")

        try:
            # Получаем GeoDataFrame с границами города
            gdf = ox.geocode_to_gdf(city_name)

            # Конвертируем в GeoJSON
            geojson = json.loads(gdf.to_json())

            # Сохраняем
            output_path = output_dir / f"{city_key}.geojson"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson, f, indent=2, ensure_ascii=False)

            # Выводим bounding box для справки
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            print(f"    Bounding box:")
            print(f"      Latitude:  [{bounds[1]:.4f}, {bounds[3]:.4f}]")
            print(f"      Longitude: [{bounds[0]:.4f}, {bounds[2]:.4f}]")
            print(f"    [OK] Сохранено: {output_path}\n")

        except Exception as e:
            print(f"    [ERROR] Ошибка для {city_name}: {e}\n")
            continue

    print("\n[SUCCESS] Все полигоны успешно получены!")
    print(f"Результаты сохранены в: {output_dir.absolute()}")


if __name__ == "__main__":
    get_city_polygons()
