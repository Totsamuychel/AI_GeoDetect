"""
inference.py — Інференс для одного зображення з візуалізацією на карті.

Завантажує навчену модель, обробляє зображення та:
- Повертає топ-5 передбачень із ймовірностями
- Генерує інтерактивну Folium-карту з позначеними локаціями

Запуск:
    python inference.py --image path/to/photo.jpg \
                        --checkpoint checkpoints/best_model.pth \
                        --config configs/baseline.yaml \
                        --output results/prediction_map.html
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from augmentations import get_val_transforms
from models import build_model
from utils import get_device, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Координати центрів міст України (довідник)
# ──────────────────────────────────────────────────────────────────────────────

UKRAINE_CITY_CENTERS: dict[str, tuple[float, float]] = {
    "Київ":        (50.4501, 30.5234),
    "Харків":      (49.9935, 36.2304),
    "Одеса":       (46.4825, 30.7233),
    "Дніпро":      (48.4647, 35.0462),
    "Запоріжжя":   (47.8388, 35.1396),
    "Львів":       (49.8397, 24.0297),
    "Кривий Ріг":  (47.9078, 33.3845),
    "Миколаїв":    (46.9750, 31.9946),
    "Вінниця":     (49.2330, 28.4682),
    "Херсон":      (46.6354, 32.6169),
    "Полтава":     (49.5883, 34.5514),
    "Черкаси":     (49.4444, 32.0598),
    "Хмельницький":(49.4216, 26.9870),
    "Чернівці":    (48.2921, 25.9354),
    "Суми":        (50.9116, 34.7981),
    "Житомир":     (50.2547, 28.6587),
    "Рівне":       (50.6197, 26.2519),
    "Івано-Франківськ": (48.9226, 24.7111),
    "Тернопіль":   (49.5535, 25.5948),
    "Луцьк":       (50.7472, 25.3254),
    "Ужгород":     (48.6208, 22.2879),
    "Чернігів":    (51.4982, 31.2893),
    "Кропивницький":(48.5079, 32.2623),
    "Маріуполь":   (47.0960, 37.5424),
    "Запоріжжя":   (47.8388, 35.1396),
}


# ──────────────────────────────────────────────────────────────────────────────
# Клас інференсу
# ──────────────────────────────────────────────────────────────────────────────

class GeoLocator:
    """
    Геолокатор зображень на основі навченої нейронної мережі.

    Завантажує модель із чекпоінту та надає методи для:
    - Передбачення міста/регіону для зображення
    - Генерації інтерактивної карти з результатами

    Аргументи:
        checkpoint_path: Шлях до .pth файлу чекпоінту.
        device:          Пристрій для інференсу (None = автоматично).
        img_size:        Розмір вхідного зображення.
        city_coords:     Словник {назва_міста: (lat, lon)}.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        img_size: int = 224,
        city_coords: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        self.device   = device or get_device()
        self.img_size = img_size
        self.transform = get_val_transforms(img_size=img_size)
        self.city_coords = city_coords or UKRAINE_CITY_CENTERS.copy()

        self.model, self.class_names, self.config = self._load_model(checkpoint_path)
        self.num_classes = len(self.class_names)

        # Таблиця координат міст для передбачень
        self._class_centers = self._build_class_centers()

        logger.info(
            f"GeoLocator ініціалізовано: {self.num_classes} класів, "
            f"arch={self.config.get('architecture', 'baseline')}"
        )

    def _load_model(
        self, checkpoint_path: Union[str, Path]
    ) -> tuple[torch.nn.Module, list[str], dict]:
        """Завантажує модель із чекпоінту."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Чекпоінт не знайдено: {path}")

        checkpoint  = torch.load(str(path), map_location=self.device, weights_only=False)
        config      = checkpoint.get("config", {})
        class_names = checkpoint.get("class_names", [])

        if not class_names:
            raise ValueError("Чекпоінт не містить class_names")

        architecture = config.get("architecture", "baseline")
        model = build_model(
            architecture=architecture,
            num_classes=len(class_names),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(self.device)
        model.eval()

        logger.info(
            f"Модель завантажено: {path.name} "
            f"(epoch={checkpoint.get('epoch', '?')}, "
            f"val_acc={checkpoint.get('val_acc', float('nan')):.4f})"
        )
        return model, class_names, config

    def _build_class_centers(self) -> dict[str, tuple[float, float]]:
        """Будує словник координат центрів для класів моделі."""
        centers: dict[str, tuple[float, float]] = {}
        for city in self.class_names:
            if city in self.city_coords:
                centers[city] = self.city_coords[city]
            else:
                # Координати невідомого міста — центр України
                logger.warning(f"Координати міста не знайдено: '{city}'. Використовуємо (49.0, 32.0)")
                centers[city] = (49.0, 32.0)
        return centers

    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Завантажує та передобробляє зображення.

        Аргументи:
            image_path: Шлях до файлу зображення.

        Повертає:
            Тензор (1, 3, H, W), готовий для моделі.
        """
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Зображення не знайдено: {path}")

        img = Image.open(str(path)).convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, torch.Tensor],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Передбачає геолокацію зображення.

        Аргументи:
            image_input: Шлях до зображення або тензор (1, 3, H, W).
            top_k:       Кількість топ-передбачень.

        Повертає:
            Список словників, відсортованих за ймовірністю:
            [
                {
                    "rank":  int,
                    "city":  str,
                    "prob":  float,
                    "lat":   float,
                    "lon":   float,
                },
                ...
            ]
        """
        if isinstance(image_input, (str, Path)):
            tensor = self.load_image(image_input)
        else:
            tensor = image_input.to(self.device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

        architecture = self.config.get("architecture", "baseline")
        if architecture == "geoclip":
            output = self.model(tensor)
            logits = output["logits"]
        else:
            logits = self.model(tensor)

        probs = F.softmax(logits, dim=1)[0]  # (C,)
        top_probs, top_indices = probs.topk(min(top_k, self.num_classes))

        predictions = []
        for rank, (prob, idx) in enumerate(
            zip(top_probs.tolist(), top_indices.tolist()), start=1
        ):
            city = self.class_names[idx]
            lat, lon = self._class_centers.get(city, (49.0, 32.0))
            predictions.append({
                "rank": rank,
                "city": city,
                "prob": round(prob, 6),
                "lat":  lat,
                "lon":  lon,
            })

        return predictions

    def predict_batch(
        self,
        image_paths: list[Union[str, Path]],
        top_k: int = 1,
    ) -> list[list[dict]]:
        """
        Пакетне передбачення для кількох зображень.

        Аргументи:
            image_paths: Список шляхів до зображень.
            top_k:       Кількість топ-передбачень для кожного.

        Повертає:
            Список результатів для кожного зображення.
        """
        results = []
        for path in image_paths:
            try:
                preds = self.predict(path, top_k=top_k)
                results.append(preds)
            except Exception as e:
                logger.error(f"Помилка інференсу для {path}: {e}")
                results.append([])
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Генерація Folium-карти
# ──────────────────────────────────────────────────────────────────────────────

def generate_folium_map(
    predictions: list[dict],
    true_location: Optional[dict] = None,
    output_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> "folium.Map":
    """
    Генерує інтерактивну Folium-карту з передбаченими локаціями.

    Аргументи:
        predictions:   Список передбачень від GeoLocator.predict().
        true_location: Словник {'city': str, 'lat': float, 'lon': float}
                       із реальним місцезнаходженням (необов'язково).
        output_path:   Шлях для збереження HTML-файлу карти.
        image_path:    Шлях до зображення (для відображення у popup).

    Повертає:
        folium.Map із позначками.
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        raise ImportError("Встановіть folium: pip install folium")

    if not predictions:
        raise ValueError("Список передбачень порожній")

    # Центруємо карту на топ-1 передбаченні
    top1 = predictions[0]
    m = folium.Map(
        location=[top1["lat"], top1["lon"]],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    # Кольорова шкала ймовірностей: від зеленого (топ-1) до блідо-синього
    marker_colors = ["green", "darkgreen", "blue", "cadetblue", "lightblue"]

    for pred in predictions:
        rank  = pred["rank"]
        city  = pred["city"]
        prob  = pred["prob"]
        lat   = pred["lat"]
        lon   = pred["lon"]
        color = marker_colors[rank - 1] if rank <= len(marker_colors) else "gray"

        # Розмір кружка пропорційний ймовірності
        radius = max(8, int(prob * 60))

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 150px;">
            <b>#{rank} {city}</b><br/>
            Ймовірність: <b>{prob*100:.1f}%</b><br/>
            Координати: {lat:.4f}°N, {lon:.4f}°E
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{rank} {city}: {prob*100:.1f}%",
        ).add_to(m)

        # Додаємо номер поряд із маркером
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px; font-weight:bold; '
                     f'color:white; background:{color}; padding:2px 5px; '
                     f'border-radius:3px; opacity:0.9;">#{rank}</div>',
                icon_size=(30, 20),
                icon_anchor=(15, 10),
            ),
        ).add_to(m)

    # Реальне місцезнаходження (зірочка червоного кольору)
    if true_location:
        t_lat  = true_location.get("lat", 0.0)
        t_lon  = true_location.get("lon", 0.0)
        t_city = true_location.get("city", "Реальне місце")

        popup_html = f"""
        <div style="font-family: Arial, sans-serif;">
            <b>✓ Реальне місце</b><br/>
            <b>{t_city}</b><br/>
            Координати: {t_lat:.4f}°N, {t_lon:.4f}°E
        </div>
        """

        folium.Marker(
            location=[t_lat, t_lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Реальне місце: {t_city}",
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(m)

        # Лінія від топ-1 передбачення до реального місця
        folium.PolyLine(
            locations=[[top1["lat"], top1["lon"]], [t_lat, t_lon]],
            color="red",
            weight=2,
            opacity=0.6,
            dash_array="8",
            tooltip="Відстань: топ-1 → реальне",
        ).add_to(m)

    # Легенда
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px; border-radius: 5px;
                border: 1px solid #ccc; font-family: Arial, font-size: 12px;">
        <b>Легенда</b><br/>
        <span style="color:green">●</span> #1 передбачення<br/>
        <span style="color:darkgreen">●</span> #2 передбачення<br/>
        <span style="color:blue">●</span> #3 передбачення<br/>
        <span style="color:red">★</span> Реальне місце
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        logger.info(f"Карта збережена: {output_path}")

    return m


# ──────────────────────────────────────────────────────────────────────────────
# Форматування результатів
# ──────────────────────────────────────────────────────────────────────────────

def format_predictions(
    predictions: list[dict],
    true_location: Optional[dict] = None,
) -> str:
    """
    Форматує передбачення для виводу в консоль.

    Аргументи:
        predictions:   Список передбачень.
        true_location: Реальне місцезнаходження (необов'язково).

    Повертає:
        Відформатований рядок.
    """
    lines = ["", "=" * 50, "РЕЗУЛЬТАТИ ГЕОЛОКАЦІЇ", "=" * 50]

    for pred in predictions:
        bar_len = int(pred["prob"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(
            f"  #{pred['rank']:1d} [{bar}] {pred['prob']*100:5.1f}%  "
            f"{pred['city']:20s}  ({pred['lat']:.4f}°N, {pred['lon']:.4f}°E)"
        )

    if true_location and predictions:
        from metrics import haversine_distance
        top1 = predictions[0]
        dist = haversine_distance(
            top1["lat"], top1["lon"],
            true_location.get("lat", 0.0), true_location.get("lon", 0.0),
        )
        lines.append("")
        lines.append(f"  Реальне місце: {true_location.get('city', 'N/A')}")
        lines.append(f"  Відстань від топ-1: {dist:.1f} км")

    lines.append("=" * 50)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Геолокація вуличного зображення за допомогою нейронної мережі",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",       type=str, required=True,
                        help="Шлях до зображення для передбачення")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Шлях до .pth чекпоінту")
    parser.add_argument("--config",      type=str, default=None,
                        help="Шлях до YAML/JSON конфіг-файлу (необов'язково)")
    parser.add_argument("--output",      type=str, default="results/prediction_map.html",
                        help="Шлях для збереження HTML-карти")
    parser.add_argument("--top-k",       type=int, default=5,
                        help="Кількість топ-передбачень")
    parser.add_argument("--img-size",    type=int, default=224,
                        help="Розмір вхідного зображення")
    parser.add_argument("--true-lat",    type=float, default=None,
                        help="Реальна широта (для оцінки точності)")
    parser.add_argument("--true-lon",    type=float, default=None,
                        help="Реальна довгота")
    parser.add_argument("--true-city",   type=str, default=None,
                        help="Реальна назва міста")
    parser.add_argument("--save-json",   type=str, default=None,
                        help="Шлях для збереження JSON з передбаченнями")
    parser.add_argument("--no-map",      action="store_true",
                        help="Не генерувати Folium-карту")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ініціалізація геолокатора
    locator = GeoLocator(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
    )

    # Передбачення
    predictions = locator.predict(args.image, top_k=args.top_k)

    # Реальне місцезнаходження (якщо задано)
    true_location = None
    if args.true_lat is not None and args.true_lon is not None:
        true_location = {
            "lat":  args.true_lat,
            "lon":  args.true_lon,
            "city": args.true_city or "Задане місце",
        }

    # Вивід результатів
    print(format_predictions(predictions, true_location))

    # Збереження JSON
    if args.save_json:
        output = {
            "image_path":  str(args.image),
            "checkpoint":  str(args.checkpoint),
            "predictions": predictions,
            "true_location": true_location,
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON збережено: {args.save_json}")

    # Генерація карти
    if not args.no_map:
        try:
            generate_folium_map(
                predictions=predictions,
                true_location=true_location,
                output_path=args.output,
                image_path=args.image,
            )
            print(f"\nКарта збережена: {args.output}")
        except ImportError:
            logger.warning("folium не встановлено. Пропускаємо генерацію карти.")
            logger.warning("Встановіть: pip install folium")


if __name__ == "__main__":
    main()
