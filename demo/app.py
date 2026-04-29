"""
demo/app.py — Gradio-демо для захисту диплому.

Завантажує зображення → передбачає місто + топ-5 класів → показує на карті.

Запуск:
    pip install gradio folium
    python demo/app.py --checkpoint checkpoints/best_model.pth

Примітка: потребує навченого чекпоінту. Якщо файл не знайдено, демо
запускається у «заглушковому» режимі, корисному для перевірки UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Додаємо code/ до sys.path, щоб імпортувати models/evaluate.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from augmentations import get_val_transforms  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402
from utils import get_device  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Глобальний стан моделі (ліниво завантажується)
# ──────────────────────────────────────────────────────────────────────────────

_MODEL: Optional[torch.nn.Module] = None
_CLASS_NAMES: list[str] = []
_TRANSFORM = None
_DEVICE: Optional[torch.device] = None
_CITY_CENTERS: dict[str, tuple[float, float]] = {
    # Fallback для карти, якщо метадані відсутні в чекпоінті.
    "Kyiv":     (50.4501, 30.5234),
    "Київ":     (50.4501, 30.5234),
    "Lviv":     (49.8397, 24.0297),
    "Львів":    (49.8397, 24.0297),
    "Odesa":    (46.4825, 30.7233),
    "Одеса":    (46.4825, 30.7233),
    "Kharkiv":  (49.9935, 36.2304),
    "Харків":   (49.9935, 36.2304),
    "Dnipro":   (48.4647, 35.0462),
    "Дніпро":   (48.4647, 35.0462),
    "Warsaw":   (52.2297, 21.0122),
    "Prague":   (50.0755, 14.4378),
    "Budapest": (47.4979, 19.0402),
}


def _init_model(checkpoint_path: str) -> None:
    """Ініціалізує модель з чекпоінту (викликається одноразово)."""
    global _MODEL, _CLASS_NAMES, _TRANSFORM, _DEVICE

    _DEVICE = get_device()
    if not Path(checkpoint_path).exists():
        print(f"[WARN] Чекпоінт не знайдено: {checkpoint_path}. Демо стартує в заглушковому режимі.")
        _CLASS_NAMES = list(_CITY_CENTERS.keys())[:8]
        _TRANSFORM = get_val_transforms(img_size=224)
        return

    model, class_names, _, _ = load_checkpoint(checkpoint_path, _DEVICE)
    model.eval()
    _MODEL = model
    _CLASS_NAMES = class_names
    _TRANSFORM = get_val_transforms(img_size=224)
    print(f"[OK] Модель завантажена: {len(class_names)} класів")


# ──────────────────────────────────────────────────────────────────────────────
# Функція передбачення
# ──────────────────────────────────────────────────────────────────────────────

def predict(image) -> tuple[dict, str]:
    """
    Аргументи:
        image: PIL.Image.Image з Gradio.

    Повертає:
        Кортеж (topk_labels, folium_html):
        - topk_labels: {class_name: probability} — для gr.Label
        - folium_html: HTML карти з маркерами топ-5 передбачень
    """
    if image is None:
        return {}, "<p>Завантажте зображення для передбачення.</p>"

    # ── Заглушковий режим без моделі ─────────────────────────────────────────
    if _MODEL is None:
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(len(_CLASS_NAMES)))
        top_idx = np.argsort(-probs)[:5]
        labels = {_CLASS_NAMES[i]: float(probs[i]) for i in top_idx}
        return labels, _build_map(labels)

    # ── Реальний інференс ────────────────────────────────────────────────────
    img_tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = _MODEL(img_tensor)
        if isinstance(logits, dict):
            logits = logits["logits"]
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_k = 5
    top_idx = np.argsort(-probs)[:top_k]
    labels = {_CLASS_NAMES[i]: float(probs[i]) for i in top_idx}
    return labels, _build_map(labels)


def _build_map(labels: dict) -> str:
    """Будує Folium-карту з маркерами топ-K міст."""
    try:
        import folium
    except ImportError:
        return "<p>Встановіть folium (<code>pip install folium</code>) для перегляду карти.</p>"

    # Центр карти — перше передбачене місто.
    first_city = next(iter(labels.keys()), None)
    center = _CITY_CENTERS.get(first_city, (49.0, 32.0)) if first_city else (49.0, 32.0)

    m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")

    for idx, (city, prob) in enumerate(labels.items()):
        coords = _CITY_CENTERS.get(city)
        if not coords:
            continue
        radius = 8 + 20 * prob
        color = "red" if idx == 0 else "blue"
        folium.CircleMarker(
            location=coords,
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.5,
            popup=f"{city}: {prob*100:.1f}%",
        ).add_to(m)

    return m._repr_html_()


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

def build_interface():
    import gradio as gr

    with gr.Blocks(title="AI_GeoDetect — Демо геолокації") as demo:
        gr.Markdown(
            "# AI_GeoDetect — демо\n"
            "Завантажте фото вулиці, щоб отримати топ-5 передбачень міста "
            "та візуалізацію ймовірностей на карті."
        )

        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Фото вулиці")
                btn = gr.Button("Передбачити", variant="primary")
            with gr.Column():
                lbl = gr.Label(num_top_classes=5, label="Топ-5 передбачень")
                mp  = gr.HTML(label="Карта")

        btn.click(fn=predict, inputs=inp, outputs=[lbl, mp])

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio-демо AI_GeoDetect")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                   help="Шлях до .pth чекпоінту (якщо файлу немає — заглушковий режим)")
    p.add_argument("--share", action="store_true",
                   help="Створити публічне посилання Gradio")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _init_model(args.checkpoint)
    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
