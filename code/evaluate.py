"""
evaluate.py — Оцінка навченої моделі на тестовому наборі даних.

Обчислює:
- Top-1 та Top-5 точність (classification accuracy)
- Середню та медіанну відстань Гаверсина (км)
- GeoScore (5000 * exp(-d / 1492.7))
- Точність по класах (per-class accuracy table)

Результати зберігаються у JSON-файл.

Запуск:
    python evaluate.py --checkpoint checkpoints/best_model.pth \
                       --manifest data/manifest.csv \
                       --output results/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmentations import get_val_transforms
from dataset import GeoDataset
from metrics import (
    compute_all_metrics,
    geoscore,
    haversine_distance,
    top_k_accuracy,
)
from models import build_model
from utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Завантаження чекпоінту
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str], dict, dict]:
    """
    Завантажує модель із чекпоінту.

    Аргументи:
        checkpoint_path: Шлях до .pth файлу.
        device:          Пристрій для завантаження.

    Повертає:
        Кортеж (model, class_names, config, checkpoint_meta), де
        checkpoint_meta — словник з полями 'epoch', 'val_loss', 'val_acc'.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Чекпоінт не знайдено: {path}")

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    config     = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names)
    architecture = config.get("architecture", "baseline")

    if num_classes == 0:
        raise ValueError("Чекпоінт не містить class_names")

    logger.info(
        f"Завантаження чекпоінту: {path.name} | "
        f"arch={architecture}, epoch={checkpoint.get('epoch', '?')}, "
        f"val_loss={checkpoint.get('val_loss', float('nan')):.4f}, "
        f"val_acc={checkpoint.get('val_acc', float('nan')):.4f}"
    )

    model = build_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    # Повертаємо легкі метадані (без model_state) щоб уникнути повторного
    # завантаження чекпоінту з диску наприкінці evaluate().
    meta = {
        "epoch":    checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "val_acc":  checkpoint.get("val_acc"),
    }

    return model, class_names, config, meta


# ──────────────────────────────────────────────────────────────────────────────
# Основна функція оцінки
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Результати оцінки моделі."""
    checkpoint_path:    str
    architecture:       str
    num_classes:        int
    num_test_samples:   int

    # Класифікація
    top1_accuracy:      float
    top5_accuracy:      float

    # Геолокація
    mean_distance_km:   float
    median_distance_km: float
    p25_distance_km:    float
    p75_distance_km:    float
    p90_distance_km:    float
    fraction_within_25km:  float
    fraction_within_200km: float
    fraction_within_750km: float

    # GeoScore
    mean_geoscore:      float
    median_geoscore:    float

    # Per-class
    per_class_accuracy: dict

    # Метаінформація
    val_loss_from_ckpt: Optional[float] = None
    val_acc_from_ckpt:  Optional[float] = None


def evaluate(
    checkpoint_path: str,
    manifest_path: str,
    image_root: Optional[str] = None,
    countries: Optional[list[str]] = None,
    quality_threshold: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 4,
    output_path: Optional[str] = None,
    use_test_split: bool = True,
    img_size: int = 224,
    split_method: str = "h3",
    seed: int = 42,
) -> EvalResult:
    """
    Оцінює модель на тестовому наборі та зберігає результати.

    Аргументи:
        checkpoint_path:   Шлях до чекпоінту моделі.
        manifest_path:     Шлях до CSV-маніфесту.
        image_root:        Корінь шляхів до зображень.
        countries:         Фільтр за країнами.
        quality_threshold: Мінімальний поріг якості.
        batch_size:        Розмір батчу.
        num_workers:       Кількість воркерів DataLoader.
        output_path:       Шлях для збереження JSON з результатами.
        use_test_split:    Якщо True, оцінює на test split; інакше — на всіх даних.
        img_size:          Розмір зображень.
        split_method:      Метод розбиття ('h3' або 'kmeans').
        seed:              Seed для відтворюваності розбиття.

    Повертає:
        EvalResult із всіма метриками.
    """
    device = get_device()

    # ── Завантаження моделі ───────────────────────────────────────────────────
    model, class_names, config, ckpt_meta = load_checkpoint(checkpoint_path, device)
    architecture = config.get("architecture", "baseline")
    num_classes  = len(class_names)

    # ── Датасет ───────────────────────────────────────────────────────────────
    val_transform = get_val_transforms(img_size=img_size)

    full_dataset = GeoDataset(
        manifest_path=manifest_path,
        transform=None,
        countries=countries,
        quality_threshold=quality_threshold,
        image_root=image_root,
    )
    # Синхронізація класів
    full_dataset.class_names  = class_names
    full_dataset._city_to_idx = {c: i for i, c in enumerate(class_names)}
    full_dataset.num_classes  = num_classes

    if use_test_split:
        _, _, test_idx = full_dataset.get_split_indices(
            method=split_method,
            seed=seed,
        )
    else:
        test_idx = full_dataset.df.index

    test_dataset = GeoDataset(
        manifest_path=manifest_path,
        transform=val_transform,
        countries=countries,
        quality_threshold=quality_threshold,
        image_root=image_root,
        subset_indices=test_idx,
    )
    test_dataset.class_names  = class_names
    test_dataset._city_to_idx = {c: i for i, c in enumerate(class_names)}
    test_dataset.num_classes  = num_classes

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Тестовий набір: {len(test_dataset)} зразків, {num_classes} класів")

    # ── Інференс ─────────────────────────────────────────────────────────────
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_pred_coords: list[np.ndarray] = []
    all_true_coords: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels, coords in test_loader:
            images = images.to(device, non_blocking=True)

            if architecture == "geoclip":
                output = model(images)
                logits = output["logits"]
            else:
                logits = model(images)

            # Передбачені координати = центр передбаченого міста
            pred_indices = logits.argmax(dim=1).cpu().numpy()
            pred_coords  = _indices_to_coords(pred_indices, class_names, test_dataset)
            # coords — тензор форми (N, 2) у форматі [lat, lon]
            true_coords = coords.cpu().numpy().astype(np.float64)

            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_pred_coords.append(pred_coords)
            all_true_coords.append(true_coords)

    all_logits     = torch.cat(all_logits, dim=0)
    all_labels     = torch.cat(all_labels, dim=0)
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    all_true_coords = np.concatenate(all_true_coords, axis=0)

    # ── Метрики ───────────────────────────────────────────────────────────────
    logger.info("Обчислення метрик...")

    top1 = top_k_accuracy(all_logits, all_labels, k=1)
    top5 = top_k_accuracy(all_logits, all_labels, k=min(5, num_classes))

    distances = haversine_distance(
        all_pred_coords[:, 0], all_pred_coords[:, 1],
        all_true_coords[:, 0], all_true_coords[:, 1],
    )
    scores = geoscore(distances)

    mean_dist   = float(np.mean(distances))
    median_dist = float(np.median(distances))
    p25_dist    = float(np.percentile(distances, 25))
    p75_dist    = float(np.percentile(distances, 75))
    p90_dist    = float(np.percentile(distances, 90))

    frac_25  = float(np.mean(distances <= 25))
    frac_200 = float(np.mean(distances <= 200))
    frac_750 = float(np.mean(distances <= 750))

    mean_score   = float(np.mean(scores))
    median_score = float(np.median(scores))

    # ── Per-class точність ────────────────────────────────────────────────────
    per_class_acc = _compute_per_class_accuracy(
        all_logits.numpy(), all_labels.numpy(), class_names
    )

    logger.info(
        f"\n{'='*55}\n"
        f"РЕЗУЛЬТАТИ ОЦІНКИ\n"
        f"{'='*55}\n"
        f"  Тестовий набір:         {len(test_dataset)} зображень\n"
        f"  Top-1 Accuracy:         {top1*100:.2f}%\n"
        f"  Top-5 Accuracy:         {top5*100:.2f}%\n"
        f"  Mean Distance:          {mean_dist:.1f} км\n"
        f"  Median Distance:        {median_dist:.1f} км\n"
        f"  90-й перцентиль:        {p90_dist:.1f} км\n"
        f"  Частка < 25 км:         {frac_25*100:.1f}%\n"
        f"  Частка < 200 км:        {frac_200*100:.1f}%\n"
        f"  Частка < 750 км:        {frac_750*100:.1f}%\n"
        f"  Mean GeoScore:          {mean_score:.1f} / 5000\n"
        f"  Median GeoScore:        {median_score:.1f} / 5000\n"
        f"{'='*55}"
    )

    # Per-class таблиця
    logger.info("\nТочність по класах:")
    per_class_df = pd.DataFrame([
        {"Місто": city, "Accuracy": f"{acc*100:.1f}%", "Count": count}
        for city, (acc, count) in sorted(per_class_acc.items(), key=lambda x: -x[1][0])
    ])
    logger.info("\n" + per_class_df.to_string(index=False))

    # ── Збереження результатів ────────────────────────────────────────────────
    result = EvalResult(
        checkpoint_path=str(checkpoint_path),
        architecture=architecture,
        num_classes=num_classes,
        num_test_samples=len(test_dataset),
        top1_accuracy=top1,
        top5_accuracy=top5,
        mean_distance_km=mean_dist,
        median_distance_km=median_dist,
        p25_distance_km=p25_dist,
        p75_distance_km=p75_dist,
        p90_distance_km=p90_dist,
        fraction_within_25km=frac_25,
        fraction_within_200km=frac_200,
        fraction_within_750km=frac_750,
        mean_geoscore=mean_score,
        median_geoscore=median_score,
        per_class_accuracy={k: {"accuracy": v[0], "count": v[1]} for k, v in per_class_acc.items()},
        val_loss_from_ckpt=ckpt_meta.get("val_loss"),
        val_acc_from_ckpt=ckpt_meta.get("val_acc"),
    )

    if output_path:
        _save_results(result, output_path)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Допоміжні функції
# ──────────────────────────────────────────────────────────────────────────────

def _indices_to_coords(
    indices: np.ndarray,
    class_names: list[str],
    dataset: GeoDataset,
) -> np.ndarray:
    """
    Конвертує індекси класів у середні координати відповідних міст.

    Аргументи:
        indices:     Масив індексів класів (N,).
        class_names: Список назв класів.
        dataset:     GeoDataset для обчислення середніх координат.

    Повертає:
        Масив координат (N, 2) у форматі (lat, lon).
    """
    # Передобчислюємо середні координати для кожного міста
    city_centers: dict[str, tuple[float, float]] = {}
    if "city" in dataset.df.columns:
        for city in class_names:
            city_df = dataset.df[dataset.df["city"] == city]
            if len(city_df) > 0:
                city_centers[city] = (
                    float(city_df["lat"].mean()),
                    float(city_df["lon"].mean()),
                )
            else:
                city_centers[city] = (0.0, 0.0)
    else:
        for city in class_names:
            city_centers[city] = (0.0, 0.0)

    coords = []
    for idx in indices:
        if idx < len(class_names):
            city = class_names[idx]
            coords.append(city_centers.get(city, (0.0, 0.0)))
        else:
            coords.append((0.0, 0.0))

    return np.array(coords, dtype=np.float64)


def _compute_per_class_accuracy(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, tuple[float, int]]:
    """
    Обчислює точність для кожного класу окремо.

    Аргументи:
        logits:      Масив логітів (N, C).
        labels:      Масив міток (N,).
        class_names: Список назв класів.

    Повертає:
        Словник {class_name: (accuracy, count)}.
    """
    predictions = np.argmax(logits, axis=1)
    per_class: dict[str, tuple[float, int]] = {}

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        count = int(mask.sum())
        if count == 0:
            per_class[class_name] = (0.0, 0)
        else:
            correct = int((predictions[mask] == class_idx).sum())
            per_class[class_name] = (correct / count, count)

    return per_class


def _save_results(result: EvalResult, output_path: str) -> None:
    """Зберігає результати оцінки у JSON-файл."""
    import dataclasses

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = dataclasses.asdict(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Результати збережено: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оцінка моделі геолокації вуличних зображень",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",    type=str, required=True,
                        help="Шлях до .pth чекпоінту")
    parser.add_argument("--manifest",      type=str, required=True,
                        help="Шлях до CSV-маніфесту")
    parser.add_argument("--image-root",    type=str, default=None,
                        help="Корінь шляхів до зображень")
    parser.add_argument("--output",        type=str, default="results/eval_results.json",
                        help="Шлях для збереження JSON з результатами")
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--num-workers",   type=int, default=4)
    parser.add_argument("--img-size",      type=int, default=224)
    parser.add_argument("--split-method",  type=str, default="h3",
                        choices=["h3", "kmeans"])
    parser.add_argument("--countries",     type=str, nargs="+", default=None,
                        help="Коди країн для фільтрації (наприклад: UA)")
    parser.add_argument("--quality",       type=float, default=0.0,
                        help="Мінімальний поріг якості")
    parser.add_argument("--all-data",      action="store_true",
                        help="Оцінювати на всіх даних (без test split)")
    parser.add_argument("--seed",          type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = evaluate(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        image_root=args.image_root,
        countries=args.countries,
        quality_threshold=args.quality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_path=args.output,
        use_test_split=not args.all_data,
        img_size=args.img_size,
        split_method=args.split_method,
        seed=args.seed,
    )

    print(f"\n✓ Оцінка завершена:")
    print(f"  Top-1: {result.top1_accuracy*100:.2f}%")
    print(f"  Top-5: {result.top5_accuracy*100:.2f}%")
    print(f"  Mean Distance: {result.mean_distance_km:.1f} км")
    print(f"  GeoScore: {result.mean_geoscore:.1f}")


if __name__ == "__main__":
    main()
