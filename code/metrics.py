"""
metrics.py — Метрики для оцінки геолокаційних моделей.

Містить функції для обчислення:
- Відстані Гаверсина (Haversine distance)
- GeoScore (геосcore за формулою GeoGuessr)
- Top-K точності класифікації

Всі функції підтримують як скалярні значення, так і батчі (numpy/torch).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Константи
# ──────────────────────────────────────────────────────────────────────────────

EARTH_RADIUS_KM: float = 6371.0          # Середній радіус Землі (км)
GEOSCORE_MAX: float = 5000.0             # Максимальна кількість балів GeoGuessr
GEOSCORE_DECAY: float = 1492.7           # Константа загасання (км), що відповідає ~1/e при ≈1493 км


# ──────────────────────────────────────────────────────────────────────────────
# Допоміжні функції перетворення
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(x: Union[float, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Перетворює скаляр, numpy-масив або torch-тензор на numpy-масив float64."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Відстань Гаверсина
# ──────────────────────────────────────────────────────────────────────────────

def haversine_distance(
    lat1: Union[float, np.ndarray, torch.Tensor],
    lon1: Union[float, np.ndarray, torch.Tensor],
    lat2: Union[float, np.ndarray, torch.Tensor],
    lon2: Union[float, np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray]:
    """
    Обчислює велику колову відстань між двома точками на поверхні Землі
    за формулою Гаверсина.

    Аргументи:
        lat1: Широта першої точки (градуси). Може бути скаляром або масивом.
        lon1: Довгота першої точки (градуси).
        lat2: Широта другої точки (градуси).
        lon2: Довгота другої точки (градуси).

    Повертає:
        Відстань у кілометрах. Якщо на вході скаляри — повертає float,
        якщо масиви — numpy.ndarray тієї ж форми.

    Приклад:
        >>> haversine_distance(50.4501, 30.5234, 48.4647, 35.0462)  # Київ → Дніпро
        394.4...
    """
    lat1_r = _to_numpy(lat1)
    lon1_r = _to_numpy(lon1)
    lat2_r = _to_numpy(lat2)
    lon2_r = _to_numpy(lon2)

    # Перевід у радіани
    lat1_r = np.radians(lat1_r)
    lon1_r = np.radians(lon1_r)
    lat2_r = np.radians(lat2_r)
    lon2_r = np.radians(lon2_r)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    distance = EARTH_RADIUS_KM * c

    # Якщо на вході були скаляри — повертаємо float
    if distance.ndim == 0:
        return float(distance)
    return distance


def haversine_batch(
    pred_coords: Union[np.ndarray, torch.Tensor],
    true_coords: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """
    Векторизоване обчислення відстані Гаверсина для батчу координат.

    Аргументи:
        pred_coords: Масив форми (N, 2) з передбаченими (lat, lon).
        true_coords:  Масив форми (N, 2) з реальними (lat, lon).

    Повертає:
        numpy.ndarray форми (N,) з відстанями в км.
    """
    pred = _to_numpy(pred_coords)
    true = _to_numpy(true_coords)
    assert pred.shape == true.shape and pred.ndim == 2 and pred.shape[1] == 2, \
        "pred_coords та true_coords мають бути масивами форми (N, 2)"

    return haversine_distance(pred[:, 0], pred[:, 1], true[:, 0], true[:, 1])


# ──────────────────────────────────────────────────────────────────────────────
# GeoScore
# ──────────────────────────────────────────────────────────────────────────────

def geoscore(
    distance_km: Union[float, np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray]:
    """
    Обчислює GeoScore — бали за геолокацію, що нагадують систему GeoGuessr.

    Формула: score = 5000 × exp(−d / 1492.7)
    При d=0 км → 5000 балів.
    При d≈1493 км → ~1840 балів.
    При d≥10000 км → ~0 балів.

    Аргументи:
        distance_km: Відстань між передбаченою та реальною точками (км).

    Повертає:
        Бали від 0 до 5000 (float або numpy.ndarray).

    Приклад:
        >>> geoscore(0.0)
        5000.0
        >>> geoscore(1492.7)
        1839.39...
    """
    d = _to_numpy(distance_km)
    d = np.clip(d, 0.0, None)
    score = GEOSCORE_MAX * np.exp(-d / GEOSCORE_DECAY)

    if score.ndim == 0:
        return float(score)
    return score


def mean_geoscore(
    pred_coords: Union[np.ndarray, torch.Tensor],
    true_coords: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Обчислює середній GeoScore для батчу передбачень.

    Аргументи:
        pred_coords: Масив (N, 2) передбачених координат (lat, lon).
        true_coords: Масив (N, 2) реальних координат (lat, lon).

    Повертає:
        Середній GeoScore як float.
    """
    distances = haversine_batch(pred_coords, true_coords)
    scores = geoscore(distances)
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# Top-K точність
# ──────────────────────────────────────────────────────────────────────────────

def top_k_accuracy(
    logits: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    k: int = 1,
) -> float:
    """
    Обчислює Top-K точність для задачі класифікації.

    Аргументи:
        logits: Масив форми (N, C) — «сирі» виходи моделі (logits або ймовірності).
        labels: Масив форми (N,) — справжні мітки класів (цілі числа).
        k:      Кількість топ-прогнозів для перевірки (за замовчуванням 1).

    Повертає:
        Частка правильних передбачень (від 0.0 до 1.0).

    Приклад:
        >>> logits = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.1, 0.2]])
        >>> labels = torch.tensor([1, 0])
        >>> top_k_accuracy(logits, labels, k=1)
        1.0
    """
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
    else:
        labels_np = np.asarray(labels, dtype=np.int64)

    n_samples = logits_np.shape[0]
    if n_samples == 0:
        return 0.0

    # Індекси топ-K передбачень для кожного зразка (за спаданням)
    top_k_preds = np.argsort(logits_np, axis=1)[:, -k:]  # (N, k)

    # Перевірка, чи входить правильна мітка до топ-K
    correct = 0
    for i in range(n_samples):
        if labels_np[i] in top_k_preds[i]:
            correct += 1

    return correct / n_samples


def top_k_accuracy_torch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    """
    Обчислює Top-K точність із використанням torch операцій (швидше для GPU).

    Аргументи:
        logits: Тензор форми (N, C).
        labels: Тензор форми (N,) з цілими мітками.
        k:      Значення K.

    Повертає:
        Скалярний torch.Tensor з точністю від 0.0 до 1.0.
    """
    with torch.no_grad():
        _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=False)  # (N, k)
        labels_expanded = labels.view(-1, 1).expand_as(top_k_preds)          # (N, k)
        correct = top_k_preds.eq(labels_expanded).any(dim=1).float()
        return correct.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Зведені метрики для оцінки
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    logits: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    pred_coords: Union[np.ndarray, torch.Tensor],
    true_coords: Union[np.ndarray, torch.Tensor],
) -> dict[str, float]:
    """
    Обчислює всі метрики одночасно: Top-1, Top-5, Haversine-відстань, GeoScore.

    Аргументи:
        logits:      Логіти класифікатора (N, C).
        labels:      Справжні мітки класів (N,).
        pred_coords: Передбачені координати (N, 2) у форматі (lat, lon).
        true_coords: Реальні координати (N, 2) у форматі (lat, lon).

    Повертає:
        Словник із метриками:
        {
            "top1_acc": float,
            "top5_acc": float,
            "mean_distance_km": float,
            "median_distance_km": float,
            "mean_geoscore": float,
        }
    """
    distances = haversine_batch(pred_coords, true_coords)
    scores = geoscore(distances)

    return {
        "top1_acc":           top_k_accuracy(logits, labels, k=1),
        "top5_acc":           top_k_accuracy(logits, labels, k=5),
        "mean_distance_km":   float(np.mean(distances)),
        "median_distance_km": float(np.median(distances)),
        "mean_geoscore":      float(np.mean(scores)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Тест при прямому запуску
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Тест метрик ===")

    # Відстань Гаверсина: Київ → Дніпро
    d = haversine_distance(50.4501, 30.5234, 48.4647, 35.0462)
    print(f"Київ → Дніпро: {d:.1f} км  (очікувано ≈ 394 км)")

    # GeoScore
    print(f"GeoScore(0 км)  = {geoscore(0.0):.1f}  (очікувано 5000.0)")
    print(f"GeoScore(1493)  = {geoscore(1492.7):.1f}")
    print(f"GeoScore(10000) = {geoscore(10000.0):.2f}")

    # Top-K точність
    logits_test = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]])
    labels_test = np.array([1, 0, 2])
    print(f"Top-1 acc = {top_k_accuracy(logits_test, labels_test, k=1):.2f}  (очікувано 1.0)")
    print(f"Top-2 acc = {top_k_accuracy(logits_test, labels_test, k=2):.2f}  (очікувано 1.0)")

    # Batch metrics
    pred_c = np.array([[50.4501, 30.5234], [48.4647, 35.0462]])
    true_c = np.array([[48.4647, 35.0462], [50.4501, 30.5234]])
    print(f"Batch відстані: {haversine_batch(pred_c, true_c)}")
    print("Всі тести пройдено успішно.")
