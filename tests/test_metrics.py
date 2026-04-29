"""
test_metrics.py — Юніт-тести для code/metrics.py.

Перевіряє коректність обчислень Haversine distance, GeoScore, top-K accuracy.
Не потребує GPU чи зовнішніх даних.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    EARTH_RADIUS_KM,
    GEOSCORE_DECAY,
    GEOSCORE_MAX,
    compute_all_metrics,
    geoscore,
    haversine_batch,
    haversine_distance,
    mean_geoscore,
    top_k_accuracy,
)


# ──────────────────────────────────────────────────────────────────────────────
# Haversine distance
# ──────────────────────────────────────────────────────────────────────────────

class TestHaversineDistance:
    def test_zero_distance(self) -> None:
        """Відстань між однаковими точками = 0."""
        assert haversine_distance(50.45, 30.52, 50.45, 30.52) == pytest.approx(0.0, abs=1e-6)

    def test_kyiv_to_dnipro(self) -> None:
        """Київ → Дніпро приблизно 394 км."""
        d = haversine_distance(50.4501, 30.5234, 48.4647, 35.0462)
        assert 380.0 < d < 410.0

    def test_symmetry(self) -> None:
        """d(A,B) == d(B,A)."""
        d1 = haversine_distance(50.0, 30.0, 48.0, 35.0)
        d2 = haversine_distance(48.0, 35.0, 50.0, 30.0)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_antipodes(self) -> None:
        """Антипод — відстань ≈ π·R ≈ 20015 км."""
        d = haversine_distance(0.0, 0.0, 0.0, 180.0)
        expected = np.pi * EARTH_RADIUS_KM
        assert d == pytest.approx(expected, rel=1e-6)

    def test_batch_numpy(self) -> None:
        """Векторизація на numpy arrays."""
        lat1 = np.array([50.0, 48.0])
        lon1 = np.array([30.0, 35.0])
        lat2 = np.array([50.0, 48.0])
        lon2 = np.array([30.0, 35.0])
        result = haversine_distance(lat1, lon1, lat2, lon2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-6)

    def test_haversine_batch_shape(self) -> None:
        pred = np.array([[50.0, 30.0], [48.0, 35.0]])
        true = np.array([[50.0, 30.0], [49.0, 35.5]])
        d = haversine_batch(pred, true)
        assert d.shape == (2,)
        assert d[0] == pytest.approx(0.0, abs=1e-6)
        assert d[1] > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# GeoScore
# ──────────────────────────────────────────────────────────────────────────────

class TestGeoScore:
    def test_zero_distance_max_score(self) -> None:
        """d=0 км → 5000 балів."""
        assert geoscore(0.0) == pytest.approx(GEOSCORE_MAX, rel=1e-9)

    def test_decay(self) -> None:
        """d ≈ 1/e → ≈ 5000/e."""
        score = geoscore(GEOSCORE_DECAY)
        expected = GEOSCORE_MAX / np.e
        assert score == pytest.approx(expected, rel=1e-6)

    def test_score_monotonic_decrease(self) -> None:
        """GeoScore спадає при зростанні відстані."""
        scores = [geoscore(d) for d in [0, 100, 500, 1000, 5000]]
        for a, b in zip(scores, scores[1:]):
            assert a > b

    def test_very_far(self) -> None:
        """Дуже великі відстані → практично 0."""
        assert geoscore(100_000.0) < 1.0

    def test_negative_clamped(self) -> None:
        """Від'ємна відстань інтерпретується як 0."""
        assert geoscore(-100.0) == pytest.approx(GEOSCORE_MAX, rel=1e-9)

    def test_mean_geoscore(self) -> None:
        pred = np.array([[50.0, 30.0], [48.0, 35.0]])
        true = np.array([[50.0, 30.0], [48.0, 35.0]])
        assert mean_geoscore(pred, true) == pytest.approx(GEOSCORE_MAX, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Top-K accuracy
# ──────────────────────────────────────────────────────────────────────────────

class TestTopKAccuracy:
    def test_all_correct_top1(self) -> None:
        logits = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        labels = np.array([1, 0])
        assert top_k_accuracy(logits, labels, k=1) == pytest.approx(1.0)

    def test_all_wrong_top1(self) -> None:
        logits = np.array([[0.9, 0.1], [0.1, 0.9]])
        labels = np.array([1, 0])
        assert top_k_accuracy(logits, labels, k=1) == pytest.approx(0.0)

    def test_top_k_inclusive(self) -> None:
        logits = np.array([[0.5, 0.4, 0.1], [0.1, 0.5, 0.4]])
        labels = np.array([1, 2])
        # Top-1 дасть індекс 0 та 1 → 0.0 правильних
        assert top_k_accuracy(logits, labels, k=1) == pytest.approx(0.0)
        # Top-2 включає обидва → 1.0
        assert top_k_accuracy(logits, labels, k=2) == pytest.approx(1.0)

    def test_empty_input(self) -> None:
        logits = np.zeros((0, 3))
        labels = np.zeros(0, dtype=np.int64)
        assert top_k_accuracy(logits, labels, k=1) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# compute_all_metrics
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_all_metrics_keys() -> None:
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])
    pred = np.array([[50.0, 30.0], [48.0, 35.0]])
    true = np.array([[50.0, 30.0], [48.0, 35.0]])
    metrics = compute_all_metrics(logits, labels, pred, true)
    assert set(metrics.keys()) >= {
        "top1_acc", "top5_acc",
        "mean_distance_km", "median_distance_km", "mean_geoscore",
    }
    assert metrics["top1_acc"] == pytest.approx(1.0)
    assert metrics["mean_distance_km"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mean_geoscore"] == pytest.approx(GEOSCORE_MAX, rel=1e-6)
