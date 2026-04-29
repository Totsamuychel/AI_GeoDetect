"""
test_dataset.py — Юніт-тести для code/dataset.py.

Перевіряє:
- Синтетичний маніфест (create_dummy_manifest)
- Завантаження маніфесту та валідацію колонок
- Фільтрацію за країною, містом, якістю
- Географічне розбиття (k-means та h3, якщо встановлено)
- Сигнатуру __getitem__ (повертає tensor coords форми (2,))
- Обчислення ваг класів
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # skip, якщо torch недоступний

from dataset import (
    GeoDataset,
    _split_by_kmeans,
    create_dummy_manifest,
)


# ──────────────────────────────────────────────────────────────────────────────
# create_dummy_manifest
# ──────────────────────────────────────────────────────────────────────────────

class TestDummyManifest:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.csv"
        df = create_dummy_manifest(out, n_samples=50, seed=0)
        assert out.exists()
        assert len(df) == 50

    def test_has_required_columns(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.csv"
        df = create_dummy_manifest(out, n_samples=20, seed=0)
        for col in ["image_id", "filepath", "lat", "lon", "country", "city"]:
            assert col in df.columns

    def test_coords_within_valid_range(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.csv"
        df = create_dummy_manifest(out, n_samples=100, seed=0)
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(-180, 180).all()


# ──────────────────────────────────────────────────────────────────────────────
# GeoDataset init / filtering
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_manifest(tmp_path: Path) -> Path:
    out = tmp_path / "manifest.csv"
    create_dummy_manifest(out, n_samples=200, seed=42)
    return out


class TestGeoDataset:
    def test_loads_without_filtering(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest, transform=None)
        assert len(ds) > 0
        assert ds.num_classes >= 1

    def test_quality_filter(self, dummy_manifest: Path) -> None:
        ds_low  = GeoDataset(dummy_manifest, quality_threshold=0.0)
        ds_high = GeoDataset(dummy_manifest, quality_threshold=0.9)
        assert len(ds_high) <= len(ds_low)

    def test_country_filter(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest, countries=["UA"])
        assert (ds.df["country"].str.upper() == "UA").all()

    def test_city_filter(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest, cities=["Київ"])
        assert (ds.df["city"] == "Київ").all()

    def test_getitem_fallback_returns_tensors(self, dummy_manifest: Path) -> None:
        """Файлів на диску немає → має спрацювати fallback_on_error=True."""
        ds = GeoDataset(dummy_manifest, transform=None, fallback_on_error=True)
        img, city_idx, coords = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3 and img.shape[0] == 3
        assert isinstance(city_idx, int)
        # КЛЮЧОВЕ: coords має бути tensor форми (2,) — це виправлення бага
        # крихкої десеріалізації coords_tuple у train.py/evaluate.py.
        assert isinstance(coords, torch.Tensor)
        assert coords.shape == (2,)
        assert coords.dtype == torch.float32

    def test_default_collate_produces_batch_tensor(self, dummy_manifest: Path) -> None:
        """Перевірка, що DataLoader.collate_fn створює acc тензор (N, 2)."""
        from torch.utils.data import DataLoader

        ds = GeoDataset(dummy_manifest, transform=None, fallback_on_error=True)
        loader = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
        images, labels, coords = next(iter(loader))
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
        assert coords.shape == (4, 2)
        assert coords.dtype == torch.float32

    def test_class_weights_sum_to_one(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest)
        w = ds.get_class_weights()
        assert w.shape == (ds.num_classes,)
        assert float(w.sum()) == pytest.approx(1.0, rel=1e-5)

    def test_sample_info_has_keys(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest)
        info = ds.get_sample_info(0)
        for key in ["image_id", "filepath", "lat", "lon", "city", "city_index"]:
            assert key in info


# ──────────────────────────────────────────────────────────────────────────────
# Географічне розбиття
# ──────────────────────────────────────────────────────────────────────────────

class TestSplit:
    def test_kmeans_split_no_overlap(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest)
        train_idx, val_idx, test_idx = ds.get_split_indices(
            method="kmeans", train_frac=0.7, val_frac=0.15, n_clusters=20, seed=0,
        )
        all_idx = set(train_idx) | set(val_idx) | set(test_idx)
        # Має не бути дубльованих індексів
        assert len(set(train_idx) & set(val_idx))  == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx)   & set(test_idx)) == 0
        # Сума покриває всі рядки
        assert len(all_idx) == len(ds.df)

    def test_kmeans_split_proportions(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest)
        train_idx, val_idx, test_idx = ds.get_split_indices(
            method="kmeans", train_frac=0.7, val_frac=0.15, n_clusters=20, seed=0,
        )
        total = len(train_idx) + len(val_idx) + len(test_idx)
        # Дозволяємо відхилення від номінальних 70/15/15
        assert len(train_idx) / total >= 0.5
        assert len(test_idx) >= 0

    def test_h3_split_fallback_if_unavailable(self, dummy_manifest: Path) -> None:
        """Якщо h3 встановлено — перевіряємо, що розбиття не падає і виправлення
        API (latlng_to_cell vs geo_to_h3) працює."""
        h3 = pytest.importorskip("h3")
        ds = GeoDataset(dummy_manifest)
        train_idx, val_idx, test_idx = ds.get_split_indices(
            method="h3", train_frac=0.7, val_frac=0.15, h3_resolution=4, seed=0,
        )
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(ds.df)

    def test_unknown_split_method_raises(self, dummy_manifest: Path) -> None:
        ds = GeoDataset(dummy_manifest)
        with pytest.raises(ValueError):
            ds.get_split_indices(method="bogus")
