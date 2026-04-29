"""
dataset.py — DataLoader для датасетів OSV-5M та Mapillary.

GeoDataset зчитує CSV-маніфест з полями:
    image_id, filepath, lat, lon, country, region, city, source,
    capture_date, quality_score

Реалізує:
- Географічно стратифіковане розбиття через H3-гексагони або k-means
  для запобігання витоку даних (просторова автокореляція)
- Фільтрацію за країною/містом та порогом якості
- Повертає тензор зображення + індекс міста + GPS-координати
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Допоміжні функції розбиття
# ──────────────────────────────────────────────────────────────────────────────

def _split_by_h3(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    h3_resolution: int = 4,
    seed: int = 42,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Стратифіковане розбиття за H3-гексагонами для запобігання просторовому витоку.

    Кожен гексагон цілком потрапляє або в train, або у val, або в test.
    Це запобігає ситуації, коли схожі сцени з одного місця є і в train, і в test.

    Аргументи:
        df:            DataFrame із колонками 'lat', 'lon'.
        train_frac:    Частка навчальних даних.
        val_frac:      Частка валідаційних даних.
        h3_resolution: Роздільна здатність H3 (4 ≈ 86 км², 5 ≈ 26 км²).
        seed:          Seed для відтворюваності.

    Повертає:
        Кортеж (train_idx, val_idx, test_idx) — pandas.Index.
    """
    try:
        import h3
    except ImportError:
        logger.warning("h3 не встановлено, використовується k-means розбиття")
        return _split_by_kmeans(df, train_frac=train_frac, val_frac=val_frac, seed=seed)

    # API h3 >= 4.0 використовує latlng_to_cell(); старий h3 < 4.0 має geo_to_h3().
    if hasattr(h3, "latlng_to_cell"):
        _latlng_to_cell = h3.latlng_to_cell  # type: ignore[attr-defined]
    elif hasattr(h3, "geo_to_h3"):
        _latlng_to_cell = h3.geo_to_h3  # type: ignore[attr-defined]
    else:
        logger.warning("Бібліотека h3 не підтримує очікуваний API, fallback до k-means")
        return _split_by_kmeans(df, train_frac=train_frac, val_frac=val_frac, seed=seed)

    rng = np.random.default_rng(seed)

    # Призначаємо кожному зображенню H3-гексагон
    df = df.copy()
    df["h3_cell"] = df.apply(
        lambda row: _latlng_to_cell(row["lat"], row["lon"], h3_resolution), axis=1
    )

    unique_cells = df["h3_cell"].unique()
    rng.shuffle(unique_cells)

    n_total = len(unique_cells)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)

    train_cells = set(unique_cells[:n_train])
    val_cells   = set(unique_cells[n_train:n_train + n_val])
    test_cells  = set(unique_cells[n_train + n_val:])

    train_idx = df[df["h3_cell"].isin(train_cells)].index
    val_idx   = df[df["h3_cell"].isin(val_cells)].index
    test_idx  = df[df["h3_cell"].isin(test_cells)].index

    logger.info(
        f"H3 розбиття (res={h3_resolution}): "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def _split_by_kmeans(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    n_clusters: int = 200,
    seed: int = 42,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Стратифіковане розбиття за k-means кластерами координат.

    Альтернатива H3 для середовищ без бібліотеки h3.

    Аргументи:
        df:         DataFrame із колонками 'lat', 'lon'.
        train_frac: Частка навчальних даних.
        val_frac:   Частка валідаційних даних.
        n_clusters: Кількість кластерів k-means.
        seed:       Seed для відтворюваності.

    Повертає:
        Кортеж (train_idx, val_idx, test_idx).
    """
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    coords = df[["lat", "lon"]].values

    n_clusters = min(n_clusters, len(df))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
    cluster_labels = kmeans.fit_predict(coords)

    unique_clusters = np.unique(cluster_labels)
    rng.shuffle(unique_clusters)

    n_total = len(unique_clusters)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)

    train_clusters = set(unique_clusters[:n_train])
    val_clusters   = set(unique_clusters[n_train:n_train + n_val])

    train_mask = np.isin(cluster_labels, list(train_clusters))
    val_mask   = np.isin(cluster_labels, list(val_clusters))
    test_mask  = ~train_mask & ~val_mask

    train_idx = df.index[train_mask]
    val_idx   = df.index[val_mask]
    test_idx  = df.index[test_mask]

    logger.info(
        f"K-means розбиття (k={n_clusters}): "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# Основний клас датасету
# ──────────────────────────────────────────────────────────────────────────────

class GeoDataset(Dataset):
    """
    Dataset для вуличних зображень із географічними мітками.

    Зчитує CSV-маніфест та завантажує зображення з диску.
    Повертає (image_tensor, city_index, (lat, lon)) для кожного елементу.

    Атрибути:
        df:          Відфільтрований та оброблений DataFrame.
        class_names: Список назв міст (упорядкований за алфавітом).
        num_classes: Кількість класів (міст).
        transform:   Функція трансформації зображень.

    Приклад:
        >>> dataset = GeoDataset("data/manifest.csv", countries=["UA"])
        >>> img, city_idx, (lat, lon) = dataset[0]
        >>> print(img.shape, city_idx, lat, lon)
    """

    REQUIRED_COLUMNS = {"image_id", "filepath", "lat", "lon"}
    OPTIONAL_COLUMNS = {
        "country", "region", "city", "source",
        "capture_date", "quality_score",
    }

    def __init__(
        self,
        manifest_path: Union[str, Path],
        transform: Optional[Callable] = None,
        countries: Optional[list[str]] = None,
        cities: Optional[list[str]] = None,
        quality_threshold: float = 0.0,
        image_root: Optional[Union[str, Path]] = None,
        subset_indices: Optional[pd.Index] = None,
        cache_images: bool = False,
        fallback_on_error: bool = True,
    ) -> None:
        """
        Аргументи:
            manifest_path:    Шлях до CSV-маніфесту.
            transform:        torchvision transform для зображень.
            countries:        Список кодів країн для фільтрації (наприклад, ['UA']).
            cities:           Список назв міст для фільтрації.
            quality_threshold: Мінімальний поріг якості (quality_score >= threshold).
            image_root:       Корінь шляхів до зображень (якщо filepath відносний).
            subset_indices:   Вибірка рядків DataFrame (для train/val/test split).
            cache_images:     Якщо True, кешує зображення в RAM.
            fallback_on_error: Якщо True, повертає чорне зображення при помилці.
        """
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.image_root = Path(image_root) if image_root else None
        self.cache_images = cache_images
        self.fallback_on_error = fallback_on_error
        self._cache: dict[int, torch.Tensor] = {}

        # Завантаження маніфесту
        self.df = self._load_manifest(manifest_path)

        # Заповнення відсутніх колонок
        for col in self.OPTIONAL_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = None

        # Фільтрація
        self.df = self._apply_filters(
            self.df,
            countries=countries,
            cities=cities,
            quality_threshold=quality_threshold,
        )

        # Вибірка підмножини
        if subset_indices is not None:
            valid_indices = subset_indices[subset_indices.isin(self.df.index)]
            self.df = self.df.loc[valid_indices]

        self.df = self.df.reset_index(drop=True)

        # Побудова словника міст → індекс
        self._build_class_mapping()

        logger.info(
            f"GeoDataset: {len(self.df)} зображень, "
            f"{self.num_classes} класів (міст)"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Внутрішні методи
    # ──────────────────────────────────────────────────────────────────────

    def _load_manifest(self, path: Union[str, Path]) -> pd.DataFrame:
        """Завантажує та валідує CSV-маніфест."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Маніфест не знайдено: {path}")

        df = pd.read_csv(path, low_memory=False)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"У маніфесті відсутні обов'язкові колонки: {missing}")

        # Типізація
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        if "quality_score" in df.columns:
            df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0.0)
        else:
            df["quality_score"] = 1.0

        # Видалення рядків із невалідними координатами
        before = len(df)
        df = df.dropna(subset=["lat", "lon", "filepath"])
        df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
        after = len(df)
        if before != after:
            logger.warning(f"Видалено {before - after} рядків із невалідними даними")

        return df

    def _apply_filters(
        self,
        df: pd.DataFrame,
        countries: Optional[list[str]],
        cities: Optional[list[str]],
        quality_threshold: float,
    ) -> pd.DataFrame:
        """Застосовує фільтри до DataFrame."""
        original_len = len(df)

        if countries and "country" in df.columns:
            df = df[df["country"].str.upper().isin([c.upper() for c in countries])]

        if cities and "city" in df.columns:
            df = df[df["city"].isin(cities)]

        df = df[df["quality_score"] >= quality_threshold]

        logger.info(
            f"Фільтрація: {original_len} → {len(df)} зображень "
            f"(країни={countries}, міста={cities}, якість≥{quality_threshold})"
        )
        return df

    def _build_class_mapping(self) -> None:
        """Будує відображення місто → індекс класу."""
        if "city" in self.df.columns and self.df["city"].notna().any():
            unique_cities = sorted(self.df["city"].dropna().unique())
        else:
            unique_cities = ["unknown"]
            self.df["city"] = "unknown"

        self.class_names: list[str] = list(unique_cities)
        self._city_to_idx: dict[str, int] = {c: i for i, c in enumerate(unique_cities)}
        self.num_classes: int = len(unique_cities)

    def _load_image(self, filepath: str) -> Image.Image:
        """Завантажує зображення з диску."""
        if self.image_root:
            full_path = self.image_root / filepath
        else:
            full_path = Path(filepath)

        if not full_path.exists():
            raise FileNotFoundError(f"Зображення не знайдено: {full_path}")

        return Image.open(full_path).convert("RGB")

    def _get_fallback_image(self, img_size: int = 224) -> torch.Tensor:
        """Повертає чорний тензор як запасний варіант при помилці."""
        return torch.zeros(3, img_size, img_size)

    # ──────────────────────────────────────────────────────────────────────
    # Dataset API
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        """
        Повертає елемент датасету.

        Аргументи:
            idx: Індекс елементу.

        Повертає:
            Кортеж (image_tensor, city_index, coords):
            - image_tensor: torch.Tensor форми (3, H, W)
            - city_index:   int — індекс міста
            - coords:       torch.Tensor форми (2,) з [lat, lon] у градусах.

        Примітка: повертаємо саме tensor (а не tuple), щоб default collate
        у DataLoader давав чистий тензор форми (N, 2), а не фрагментовану
        структуру, що залежить від версії PyTorch.
        """
        if self.cache_images and idx in self._cache:
            img_tensor = self._cache[idx]
        else:
            row = self.df.iloc[idx]
            try:
                img = self._load_image(str(row["filepath"]))
                if self.transform:
                    img_tensor = self.transform(img)
                else:
                    img_tensor = torch.from_numpy(
                        np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                    )
            except Exception as e:
                logger.warning(f"Помилка завантаження зображення [{idx}]: {e}")
                if self.fallback_on_error:
                    img_tensor = self._get_fallback_image()
                else:
                    raise

            if self.cache_images:
                self._cache[idx] = img_tensor

        row = self.df.iloc[idx]
        city = str(row.get("city", "unknown")) if pd.notna(row.get("city")) else "unknown"
        city_idx = self._city_to_idx.get(city, 0)
        lat = float(row["lat"])
        lon = float(row["lon"])
        coords = torch.tensor([lat, lon], dtype=torch.float32)

        return img_tensor, city_idx, coords

    def get_sample_info(self, idx: int) -> dict:
        """Повертає метадані зразка без завантаження зображення."""
        row = self.df.iloc[idx]
        return {
            "image_id":      row.get("image_id"),
            "filepath":      row.get("filepath"),
            "lat":           float(row["lat"]),
            "lon":           float(row["lon"]),
            "country":       row.get("country"),
            "region":        row.get("region"),
            "city":          row.get("city"),
            "source":        row.get("source"),
            "capture_date":  row.get("capture_date"),
            "quality_score": float(row.get("quality_score", 0.0)),
            "city_index":    self._city_to_idx.get(str(row.get("city", "")), 0),
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Обчислює ваги класів для збалансованого навчання (зворотна частота).

        Повертає:
            torch.Tensor форми (num_classes,).
        """
        city_counts = self.df["city"].value_counts()
        weights = torch.zeros(self.num_classes)
        for city, idx in self._city_to_idx.items():
            count = city_counts.get(city, 1)
            weights[idx] = 1.0 / count
        weights = weights / weights.sum()
        return weights

    # ──────────────────────────────────────────────────────────────────────
    # Географічне розбиття
    # ──────────────────────────────────────────────────────────────────────

    def get_split_indices(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        method: str = "h3",
        h3_resolution: int = 4,
        n_clusters: int = 200,
        seed: int = 42,
    ) -> tuple[pd.Index, pd.Index, pd.Index]:
        """
        Географічно стратифіковане розбиття датасету.

        Аргументи:
            train_frac:    Частка тренувальних даних.
            val_frac:      Частка валідаційних даних.
            method:        'h3' або 'kmeans'.
            h3_resolution: Роздільна здатність H3 (лише для method='h3').
            n_clusters:    Кількість кластерів (лише для method='kmeans').
            seed:          Seed для відтворюваності.

        Повертає:
            Кортеж (train_idx, val_idx, test_idx).
        """
        if method == "h3":
            return _split_by_h3(
                self.df,
                train_frac=train_frac,
                val_frac=val_frac,
                h3_resolution=h3_resolution,
                seed=seed,
            )
        elif method == "kmeans":
            return _split_by_kmeans(
                self.df,
                train_frac=train_frac,
                val_frac=val_frac,
                n_clusters=n_clusters,
                seed=seed,
            )
        else:
            raise ValueError(f"Невідомий метод розбиття: {method}. Підтримуються: 'h3', 'kmeans'")


# ──────────────────────────────────────────────────────────────────────────────
# Фабричні функції для DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    manifest_path: Union[str, Path],
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    countries: Optional[list[str]] = None,
    cities: Optional[list[str]] = None,
    quality_threshold: float = 0.0,
    image_root: Optional[Union[str, Path]] = None,
    split_method: str = "h3",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Створює DataLoader для тренування, валідації та тестування.

    Аргументи:
        manifest_path:     Шлях до CSV-маніфесту.
        train_transform:   Трансформації для тренувального набору.
        val_transform:     Трансформації для валідаційного/тестового набору.
        countries:         Фільтр за країнами.
        cities:            Фільтр за містами.
        quality_threshold: Мінімальний поріг якості.
        image_root:        Корінь шляхів до зображень.
        split_method:      Метод розбиття: 'h3' або 'kmeans'.
        train_frac:        Частка тренувальних даних.
        val_frac:          Частка валідаційних даних.
        batch_size:        Розмір батчу.
        num_workers:       Кількість воркерів DataLoader.
        pin_memory:        Пришпилення пам'яті для GPU.
        seed:              Seed для відтворюваності.

    Повертає:
        Словник {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.
    """
    from augmentations import get_train_transforms, get_val_transforms

    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()

    # Повний датасет для обчислення розбиття
    full_dataset = GeoDataset(
        manifest_path=manifest_path,
        transform=None,
        countries=countries,
        cities=cities,
        quality_threshold=quality_threshold,
        image_root=image_root,
    )

    train_idx, val_idx, test_idx = full_dataset.get_split_indices(
        train_frac=train_frac,
        val_frac=val_frac,
        method=split_method,
        seed=seed,
    )

    # Окремі датасети зі своїми трансформаціями
    train_ds = GeoDataset(
        manifest_path=manifest_path,
        transform=train_transform,
        countries=countries,
        cities=cities,
        quality_threshold=quality_threshold,
        image_root=image_root,
        subset_indices=train_idx,
    )
    # Синхронізуємо class_names
    train_ds.class_names = full_dataset.class_names
    train_ds._city_to_idx = full_dataset._city_to_idx
    train_ds.num_classes = full_dataset.num_classes

    val_ds = GeoDataset(
        manifest_path=manifest_path,
        transform=val_transform,
        countries=countries,
        cities=cities,
        quality_threshold=quality_threshold,
        image_root=image_root,
        subset_indices=val_idx,
    )
    val_ds.class_names = full_dataset.class_names
    val_ds._city_to_idx = full_dataset._city_to_idx
    val_ds.num_classes = full_dataset.num_classes

    test_ds = GeoDataset(
        manifest_path=manifest_path,
        transform=val_transform,
        countries=countries,
        cities=cities,
        quality_threshold=quality_threshold,
        image_root=image_root,
        subset_indices=test_idx,
    )
    test_ds.class_names = full_dataset.class_names
    test_ds._city_to_idx = full_dataset._city_to_idx
    test_ds.num_classes = full_dataset.num_classes

    def _make_loader(ds: GeoDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=shuffle,  # drop_last лише для тренування
            persistent_workers=num_workers > 0,
        )

    return {
        "train": _make_loader(train_ds, shuffle=True),
        "val":   _make_loader(val_ds,   shuffle=False),
        "test":  _make_loader(test_ds,  shuffle=False),
        # Зберігаємо метадані для зручності
        "num_classes": full_dataset.num_classes,
        "class_names": full_dataset.class_names,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Генератор синтетичного маніфесту для тестів
# ──────────────────────────────────────────────────────────────────────────────

def create_dummy_manifest(
    output_path: Union[str, Path],
    n_samples: int = 1000,
    cities: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерує синтетичний CSV-маніфест для тестування без реальних даних.

    Аргументи:
        output_path: Шлях для збереження CSV.
        n_samples:   Кількість синтетичних зразків.
        cities:      Список міст (за замовчуванням — основні міста України).
        seed:        Seed для відтворюваності.

    Повертає:
        DataFrame із синтетичним маніфестом.
    """
    if cities is None:
        cities = [
            "Київ", "Харків", "Одеса", "Дніпро", "Запоріжжя",
            "Львів", "Кривий Ріг", "Миколаїв", "Вінниця", "Херсон",
        ]

    # Приблизні координати центрів міст України
    city_centers = {
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
    }

    rng = np.random.default_rng(seed)

    records = []
    for i in range(n_samples):
        city = rng.choice(cities)
        center = city_centers.get(city, (49.0, 32.0))
        lat = float(center[0] + rng.normal(0, 0.1))
        lon = float(center[1] + rng.normal(0, 0.1))

        records.append({
            "image_id":     f"img_{i:06d}",
            "filepath":     f"images/{city}/{i:06d}.jpg",
            "lat":          round(lat, 6),
            "lon":          round(lon, 6),
            "country":      "UA",
            "region":       "Невідомо",
            "city":         city,
            "source":       rng.choice(["mapillary", "osv5m"]),
            "capture_date": f"2023-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}",
            "quality_score": round(float(rng.uniform(0.3, 1.0)), 3),
        })

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Синтетичний маніфест збережено: {output_path} ({n_samples} зразків)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Тест при прямому запуску
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import sys

    logging.basicConfig(level=logging.INFO)
    print("=== Тест GeoDataset ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.csv"
        df = create_dummy_manifest(manifest_path, n_samples=200, seed=42)
        print(f"Маніфест створено: {len(df)} рядків")

        # Ініціалізація датасету без реальних зображень
        dataset = GeoDataset(
            manifest_path=manifest_path,
            transform=None,
            countries=["UA"],
            quality_threshold=0.5,
        )
        print(f"Датасет: {len(dataset)} зразків, {dataset.num_classes} міст")
        print(f"Міста: {dataset.class_names}")

        # Тест розбиття (k-means, бо h3 може бути відсутній)
        train_idx, val_idx, test_idx = dataset.get_split_indices(
            method="kmeans", train_frac=0.7, val_frac=0.15, n_clusters=20, seed=42
        )
        print(f"K-means розбиття: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Ваги класів
        weights = dataset.get_class_weights()
        assert len(weights) == dataset.num_classes
        print(f"Ваги класів: shape={weights.shape}, sum={weights.sum():.3f}")

        # Тест метаданих (без завантаження зображень)
        info = dataset.get_sample_info(0)
        print(f"Метадані зразка [0]: {info}")

        print("Тест GeoDataset пройдено успішно!")
