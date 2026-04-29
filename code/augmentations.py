"""
augmentations.py — Аугментації даних для вуличних зображень.

Реалізує трансформації для навчання та валідації на основі torchvision.transforms.
Оптимізовано для вуличних панорам та знімків з автомобілів:
  - Без вертикального перевертання (знімки завжди «правосторонні»)
  - Без екстремальних геометричних викривлень
  - ColorJitter та RandomGrayscale для робастності до освітлення
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ──────────────────────────────────────────────────────────────────────────────
# Константи нормалізації ImageNet
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float]  = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# Основні функції трансформацій
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(
    img_size: int = 224,
    color_jitter_strength: float = 0.4,
    grayscale_prob: float = 0.1,
    random_crop_scale: Tuple[float, float] = (0.7, 1.0),
    random_crop_ratio: Tuple[float, float] = (0.85, 1.15),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Побудова пайплайну аугментацій для тренувального набору даних.

    Включає:
      - RandomResizedCrop: вирізання довільної ділянки + масштабування
      - RandomHorizontalFlip: горизонтальне дзеркалення (50%)
      - ColorJitter: випадкова зміна яскравості, контрасту, насиченості, тону
      - RandomGrayscale: конвертація у відтінки сірого (10%)
      - Нормалізація за статистиками ImageNet

    Аргументи:
        img_size:              Розмір вихідного зображення (пікселів).
        color_jitter_strength: Сила ColorJitter (0.0–1.0).
        grayscale_prob:        Ймовірність конвертації у сірий.
        random_crop_scale:     Діапазон масштабу вирізання.
        random_crop_ratio:     Діапазон співвідношення сторін.
        mean:                  Середні значення для нормалізації.
        std:                   Стандартні відхилення для нормалізації.

    Повертає:
        transforms.Compose — пайплайн трансформацій.
    """
    s = color_jitter_strength
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size,
            scale=random_crop_scale,
            ratio=random_crop_ratio,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        # Без вертикального перевертання: вуличні знімки мають чіткий «верх»
        transforms.ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s,
        ),
        transforms.RandomGrayscale(p=grayscale_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def get_val_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Побудова пайплайну трансформацій для валідаційного/тестового набору.

    Включає лише детерміновані операції:
      - Resize до (img_size * 256/224) для наступного CenterCrop
      - CenterCrop до img_size
      - Нормалізація за статистиками ImageNet

    Аргументи:
        img_size: Розмір вихідного зображення (пікселів).
        mean:     Середні значення для нормалізації.
        std:      Стандартні відхилення для нормалізації.

    Повертає:
        transforms.Compose — детермінований пайплайн трансформацій.
    """
    resize_size = int(img_size * 256 / 224)  # ~256 для img_size=224
    return transforms.Compose([
        transforms.Resize(
            resize_size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def get_strong_train_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Посилений пайплайн аугментацій для навчання з підвищеною регуляризацією.

    Додатково включає:
      - RandAugment: автоматичний вибір оптимальних аугментацій
      - RandomErasing: випадкове затирання прямокутних регіонів
      - Більш агресивний ColorJitter

    Аргументи:
        img_size: Розмір вихідного зображення.
        mean:     Середні значення для нормалізації.
        std:      Стандартні відхилення для нормалізації.

    Повертає:
        transforms.Compose — посилений пайплайн.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.6, 1.0),
            ratio=(0.8, 1.2),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_tta_transforms(
    img_size: int = 224,
    n_augmentations: int = 5,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> list[transforms.Compose]:
    """
    Test-Time Augmentation (TTA): список трансформацій для ансамблювання під час інференсу.

    Генерує n_augmentations варіантів аугментацій із різними налаштуваннями.
    Першою завжди йде стандартна валідаційна трансформація.

    Аргументи:
        img_size:        Розмір вихідного зображення.
        n_augmentations: Кількість TTA-варіантів.
        mean:            Середні значення нормалізації.
        std:             Стандартні відхилення нормалізації.

    Повертає:
        Список із n_augmentations трансформацій.
    """
    base = get_val_transforms(img_size=img_size, mean=mean, std=std)
    tta_list = [base]

    # Горизонтально дзеркалене зображення
    flipped = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0),  # завжди перевертаємо
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
    tta_list.append(flipped)

    # Злегка відмінні вирізання
    scales = [0.8, 0.85, 0.9]
    for i, scale in enumerate(scales[:n_augmentations - 2]):
        aug = transforms.Compose([
            transforms.RandomResizedCrop(
                size=img_size,
                scale=(scale, scale + 0.05),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])
        tta_list.append(aug)

    return tta_list[:n_augmentations]


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Зворотна нормалізація тензора зображення для візуалізації.

    Аргументи:
        tensor: Нормалізований тензор форми (C, H, W) або (N, C, H, W).
        mean:   Середні значення нормалізації.
        std:    Стандартні відхилення нормалізації.

    Повертає:
        Денормалізований тензор у тому ж форматі (значення від 0 до 1).
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t  = torch.tensor(std,  dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 3:  # (C, H, W)
        mean_t = mean_t.view(-1, 1, 1)
        std_t  = std_t.view(-1, 1, 1)
    elif tensor.ndim == 4:  # (N, C, H, W)
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t  = std_t.view(1, -1, 1, 1)
    else:
        raise ValueError(f"Очікуваний тензор 3D або 4D, отримано {tensor.ndim}D")

    return tensor * std_t + mean_t


# ──────────────────────────────────────────────────────────────────────────────
# Тест при прямому запуску
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    print("=== Тест аугментацій ===")

    # Створюємо тестове зображення
    dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    train_tf = get_train_transforms(img_size=224)
    val_tf   = get_val_transforms(img_size=224)
    strong_tf = get_strong_train_transforms(img_size=224)

    t_train = train_tf(dummy_img)
    t_val   = val_tf(dummy_img)
    t_strong = strong_tf(dummy_img)

    assert t_train.shape == (3, 224, 224), f"Невірна форма train: {t_train.shape}"
    assert t_val.shape   == (3, 224, 224), f"Невірна форма val: {t_val.shape}"
    assert t_strong.shape == (3, 224, 224), f"Невірна форма strong: {t_strong.shape}"

    # Перевіряємо денормалізацію
    restored = denormalize(t_val)
    assert restored.min() >= -0.1 and restored.max() <= 1.1, "Денормалізація: значення поза [0,1]"

    # TTA
    tta = get_tta_transforms(img_size=224, n_augmentations=4)
    assert len(tta) == 4
    for i, tf in enumerate(tta):
        out = tf(dummy_img)
        assert out.shape == (3, 224, 224), f"TTA [{i}]: невірна форма {out.shape}"

    print("Всі трансформації мають правильну форму (3, 224, 224)")
    print("Тест аугментацій пройдено успішно!")
