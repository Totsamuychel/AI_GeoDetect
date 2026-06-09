"""
augmentations.py — Data augmentation for street imagery.

Implements transformations for training and validation based on torchvision.transforms.
Optimized for street panoramas and car shots:
- No vertical flipping (images are always "right-side up")
- No extreme geometric distortions
- ColorJitter and RandomGrayscale for robustness to lighting
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet normalization constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

# OpenAI CLIP normalization constants (StreetCLIP/GeoCLIP use the CLIP
# vision encoder, which expects these mean/std; HF CLIPModel does NOT normalize
# the raw tensor itself — applying an ImageNet norm degrades the features).
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD: Tuple[float, float, float]  = (0.26862954, 0.26130258, 0.27577711)


def get_norm_for(architecture: str) -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float]
]:
"""
Returns (mean, std) normalizations according to the architecture.

baseline (EfficientNet-B2, ImageNet) → ImageNet stats;
streetclip / geoclip (CLIP backbone) → CLIP stats.
"""
    if str(architecture).lower().strip() in ("streetclip", "geoclip",
                                              "street_clip", "geo_clip"):
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


# ──────────────────────────────────────────────────────────────────────────────
# Main functions of transformations
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
Building an augmentation pipeline for the training dataset.

Includes:
- RandomResizedCrop: cropping an arbitrary area + scaling
- RandomHorizontalFlip: horizontal flipping (50%)
- ColorJitter: random change in brightness, contrast, saturation, hue
- RandomGrayscale: conversion to grayscale (10%)
- Normalization using ImageNet statistics

Arguments:
img_size: Size of the original image (pixels).
color_jitter_strength: ColorJitter strength (0.0–1.0).
grayscale_prob: Probability of conversion to gray.
random_crop_scale: Crop scale range.
random_crop_ratio: Aspect ratio range.
mean: Average values ​​for normalization.
std: Standard deviations for normalization.

Returns:
transforms.Compose — transformation pipeline.
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
        # No vertical flipping: street shots have a clear “top”
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
Build a transformation pipeline for the validation/test set.

Includes only deterministic operations:
- Resize to (img_size * 256/224) for next CenterCrop
- CenterCrop to img_size
- Normalize using ImageNet statistics

Arguments:
img_size: The size of the original image (in pixels).
mean: The average values ​​to normalize.
std: The standard deviations to normalize.

Returns:
transforms.Compose — a deterministic transformation pipeline.
    """
    resize_size = int(img_size * 256 / 224)  # ~256 for img_size=224
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
Enhanced augmentation pipeline for learning with increased regularization.

Additionally includes:
- RandAugment: automatic selection of optimal augmentations
- RandomErasing: random erasing of rectangular regions
- More aggressive ColorJitter

Arguments:
img_size: Size of the original image.
mean: Average values ​​for normalization.
std: Standard deviations for normalization.

Returns:
transforms.Compose — enhanced pipeline.
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
Test-Time Augmentation (TTA): A list of transformations to be assembled during inference.

Generates n_augmentations of augmentation variants with different settings.
The standard validation transformation is always first.

Arguments:
img_size: The size of the original image.
n_augmentations: The number of TTA variants.
mean: The average normalization values.
std: The standard deviations of the normalization.

Returns:
A list of n_augmentations transformations.
    """
    base = get_val_transforms(img_size=img_size, mean=mean, std=std)
    tta_list = [base]

    # Horizontally mirrored image
    flipped = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0), 
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
    tta_list.append(flipped)

    # Slightly different cuts
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
Reverse normalization of an image tensor for visualization.

Arguments:
tensor: Normalized tensor of form (C, H, W) or (N, C, H, W).
mean: Mean values ​​of normalization.
std: Standard deviations of normalization.

Returns:
Denormalized tensor in the same format (values ​​from 0 to 1).
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
# Direct start test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    print("=== Тест аугментацій ===")

    # Create a test image
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

    # Check denormalization
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
