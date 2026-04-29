"""
models.py — Архітектури нейронних мереж для геолокації вуличних зображень.

Реалізовано три архітектури:
1. BaselineCNN       — EfficientNet-B2 із замінною головою класифікатора
2. StreetCLIPModel   — geolocal/StreetCLIP із HuggingFace + лінійний пробник
3. GeoCLIPModel      — CLIP ViT + GPS Location Encoder (Random Fourier Features),
                       контрастивне навчання у стилі SimCLR

Кожна модель реалізує:
    predict(image_tensor)   → top-5 передбачень із ймовірностями
    get_embeddings(batch)   → embedding-вектори для t-SNE та пошуку
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. BaselineCNN — EfficientNet-B2 з налаштовуваною головою
# ──────────────────────────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """
    Базова CNN-архітектура на основі EfficientNet-B2.

    Використовує pretrained-ваги ImageNet з torchvision.
    Класифікаційна голова повністю замінюється під потрібну кількість класів.

    Архітектура голови:
        Linear(1408 → 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512 → N)

    Аргументи:
        num_classes:    Кількість вихідних класів (міст).
        pretrained:     Якщо True, завантажує ваги ImageNet.
        dropout_rate:   Коефіцієнт Dropout у голові класифікатора.
        freeze_backbone: Якщо True, заморожує backbone після ініціалізації.
    """

    BACKBONE_OUT_FEATURES = 1408  # Розмір виходу EfficientNet-B2

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Завантаження EfficientNet-B2
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b2(weights=weights)

        # Вилучаємо всі шари крім фінального класифікатора
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        self._embed_dim = self.BACKBONE_OUT_FEATURES

        # Нова класифікаційна голова
        self.classifier = nn.Sequential(
            nn.Linear(self._embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Заморожує всі шари backbone (крім класифікатора)."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("BaselineCNN: backbone заморожено")

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """
        Розморожує останні n блоків backbone для тонкого налаштування.

        Аргументи:
            n: Кількість блоків для розморожування.
        """
        blocks = list(self.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"BaselineCNN: розморожено {n} блоків, {trainable:,} навчальних параметрів")

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Витягує embedding-вектори перед класифікаційною головою.

        Аргументи:
            x: Батч зображень (N, 3, H, W).

        Повертає:
            Тензор форми (N, 512) — normalized embeddings.
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)  # (N, 1408)

        # Перший шар голови як embedding
        with torch.no_grad():
            emb = self.classifier[0](feat)  # Linear → (N, 512)
            emb = self.classifier[1](emb)   # BN
            emb = self.classifier[2](emb)   # ReLU

        return F.normalize(emb, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід.

        Аргументи:
            x: Батч (N, 3, H, W).

        Повертає:
            Логіти (N, num_classes).
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)
        return self.classifier(feat)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        class_names: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Передбачення для одного зображення або батчу.

        Аргументи:
            x:           Зображення (3, H, W) або батч (N, 3, H, W).
            class_names: Список назв класів.
            top_k:       Кількість топ-передбачень.

        Повертає:
            Список словників [{'class': str, 'index': int, 'prob': float}, ...].
        """
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        probs  = F.softmax(logits, dim=1)

        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)

        results = []
        for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
            name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
            results.append({"class": name, "index": idx, "prob": round(prob, 6)})

        return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. StreetCLIPModel — geolocal/StreetCLIP + лінійний пробник
# ──────────────────────────────────────────────────────────────────────────────

class StreetCLIPModel(nn.Module):
    """
    Модель на основі StreetCLIP (geolocal/StreetCLIP) з HuggingFace.

    StreetCLIP — CLIP-модель, дообкована на вуличних зображеннях зі всього світу.
    Візуальний енкодер заморожується; додається лінійна голова для класифікації.

    Архітектура:
        CLIPVisionModel → embed (512/768) → Linear Probe → N класів

    Аргументи:
        num_classes:       Кількість класів.
        model_name:        HuggingFace ідентифікатор моделі.
        freeze_vision:     Якщо True, заморожує CLIP vision encoder.
        hidden_dim:        Розмір прихованого шару (0 = пряма лінійна проба).
        dropout_rate:      Dropout у голові.
    """

    HUGGINGFACE_MODEL = "geolocal/StreetCLIP"

    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "geolocal/StreetCLIP",
        freeze_vision: bool = True,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("Встановіть transformers: pip install transformers")

        logger.info(f"Завантаження StreetCLIP: {model_name}")
        clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Використовуємо лише візуальний енкодер
        self.vision_model  = clip.vision_model
        self.visual_projection = clip.visual_projection  # (hidden_size → projection_dim)
        self._embed_dim = clip.config.projection_dim  # Зазвичай 512

        if freeze_vision:
            self.freeze_vision_encoder()

        # Лінійна голова (або MLP)
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(self._embed_dim),
                nn.Linear(self._embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self._embed_dim),
                nn.Linear(self._embed_dim, num_classes),
            )

    def freeze_vision_encoder(self) -> None:
        """Заморожує CLIP vision encoder та projection."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.visual_projection.parameters():
            param.requires_grad = False
        logger.info("StreetCLIPModel: vision encoder заморожено")

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        """
        Розморожує останні n трансформерних шарів для тонкого налаштування.

        Аргументи:
            n: Кількість шарів для розморожування.
        """
        encoder_layers = self.vision_model.encoder.layers
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Також розморожуємо projection
        for param in self.visual_projection.parameters():
            param.requires_grad = True
        logger.info(f"StreetCLIPModel: розморожено {n} останніх шарів + projection")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Кодує зображення через CLIP vision encoder.

        Аргументи:
            pixel_values: Тензор (N, 3, H, W) передоброблений процесором.

        Повертає:
            L2-нормалізований embedding (N, embed_dim).
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output  # (N, hidden_size)
        projected = self.visual_projection(pooled)  # (N, projection_dim)
        return F.normalize(projected, dim=1)

    def get_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Аналог encode_image для сумісності з іншими моделями.

        Аргументи:
            pixel_values: Батч зображень (N, 3, H, W).

        Повертає:
            Нормалізовані embeddings (N, embed_dim).
        """
        return self.encode_image(pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід.

        Аргументи:
            pixel_values: Батч (N, 3, H, W).

        Повертає:
            Логіти (N, num_classes).
        """
        emb = self.encode_image(pixel_values)
        return self.head(emb)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        class_names: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Передбачення для зображення або батчу.

        Аргументи:
            x:           Зображення (3, H, W) або батч (N, 3, H, W).
            class_names: Список назв класів.
            top_k:       Кількість топ-передбачень.

        Повертає:
            Список словників [{'class': str, 'index': int, 'prob': float}, ...].
        """
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        probs  = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)

        results = []
        for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
            name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
            results.append({"class": name, "index": idx, "prob": round(prob, 6)})
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. GeoCLIPModel — CLIP ViT + GPS Location Encoder
# ──────────────────────────────────────────────────────────────────────────────

class RandomFourierGPSEncoder(nn.Module):
    """
    Кодувальник GPS-координат за допомогою Random Fourier Features (RFF).

    Перетворює (lat, lon) → Fourier embedding → MLP → вектор фіксованої довжини.
    RFF апроксимують ядро Гауса для GPS-координат на сфері.

    Аргументи:
        embed_dim:      Розмір вихідного embedding (512 за замовчуванням).
        num_frequencies: Кількість Fourier частот.
        sigma:           Масштаб Гауса (відповідає масштабу координат).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_frequencies: int = 256,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies

        # Фіксована матриця випадкових частот (не навчається)
        B = torch.randn(2, num_frequencies) * sigma
        self.register_buffer("B", B)

        # Розмір Fourier embedding: 2 * num_frequencies (sin + cos)
        fourier_dim = 2 * num_frequencies

        # MLP для перетворення Fourier features → embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Аргументи:
            coords: Тензор форми (N, 2) з (lat, lon) нормалізованими до [-1, 1].

        Повертає:
            L2-нормалізований embedding (N, embed_dim).
        """
        # Проєкція координат на простір частот
        proj = 2 * math.pi * coords @ self.B  # (N, num_frequencies)
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)  # (N, 2*F)

        emb = self.mlp(fourier)
        return F.normalize(emb, dim=1)


class GeoCLIPModel(nn.Module):
    """
    GeoCLIP: CLIP ViT image encoder + GPS Location Encoder.

    Контрастивне навчання у стилі SimCLR:
    - Image encoder (ViT-L/14 або openai/clip-vit-base-patch32) кодує зображення
    - GPS encoder (Random Fourier Features + MLP) кодує координати
    - InfoNCE loss наближує пари (image, gps) та відштовхує негативи

    Після навчання image encoder використовується для retrieval:
    given query image → знаходимо найближчий GPS з gallery.

    Аргументи:
        num_classes:     Кількість класів (для класифікаційної голови).
        clip_model_name: HuggingFace CLIP model ID.
        embed_dim:       Розмір спільного embedding-простору.
        gps_num_freqs:   Кількість Fourier частот GPS encoder.
        temperature:     Початкова температура для контрастивного loss.
        freeze_clip:     Якщо True, заморожує CLIP encoder.
    """

    def __init__(
        self,
        num_classes: int = 10,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        gps_num_freqs: int = 256,
        temperature: float = 0.07,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ── CLIP Vision Encoder ──────────────────────────────────────────────
        try:
            from transformers import CLIPModel
        except ImportError:
            raise ImportError("Встановіть transformers: pip install transformers")

        logger.info(f"Завантаження CLIP: {clip_model_name}")
        clip = CLIPModel.from_pretrained(clip_model_name)
        self.vision_model      = clip.vision_model
        self.visual_projection = clip.visual_projection
        self._clip_embed_dim   = clip.config.projection_dim

        if freeze_clip:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            for p in self.visual_projection.parameters():
                p.requires_grad = False
            logger.info("GeoCLIPModel: CLIP encoder заморожено")

        # Додаткова проєкція (якщо embed_dim ≠ CLIP projection_dim)
        if self._clip_embed_dim != embed_dim:
            self.img_proj = nn.Linear(self._clip_embed_dim, embed_dim, bias=False)
        else:
            self.img_proj = nn.Identity()

        # ── GPS Location Encoder ─────────────────────────────────────────────
        self.gps_encoder = RandomFourierGPSEncoder(
            embed_dim=embed_dim,
            num_frequencies=gps_num_freqs,
            sigma=0.1,
        )

        # ── Навчальна температура (логарифмічна) ────────────────────────────
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature))
        )

        # ── Класифікаційна голова (для supervised fine-tuning) ────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # ── GPS Gallery (для retrieval на inference) ─────────────────────────
        self.register_buffer("gallery_coords", torch.zeros(0, 2))
        self.register_buffer("gallery_embeddings", torch.zeros(0, embed_dim))

    # ──────────────────────────────────────────────────────────────────────
    # Нормалізація координат
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
        """
        Нормалізує GPS-координати до [-1, 1].

        Аргументи:
            coords: Тензор (N, 2) із (lat°, lon°).

        Повертає:
            Нормалізований тензор (N, 2).
        """
        lat = coords[:, 0:1] / 90.0
        lon = coords[:, 1:2] / 180.0
        return torch.cat([lat, lon], dim=1)

    # ──────────────────────────────────────────────────────────────────────
    # Кодування
    # ──────────────────────────────────────────────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Кодує зображення через CLIP + проєкцію.

        Аргументи:
            pixel_values: (N, 3, H, W).

        Повертає:
            L2-нормалізований embedding (N, embed_dim).
        """
        vision_out = self.vision_model(pixel_values=pixel_values)
        pooled     = vision_out.pooler_output
        proj       = self.visual_projection(pooled)
        proj       = self.img_proj(proj)
        return F.normalize(proj, dim=1)

    def encode_gps(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Кодує GPS-координати через RFF + MLP.

        Аргументи:
            coords: (N, 2) з (lat, lon) у градусах.

        Повертає:
            L2-нормалізований embedding (N, embed_dim).
        """
        coords_norm = self.normalize_coords(coords)
        return self.gps_encoder(coords_norm)

    def get_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Витягує image embeddings для t-SNE та retrieval.

        Аргументи:
            pixel_values: Батч (N, 3, H, W).

        Повертає:
            Embeddings (N, embed_dim).
        """
        return self.encode_image(pixel_values)

    # ──────────────────────────────────────────────────────────────────────
    # Контрастивний loss
    # ──────────────────────────────────────────────────────────────────────

    def contrastive_loss(
        self,
        img_emb: torch.Tensor,
        gps_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE (SimCLR-style) контрастивний loss.

        Максимізує cos-подібність між правильними парами (image_i, gps_i)
        та мінімізує для всіх інших пар у батчі.

        Аргументи:
            img_emb: Нормалізовані image embeddings (N, D).
            gps_emb: Нормалізовані GPS embeddings (N, D).

        Повертає:
            Скалярний loss.
        """
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        N = img_emb.shape[0]

        # Матриця подібностей (N, N)
        similarity = img_emb @ gps_emb.T * temperature  # (N, N)

        # Діагональні елементи — правильні пари (мітки 0, 1, ..., N-1)
        labels = torch.arange(N, device=img_emb.device)

        # Симетричний loss: image→GPS та GPS→image
        loss_img2gps = F.cross_entropy(similarity,   labels)
        loss_gps2img = F.cross_entropy(similarity.T, labels)

        return (loss_img2gps + loss_gps2img) / 2.0

    # ──────────────────────────────────────────────────────────────────────
    # Gallery для retrieval
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def build_gallery(
        self,
        coords_list: list[tuple[float, float]],
        batch_size: int = 256,
    ) -> None:
        """
        Будує GPS gallery: передобчислює embeddings для відомих локацій.

        Аргументи:
            coords_list: Список GPS-координат (lat, lon) для галереї.
            batch_size:  Розмір батчу для обчислення embeddings.
        """
        self.eval()
        coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
        all_embs = []

        for start in range(0, len(coords_tensor), batch_size):
            batch_coords = coords_tensor[start:start + batch_size]
            # Переносимо на пристрій моделі
            device = next(self.parameters()).device
            batch_coords = batch_coords.to(device)
            embs = self.encode_gps(batch_coords)
            all_embs.append(embs.cpu())

        self.gallery_coords     = coords_tensor
        self.gallery_embeddings = torch.cat(all_embs, dim=0)
        logger.info(f"Gallery побудовано: {len(coords_list)} локацій")

    @torch.no_grad()
    def retrieve_gps(
        self,
        pixel_values: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieval: знаходить топ-K GPS-локацій для зображення.

        Аргументи:
            pixel_values: Зображення (3, H, W) або батч (N, 3, H, W).
            top_k:        Кількість найближчих локацій.

        Повертає:
            Кортеж (coords, similarities):
            - coords:       (N, top_k, 2) — координати топ-K локацій
            - similarities: (N, top_k) — косинусна подібність
        """
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        img_emb = self.encode_image(pixel_values)  # (N, D)

        device = img_emb.device
        gallery_emb = self.gallery_embeddings.to(device)  # (G, D)

        sim = img_emb @ gallery_emb.T  # (N, G)
        top_sim, top_idx = sim.topk(min(top_k, gallery_emb.shape[0]), dim=1)  # (N, k)

        gallery_coords = self.gallery_coords.to(device)
        top_coords = gallery_coords[top_idx]  # (N, k, 2)

        return top_coords, top_sim

    # ──────────────────────────────────────────────────────────────────────
    # Forward / Predict
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Прямий прохід.

        Аргументи:
            pixel_values: Батч зображень (N, 3, H, W).
            coords:       GPS-координати (N, 2) — потрібні для contrastive loss.

        Повертає:
            Словник із ключами:
            - 'logits':          класифікаційні логіти (N, C)
            - 'image_embeddings': image embeddings (N, D)
            - 'gps_embeddings':  GPS embeddings (N, D) — якщо coords передано
            - 'contrastive_loss': скалярний contrastive loss — якщо coords передано
        """
        img_emb = self.encode_image(pixel_values)
        logits  = self.classifier(img_emb)

        output: dict[str, torch.Tensor] = {
            "logits":           logits,
            "image_embeddings": img_emb,
        }

        if coords is not None:
            gps_emb = self.encode_gps(coords)
            output["gps_embeddings"]   = gps_emb
            output["contrastive_loss"] = self.contrastive_loss(img_emb, gps_emb)

        return output

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        class_names: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Передбачення міста/класу для зображення.

        Аргументи:
            x:           Зображення (3, H, W) або батч (N, 3, H, W).
            class_names: Список назв класів.
            top_k:       Кількість топ-передбачень.

        Повертає:
            Список [{'class': str, 'index': int, 'prob': float}, ...].
        """
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        out    = self.forward(x)
        probs  = F.softmax(out["logits"], dim=1)
        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)

        results = []
        for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
            name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
            results.append({"class": name, "index": idx, "prob": round(prob, 6)})
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Фабрична функція
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Фабрична функція для створення моделі за назвою архітектури.

    Аргументи:
        architecture: 'baseline', 'streetclip' або 'geoclip'.
        num_classes:  Кількість класів.
        pretrained:   Якщо True, завантажує pretrained ваги.
        **kwargs:     Додаткові аргументи для конструктора моделі.

    Повертає:
        Ініціалізована torch.nn.Module.

    Приклад:
        >>> model = build_model('baseline', num_classes=10, pretrained=True)
    """
    arch = architecture.lower().strip()

    if arch in ("baseline", "baselinecnn", "efficientnet"):
        return BaselineCNN(num_classes=num_classes, pretrained=pretrained, **kwargs)

    elif arch in ("streetclip", "street_clip"):
        return StreetCLIPModel(num_classes=num_classes, **kwargs)

    elif arch in ("geoclip", "geo_clip"):
        return GeoCLIPModel(num_classes=num_classes, **kwargs)

    else:
        raise ValueError(
            f"Невідома архітектура: '{architecture}'. "
            f"Підтримуються: 'baseline', 'streetclip', 'geoclip'"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Тест при прямому запуску
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Тест архітектур моделей ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, C, H, W = 4, 3, 224, 224
    num_classes = 10

    dummy_images = torch.randn(N, C, H, W).to(device)
    dummy_coords = torch.tensor([
        [50.45, 30.52],
        [49.99, 36.23],
        [46.48, 30.72],
        [48.46, 35.05],
    ], dtype=torch.float32).to(device)

    # 1. BaselineCNN
    print("\n--- BaselineCNN ---")
    model1 = BaselineCNN(num_classes=num_classes, pretrained=True).to(device)
    model1.freeze_backbone()
    logits1 = model1(dummy_images)
    assert logits1.shape == (N, num_classes), f"Невірна форма: {logits1.shape}"
    emb1 = model1.get_embeddings(dummy_images)
    print(f"  Логіти: {logits1.shape}, Embeddings: {emb1.shape}")

    preds1 = model1.predict(dummy_images[0], class_names=[f"місто_{i}" for i in range(num_classes)])
    print(f"  Передбачення: {preds1[:2]}")
    print("  BaselineCNN — OK")

    # GPS encoder тест
    print("\n--- RandomFourierGPSEncoder ---")
    gps_enc = RandomFourierGPSEncoder(embed_dim=512, num_frequencies=256).to(device)
    coords_norm = dummy_coords / torch.tensor([[90.0, 180.0]], device=device)
    gps_emb = gps_enc(coords_norm)
    assert gps_emb.shape == (N, 512)
    print(f"  GPS embeddings: {gps_emb.shape} — OK")

    print("\nВсі базові тести пройдено!")
    print("(StreetCLIPModel та GeoCLIPModel потребують HuggingFace credentials/інтернет)")
