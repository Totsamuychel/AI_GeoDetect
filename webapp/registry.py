"""
registry.py — Реєстр моделей для веб-демо геолокації.

Ліниво завантажує три навчені моделі (baseline / streetclip / geoclip),
виконує інференс одного зображення та визначає, чи належить фото до одного
з чотирьох міст (Київ / Варшава / Прага / Будапешт), чи є поза-розподільним (OOD).

OOD-детекція: гейт на основі косинусної подібності до прототипів класів
в embedding-просторі моделі, відкалібрований на TRAIN-наборі (без витоку
тест-даних). Це чесний сигнал «фото не з цих міст», якого softmax над трьома
класами сам по собі дати не може (він завжди впевнений в одному з трьох).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

# Додаємо code/ до sys.path, щоб імпортувати моделі та утиліти проєкту.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from augmentations import get_norm_for, get_val_transforms  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402
from utils import get_device  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("registry")

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Конфігурація моделей та міст
# ──────────────────────────────────────────────────────────────────────────────

# Кожна модель: шлях до чекпоінту, рідний розмір входу (як при навчанні).
MODELS: dict[str, dict] = {
    "streetclip": {
        "checkpoint": ROOT / "checkpoints/streetclip_v2/best_model.pth",
        "img_size":   336,
        "label":      "StreetCLIP",
        "subtitle":   "CLIP ViT-L/14@336 + лінійний пробник",
        "accuracy":   "89.4%",
    },
    "baseline": {
        "checkpoint": ROOT / "checkpoints/baseline_v2/best_model.pth",
        "img_size":   260,
        "label":      "Baseline CNN",
        "subtitle":   "EfficientNet-B2",
        "accuracy":   "71.5%",
    },
    "geoclip": {
        "checkpoint": ROOT / "checkpoints/geoclip_v2/best_model.pth",
        "img_size":   224,
        "label":      "GeoCLIP",
        "subtitle":   "CLIP ViT-B/32 + GPS-енкодер",
        "accuracy":   "83.8%",
    },
}

# Українські назви та координати центрів міст (ключі lowercase = class_names).
CITY_INFO: dict[str, dict] = {
    "kyiv":     {"ua": "Київ",     "country": "Україна",  "lat": 50.4501, "lon": 30.5234},
    "warsaw":   {"ua": "Варшава",  "country": "Польща",   "lat": 52.2297, "lon": 21.0122},
    "prague":   {"ua": "Прага",    "country": "Чехія",    "lat": 50.0755, "lon": 14.4378},
    "budapest": {"ua": "Будапешт", "country": "Угорщина", "lat": 47.4979, "lon": 19.0402},
}

# OOD-гейт: Mahalanobis-відстань в embedding-просторі моделі до найближчого
# класу (об'єднана коваріація зі shrinkage). На бенчмарку (StreetCLIP, негативи
# osv5m/Румунія) Mahalanobis дав AUROC 0.906 проти 0.845 у косинус-прототипів.
# Скільки фото на місто брати для калібрування.
OOD_SAMPLES_PER_CITY = 400
# Частка калібрувальних embedding-ів, відкладена для виставлення порога (решта
# йде на оцінку середніх/коваріації). Поріг ставимо на цільовий FPR.
OOD_CAL_HOLDOUT = 0.25
# Цільовий FPR: який відсоток справжніх фото міст дозволяємо хибно позначити як
# OOD. Компроміс: 10% дає ~75%+ вилову на «важких» сусідніх країнах (Румунія) і
# майже 100% на явно чужих фото (інші континенти/приміщення/меми). Дилка одним
# числом: менше → менше хибних тривог, але гірший вилов; більше → навпаки.
OOD_FPR_TARGET = 10.0
# Shrinkage коваріації (стабілізує обернення для D≈768..1408 при ~N образцях).
OOD_COV_SHRINKAGE = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# Завантажена модель + стан OOD
# ──────────────────────────────────────────────────────────────────────────────

class _LoadedModel:
    def __init__(self, model, class_names, arch, img_size, transform):
        self.model = model
        self.class_names = class_names
        self.arch = arch
        self.img_size = img_size
        self.transform = transform
        # OOD-калібрування (Mahalanobis; заповнюється у _calibrate_ood)
        self.ood_means: Optional[np.ndarray] = None      # (C, D) середні класів
        self.ood_cov_inv: Optional[np.ndarray] = None    # (D, D) обернена коваріація
        self.ood_threshold: Optional[float] = None       # поріг на score=-min_c maha


class ModelRegistry:
    """Тримає завантажені моделі та виконує інференс. Потокобезпечний."""

    def __init__(self):
        self.device = get_device()
        self._loaded: dict[str, _LoadedModel] = {}
        self._lock = threading.Lock()  # серіалізує GPU-доступ
        logger.info(f"Пристрій інференсу: {self.device}")

    # ── Перелік доступних моделей ────────────────────────────────────────────
    def available(self) -> list[dict]:
        out = []
        for arch, cfg in MODELS.items():
            out.append({
                "id":        arch,
                "label":     cfg["label"],
                "subtitle":  cfg["subtitle"],
                "accuracy":  cfg["accuracy"],
                "available": cfg["checkpoint"].exists(),
                "loaded":    arch in self._loaded,
            })
        return out

    # ── Ліниве завантаження ──────────────────────────────────────────────────
    def _ensure_loaded(self, arch: str) -> _LoadedModel:
        if arch in self._loaded:
            return self._loaded[arch]
        if arch not in MODELS:
            raise ValueError(f"Невідома модель: {arch}")

        cfg = MODELS[arch]
        ckpt = cfg["checkpoint"]
        if not ckpt.exists():
            raise FileNotFoundError(f"Чекпоінт не знайдено: {ckpt}")

        logger.info(f"Завантаження моделі «{arch}» з {ckpt.name} …")
        model, class_names, _config, _meta = load_checkpoint(str(ckpt), self.device)
        model.eval()

        mean, std = get_norm_for(arch)
        transform = get_val_transforms(img_size=cfg["img_size"], mean=mean, std=std)

        lm = _LoadedModel(model, class_names, arch, cfg["img_size"], transform)
        self._calibrate_ood(lm)
        self._loaded[arch] = lm
        logger.info(f"Модель «{arch}» готова ({len(class_names)} класів).")
        return lm

    # ── Витяг embedding (нормалізований) + логіти ────────────────────────────
    def _infer_tensor(self, lm: _LoadedModel, tensor: torch.Tensor):
        """Повертає (probs: np[C], emb: torch (N,D) нормалізований)."""
        model, arch = lm.model, lm.arch
        with torch.no_grad():
            if arch == "baseline":
                emb = model.get_embeddings(tensor)      # вже L2-нормалізований
                logits = model(tensor)
            elif arch == "streetclip":
                emb = model.encode_image(tensor)        # нормалізований
                logits = model.head(emb)
            elif arch == "geoclip":
                emb = model.encode_image(tensor)
                logits = model.classifier(emb)
            else:
                raise ValueError(arch)
            probs = F.softmax(logits, dim=1)[0].float().cpu().numpy()
        return probs, emb

    # ── Калібрування OOD (Mahalanobis) на TRAIN ───────────────────────────────
    def _calibrate_ood(self, lm: _LoadedModel) -> None:
        cache_file = CACHE_DIR / f"ood_{lm.arch}.npz"
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                if list(data["class_names"]) == list(lm.class_names):
                    lm.ood_means = data["means"].astype(np.float32)
                    lm.ood_cov_inv = data["cov_inv"].astype(np.float32)
                    lm.ood_threshold = float(data["threshold"])
                    logger.info(
                        f"OOD-кеш «{lm.arch}» завантажено "
                        f"(поріг Mahalanobis-score={lm.ood_threshold:.2f})."
                    )
                    return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Не вдалося прочитати OOD-кеш {cache_file}: {e}")

        manifest = ROOT / "dataset/manifests_sv/train.csv"
        if not manifest.exists():
            logger.warning(
                f"TRAIN-маніфест відсутній — OOD-гейт для «{lm.arch}» вимкнено "
                f"(буде використано лише поріг впевненості softmax)."
            )
            return

        try:
            import pandas as pd
            df = pd.read_csv(manifest, low_memory=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Не вдалося прочитати маніфест: {e}; OOD вимкнено.")
            return

        logger.info(f"Калібрування OOD (Mahalanobis) для «{lm.arch}» (разова операція)…")
        per_class_embs: dict[str, list[np.ndarray]] = {c: [] for c in lm.class_names}

        for city in lm.class_names:
            sub = df[df["city"].astype(str).str.lower() == city.lower()].head(OOD_SAMPLES_PER_CITY)
            batch = []
            for _, row in sub.iterrows():
                # filepath у v2-маніфесті відносний до кореня проєкту
                # (data/images/...), у старих — відносний до dataset/.
                fp = str(row["filepath"])
                p = ROOT / fp
                if not p.exists():
                    p = ROOT / "dataset" / fp
                if not p.exists():
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                batch.append(lm.transform(img))
                if len(batch) == 16:
                    self._accumulate_embs(lm, batch, per_class_embs[city])
                    batch = []
            if batch:
                self._accumulate_embs(lm, batch, per_class_embs[city])
            logger.info(f"  {city}: {len(per_class_embs[city])} embedding-ів")

        if not all(len(v) >= 20 for v in per_class_embs.values()):
            logger.warning(f"Замало зображень для калібрування «{lm.arch}» — OOD вимкнено.")
            return

        # Розбиваємо кожен клас на fit (середні+коваріація) та cal (поріг на FPR).
        rng = np.random.default_rng(42)
        fit_embs, fit_lbl, cal_embs = [], [], []
        for ci, c in enumerate(lm.class_names):
            arr = np.stack(per_class_embs[c])
            idx = rng.permutation(len(arr))
            n_hold = max(5, int(len(arr) * OOD_CAL_HOLDOUT))
            cal_embs.append(arr[idx[:n_hold]])
            fit = arr[idx[n_hold:]]
            fit_embs.append(fit); fit_lbl.append(np.full(len(fit), ci))
        X = np.concatenate(fit_embs).astype(np.float64)
        y = np.concatenate(fit_lbl)
        Xcal = np.concatenate(cal_embs).astype(np.float64)
        D = X.shape[1]

        # Середні класів + об'єднана коваріація зі shrinkage (стабільне обернення).
        means = np.stack([X[y == i].mean(0) for i in range(len(lm.class_names))])  # (C,D)
        Z = X - means[y]
        cov = (Z.T @ Z) / len(X)
        cov = (1 - OOD_COV_SHRINKAGE) * cov + OOD_COV_SHRINKAGE * (np.trace(cov) / D) * np.eye(D)
        cov_inv = np.linalg.pinv(cov)

        def maha_score(E):
            diff = E[:, None, :] - means[None, :, :]            # (N,C,D)
            md = np.einsum("ncd,de,nce->nc", diff, cov_inv, diff)
            return -md.min(1)                                   # вище = in-dist

        # Поріг = OOD_FPR_TARGET-перцентиль score на відкладеному cal-наборі.
        threshold = float(np.percentile(maha_score(Xcal), OOD_FPR_TARGET))

        lm.ood_means = means.astype(np.float32)
        lm.ood_cov_inv = cov_inv.astype(np.float32)
        lm.ood_threshold = threshold

        np.savez_compressed(
            cache_file,
            class_names=np.array(lm.class_names),
            means=lm.ood_means, cov_inv=lm.ood_cov_inv,
            threshold=np.array(threshold, dtype=np.float64),
        )
        logger.info(
            f"OOD «{lm.arch}» відкалібровано (Mahalanobis): поріг score={threshold:.2f}, "
            f"D={D}, цільовий FPR={OOD_FPR_TARGET}% (кеш збережено)."
        )

    def _accumulate_embs(self, lm, batch_tensors, sink: list) -> None:
        t = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            if lm.arch == "baseline":
                emb = lm.model.get_embeddings(t)
            else:
                emb = lm.model.encode_image(t)
        for row in emb.float().cpu().numpy():
            sink.append(row)

    # ── Публічний інференс ─────────────────────────────────────────────────────
    def predict(self, arch: str, image: Image.Image, fallback: bool = True) -> dict:
        with self._lock:
            lm = self._ensure_loaded(arch)
            tensor = lm.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            probs, emb = self._infer_tensor(lm, tensor)

            # OOD: Mahalanobis-score = -min_c відстань до класу в emb-просторі.
            # Вище score → ближче до розподілу міст; нижче за поріг → OOD.
            ood_score = None
            is_ood = False
            ood_enabled = (
                lm.ood_means is not None
                and lm.ood_cov_inv is not None
                and lm.ood_threshold is not None
            )
            if ood_enabled:
                e = emb[0].float().cpu().numpy().astype(np.float32)       # (D,)
                diff = e[None, :] - lm.ood_means                         # (C,D)
                md = np.einsum("cd,de,ce->c", diff, lm.ood_cov_inv, diff)
                ood_score = float(-md.min())
                is_ood = ood_score < lm.ood_threshold

        # Сортуємо передбачення за ймовірністю.
        order = np.argsort(-probs)
        predictions = []
        for i in order:
            city = lm.class_names[i]
            info = CITY_INFO.get(city.lower(), {})
            predictions.append({
                "city":     city,
                "city_ua":  info.get("ua", city),
                "country":  info.get("country", ""),
                "prob":     float(probs[i]),
                "lat":      info.get("lat"),
                "lon":      info.get("lon"),
            })

        result = {
            "model":       arch,
            "predictions": predictions,
            "ood": {
                "is_ood":         bool(is_ood),
                "method":         "mahalanobis",
                "score":          None if ood_score is None else round(ood_score, 2),
                "max_similarity": None if ood_score is None else round(ood_score, 2),  # back-compat
                "threshold":      None if lm.ood_threshold is None else round(lm.ood_threshold, 2),
                "enabled":        ood_enabled,
            },
        }

        # #5 OOD-fallback: якщо гейт спрацював, додаємо підказку від GeoCLIP
        # (інша архітектура → інший погляд). Лок уже відпущено, тож виклик
        # predict('geoclip') безпечний; fallback=False, щоб не зациклитись.
        if fallback and is_ood and arch != "geoclip" and "geoclip" in MODELS:
            try:
                g = self.predict("geoclip", image, fallback=False)
                gtop = g["predictions"][0]
                result["ood"]["fallback"] = {
                    "model":    "geoclip",
                    "city":     gtop["city"],
                    "city_ua":  gtop["city_ua"],
                    "prob":     gtop["prob"],
                    "is_ood":   g["ood"]["is_ood"],
                }
            except Exception as e:  # noqa: BLE001
                logger.warning(f"GeoCLIP-fallback не вдався: {e}")

        return result

    # ── #4 Attention-rollout (де «дивилась» ViT) ──────────────────────────────
    def explain(self, arch: str, image: Image.Image) -> dict:
        """Карта уваги ViT (attention-rollout) для CLIP-моделей. Повертає
        base64-overlay поверх вхідного фото."""
        if arch not in ("streetclip", "geoclip"):
            return {"available": False,
                    "reason": "Карта уваги доступна лише для ViT-моделей (StreetCLIP / GeoCLIP)."}
        with self._lock:
            lm = self._ensure_loaded(arch)
            tensor = lm.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            vm = lm.model.vision_model
            # transformers 5.x за замовчуванням використовує SDPA, який НЕ
            # повертає ваги уваги. Тимчасово вмикаємо eager-attention.
            prev_impl = getattr(vm.config, "_attn_implementation", None)
            try:
                vm.config._attn_implementation = "eager"
                with torch.no_grad():
                    out = vm(pixel_values=tensor, output_attentions=True)
            finally:
                if prev_impl is not None:
                    vm.config._attn_implementation = prev_impl
            atts = out.attentions or ()                # tuple(L) of (B,H,T,T)
            if not atts:
                return {"available": False,
                        "reason": "Не вдалося отримати ваги уваги (SDPA backend)."}
            T = atts[0].size(-1)
            roll = torch.eye(T, device=self.device)
            for a in atts:
                a = a.mean(1)[0]                       # head-mean (T,T)
                a = a + torch.eye(T, device=self.device)
                a = a / a.sum(-1, keepdim=True)
                roll = a @ roll
            cls = roll[0, 1:]                          # CLS → патчі
            g = int(round(cls.numel() ** 0.5))
            grid = cls[: g * g].reshape(g, g).float().cpu().numpy()
        return {"available": True, "grid_size": g,
                "heatmap": self._heatmap_overlay(image, grid)}

    def _heatmap_overlay(self, image: Image.Image, grid: np.ndarray) -> str:
        import base64
        import io
        img = image.convert("RGB")
        W, H = img.size
        g = grid - grid.min()
        g = g / (g.max() + 1e-8)
        heat = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        heat = np.asarray(heat).astype(np.float32) / 255.0
        rgba = self._colormap_rgba(heat)
        comp = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(rgba, "RGBA"))
        buf = io.BytesIO()
        comp.convert("RGB").save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _colormap_rgba(heat: np.ndarray) -> np.ndarray:
        """(H,W) у [0,1] → (H,W,4) uint8 RGBA «jet»-палітрою (без matplotlib).
        alpha ∝ інтенсивність, тож холодні зони майже прозорі."""
        v = np.clip(heat, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
        H, W = heat.shape
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0] = (r * 255).astype(np.uint8)
        rgba[..., 1] = (g * 255).astype(np.uint8)
        rgba[..., 2] = (b * 255).astype(np.uint8)
        rgba[..., 3] = (v ** 0.8 * 200).astype(np.uint8)         # alpha
        return rgba

    # ── Запуск усіх моделей одразу (#6 «порівняти всі») ───────────────────────
    def predict_all(self, image: Image.Image) -> dict:
        results = []
        for arch, cfg in MODELS.items():
            if not cfg["checkpoint"].exists():
                continue
            try:
                results.append(self.predict(arch, image, fallback=False))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"predict_all: модель «{arch}» впала: {e}")
                results.append({"model": arch, "error": str(e)})
        return {"results": results}


# Глобальний синглтон
_REGISTRY: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY


def decode_image(data_url: str) -> Image.Image:
    """Декодує data-URL (base64) або чистий base64 у PIL.Image."""
    import base64
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(BytesIO(raw))
