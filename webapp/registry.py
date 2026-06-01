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

# Скільки фото на місто брати для калібрування OOD-прототипів.
OOD_SAMPLES_PER_CITY = 300
# Перцентиль in-distribution подібностей як поріг (нижче → OOD).
# 1 = відсікаємо лише ~1% найнетиповіших справжніх фото — це майже не дає
# хибних спрацювань на справжніх фото міст (зокрема туристичних/нетипових
# ракурсів), але все одно відсіює явно чужі знімки.
OOD_PERCENTILE = 1.0
# Додатковий запас: поріг ще трохи знижуємо, щоб справжні фото міст, які
# відрізняються від «вуличного» розподілу навчання, не позначались як OOD.
OOD_MARGIN = 0.90


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
        # OOD-калібрування (заповнюється у _calibrate_ood)
        self.prototypes: Optional[torch.Tensor] = None  # (C, D) нормалізовані
        self.sim_threshold: Optional[float] = None


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

    # ── Калібрування OOD-прототипів на TRAIN ──────────────────────────────────
    def _calibrate_ood(self, lm: _LoadedModel) -> None:
        cache_file = CACHE_DIR / f"ood_{lm.arch}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if data.get("class_names") == lm.class_names:
                    lm.prototypes = torch.tensor(
                        data["prototypes"], dtype=torch.float32, device=self.device
                    )
                    lm.sim_threshold = float(data["sim_threshold"])
                    logger.info(
                        f"OOD-кеш «{lm.arch}» завантажено "
                        f"(поріг подібності={lm.sim_threshold:.3f})."
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

        logger.info(f"Калібрування OOD для «{lm.arch}» (це разова операція)…")
        per_class_embs: dict[str, list[np.ndarray]] = {c: [] for c in lm.class_names}
        idx_of = {c: i for i, c in enumerate(lm.class_names)}

        for city in lm.class_names:
            sub = df[df["city"].astype(str).str.lower() == city.lower()]
            sub = sub.head(OOD_SAMPLES_PER_CITY)
            batch, paths = [], []
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
                paths.append(p)
                if len(batch) == 16:
                    self._accumulate_embs(lm, batch, per_class_embs[city])
                    batch = []
            if batch:
                self._accumulate_embs(lm, batch, per_class_embs[city])
            logger.info(f"  {city}: {len(per_class_embs[city])} embedding-ів")

        # Прототипи = середній нормалізований embedding кожного класу.
        protos = torch.zeros(len(lm.class_names), 0)
        proto_list = []
        all_own_sims: list[float] = []
        valid = all(len(v) >= 10 for v in per_class_embs.values())
        if not valid:
            logger.warning(f"Замало зображень для калібрування «{lm.arch}» — OOD вимкнено.")
            return

        # Будуємо прототипи
        proto_tensors = []
        for c in lm.class_names:
            arr = np.stack(per_class_embs[c])  # (n, D)
            proto = arr.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            proto_tensors.append(proto)
        protos_np = np.stack(proto_tensors)  # (C, D)

        # Власні подібності кожного train-embedding до прототипу свого класу.
        for c in lm.class_names:
            arr = np.stack(per_class_embs[c])
            sims = arr @ protos_np[idx_of[c]]  # (n,)
            all_own_sims.extend(sims.tolist())

        sim_threshold = float(np.percentile(all_own_sims, OOD_PERCENTILE)) * OOD_MARGIN
        lm.prototypes = torch.tensor(protos_np, dtype=torch.float32, device=self.device)
        lm.sim_threshold = sim_threshold

        cache_file.write_text(json.dumps({
            "class_names":   lm.class_names,
            "prototypes":    protos_np.tolist(),
            "sim_threshold": sim_threshold,
        }), encoding="utf-8")
        logger.info(
            f"OOD «{lm.arch}» відкалібровано: поріг подібності={sim_threshold:.3f} "
            f"(кеш збережено)."
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
    def predict(self, arch: str, image: Image.Image) -> dict:
        with self._lock:
            lm = self._ensure_loaded(arch)
            tensor = lm.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            probs, emb = self._infer_tensor(lm, tensor)

            # OOD: максимальна косинусна подібність до прототипів класів.
            max_sim = None
            is_ood = False
            if lm.prototypes is not None and lm.sim_threshold is not None:
                sims = (emb @ lm.prototypes.T)[0].float().cpu().numpy()  # (C,)
                max_sim = float(sims.max())
                is_ood = max_sim < lm.sim_threshold

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
                "is_ood":        bool(is_ood),
                "max_similarity": None if max_sim is None else round(max_sim, 4),
                "threshold":      None if lm.sim_threshold is None else round(lm.sim_threshold, 4),
                "enabled":        lm.prototypes is not None,
            },
        }
        return result


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
