"""
train.py — Повний цикл навчання для геолокаційних моделей.

Реалізує:
- AdamW оптимізатор із CosineAnnealingLR планувальником
- Двоетапне навчання: заморожений backbone → розморожені останні шари
- Early stopping з налаштовуваною терпимістю
- W&B або MLflow логування (вмикається через конфіг)
- Збереження найкращого чекпоінту за val_loss
- Mixed precision (torch.cuda.amp)

Запуск:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/streetclip.yaml --no-wandb
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from augmentations import get_train_transforms, get_val_transforms, get_norm_for
from dataset import GeoDataset, create_dataloaders
from metrics import top_k_accuracy_torch, macro_f1, balanced_accuracy
from models import build_model
from utils import get_device, seed_everything, count_parameters, format_param_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Конфігурація навчання
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Конфігурація повного процесу навчання."""

    # Дані
    manifest_path:     str   = "dataset/raw/mapillary/manifest.csv"
    image_root:        str   = "dataset"
    countries:         list  = field(default_factory=lambda: ["PL", "CZ", "HU"])
    quality_threshold: float = 0.4
    split_method:      str   = "h3"       # 'h3', 'kmeans' або 'prebuilt'
    h3_resolution:     int   = 4          # Рівень деталізації H3 (3-9)
    train_frac:        float = 0.7        # використовується лише для h3/kmeans
    val_frac:          float = 0.15       # використовується лише для h3/kmeans
    img_size:          int   = 224

    # Архітектура
    architecture:      str   = "baseline"  # 'baseline', 'streetclip', 'geoclip'
    pretrained:        bool  = True

    # Навчання — Стадія 1 (заморожений backbone)
    stage1_epochs:     int   = 10
    stage1_lr:         float = 1e-3
    stage1_unfreeze_n: int   = 0       # 0 = повністю заморожений backbone

    # Навчання — Стадія 2 (розморожені шари)
    stage2_epochs:     int   = 20
    stage2_lr:         float = 1e-4
    stage2_unfreeze_n: int   = 3       # Кількість блоків/шарів для розморожування

    # Оптимізатор / регуляризатор
    weight_decay:      float = 0.01
    batch_size:        int   = 32
    num_workers:       int   = 4
    grad_clip:         float = 1.0     # Максимальна норма градієнтів
    contrastive_loss_weight: float = 0.1  # вага InfoNCE-лоса (лише GeoCLIP)

    # Early stopping
    patience:          int   = 7
    min_delta:         float = 1e-4

    # Логування
    use_wandb:         bool  = False
    use_mlflow:        bool  = False
    wandb_project:     str   = "geolocation-warsaw-prague-budapest"
    wandb_run_name:    str   = ""
    mlflow_uri:        str   = "mlruns"
    experiment_name:   str   = "geolocation"

    # Збереження
    checkpoint_dir:    str   = "checkpoints"
    save_top_k:        int   = 3      # Зберігати K найкращих чекпоінтів

    # Відтворюваність
    seed:              int   = 42
    mixed_precision:   bool  = True
    cudnn_benchmark:   bool  = False  # True = швидше при фіксованому розмірі входу
    prefetch_factor:   int   = 4      # буфер DataLoader на воркер

    def __post_init__(self) -> None:
        # Not a dataclass field → excluded from asdict / checkpoint serialisation.
        self.progress_callback: Optional[Callable[[dict], None]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Механізм ранньої зупинки навчання.

    Зупиняє навчання, якщо validation loss не покращується протягом `patience` епох.

    Аргументи:
        patience:   Кількість епох очікування без покращення.
        min_delta:  Мінімальне абсолютне покращення для рахування.
        mode:       'min' (менше = краще) або 'max' (більше = краще).
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.reset()

    def reset(self) -> None:
        """Скидає стан ранньої зупинки для нового етапу навчання."""
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Оновлює стан та перевіряє умову зупинки.

        Аргументи:
            value: Поточне значення метрики.

        Повертає:
            True якщо слід зупинити навчання.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping після {self.patience} епох без покращення. "
                    f"Найкраще значення: {self.best_value:.6f}"
                )

        return self.should_stop


# ──────────────────────────────────────────────────────────────────────────────
# Менеджер чекпоінтів
# ──────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Управляє збереженням та завантаженням чекпоінтів моделі.

    Зберігає top-K найкращих чекпоінтів за val_loss.

    Аргументи:
        checkpoint_dir: Директорія для збереження чекпоінтів.
        save_top_k:     Кількість найкращих чекпоінтів.
        mode:           'min' (val_loss) або 'max' (val_acc).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_top_k: int = 3,
        mode: str = "min",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.mode = mode
        self._checkpoints: list[tuple[float, Path]] = []  # (value, path)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        val_loss: float,
        val_acc: float,
        config: TrainConfig,
        class_names: list[str],
    ) -> Path:
        """
        Зберігає чекпоінт, якщо він кращий за поточні збережені.

        Аргументи:
            model:       Модель PyTorch.
            optimizer:   Оптимізатор.
            scheduler:   LR scheduler.
            epoch:       Номер поточної епохи.
            val_loss:    Validation loss.
            val_acc:     Validation accuracy.
            config:      Конфігурація навчання.
            class_names: Список назв класів.

        Повертає:
            Шлях до збереженого файлу.
        """
        filename = f"epoch{epoch:03d}_valloss{val_loss:.4f}_valacc{val_acc:.4f}.pth"
        path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch":       epoch,
            "val_loss":    val_loss,
            "val_acc":     val_acc,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "config":      asdict(config),
            "class_names": class_names,
        }
        import shutil
        tmp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        shutil.move(str(tmp_path), str(path))

        value = val_loss if self.mode == "min" else val_acc
        self._checkpoints.append((value, path))

        # Сортування: для 'min' — зростаючий (гірші спереду), для 'max' — спадаючий
        reverse = (self.mode == "max")
        self._checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Видалення зайвих чекпоінтів
        while len(self._checkpoints) > self.save_top_k:
            worst_value, worst_path = self._checkpoints.pop()
            if worst_path.exists():
                worst_path.unlink()
                logger.debug(f"Видалено старий чекпоінт: {worst_path.name}")

        # Копія найкращого чекпоінту у фіксований файл best_model.pth
        best_path = self.checkpoint_dir / "best_model.pth"
        best_checkpoint = min(self._checkpoints, key=lambda x: x[0]) if self.mode == "min" \
                          else max(self._checkpoints, key=lambda x: x[0])
        if best_checkpoint[1].exists():
            import shutil
            shutil.copy2(best_checkpoint[1], best_path)

        logger.info(f"Чекпоінт збережено: {filename}")
        return path

    @staticmethod
    def load(path: str, device: torch.device) -> dict:
        """
        Завантажує чекпоінт із диску.

        Аргументи:
            path:   Шлях до .pth файлу.
            device: Пристрій для завантаження.

        Повертає:
            Словник чекпоінту.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        logger.info(
            f"Чекпоінт завантажено: {Path(path).name} "
            f"(epoch={checkpoint.get('epoch', '?')}, "
            f"val_loss={checkpoint.get('val_loss', '?'):.4f})"
        )
        return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# W&B та MLflow логування
# ──────────────────────────────────────────────────────────────────────────────

class Logger:
    """
    Уніфікований логер для W&B та MLflow.

    Аргументи:
        config: Конфігурація навчання.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._wandb_run = None
        self._mlflow_run = None

        if config.use_wandb:
            self._init_wandb()
        if config.use_mlflow:
            self._init_mlflow()

    def _init_wandb(self) -> None:
        try:
            import wandb
            run_name = self.config.wandb_run_name or \
                       f"{self.config.architecture}_{self.config.stage1_epochs + self.config.stage2_epochs}ep"
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=asdict(self.config),
                reinit="finish_previous",  # wandb ≥0.16: bool reinit застарів
            )
            logger.info(f"W&B ініціалізовано: {self.config.wandb_project}/{run_name}")
        except ImportError:
            logger.warning("wandb не встановлено. Пропускаємо W&B логування.")
            self.config.use_wandb = False

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._mlflow_run = mlflow.start_run()
            mlflow.log_params(asdict(self.config))
            logger.info(f"MLflow ініціалізовано: {self.config.experiment_name}")
        except ImportError:
            logger.warning("mlflow не встановлено. Пропускаємо MLflow логування.")
            self.config.use_mlflow = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Логує метрики на поточному кроці."""
        if self.config.use_wandb and self._wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.debug(f"W&B log помилка: {e}")

        if self.config.use_mlflow:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.debug(f"MLflow log помилка: {e}")

    def finish(self) -> None:
        """Завершує логування."""
        if self.config.use_wandb and self._wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        if self.config.use_mlflow:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Функції навчання / валідації
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    grad_clip: float = 1.0,
    use_amp: bool = True,
    architecture: str = "baseline",
    amp_dtype: torch.dtype = torch.float16,
    contrastive_weight: float = 0.1,
    _cb: Optional[Callable] = None,
    _cb_epoch: int = 0,
    _cb_stage: int = 1,
    _cb_total_epochs: int = 1,
) -> dict[str, float]:
    """
    Навчання моделі протягом однієї епохи.

    Аргументи:
        model:       Модель PyTorch.
        loader:      DataLoader навчального набору.
        optimizer:   Оптимізатор.
        criterion:   Функція втрат.
        device:      Пристрій обчислень.
        scaler:      GradScaler для AMP.
        grad_clip:   Максимальна норма градієнтів.
        use_amp:     Чи використовувати mixed precision.
        architecture: Назва архітектури (для обробки специфічного forward).

    Повертає:
        Словник {'loss': float, 'acc': float}.
    """
    model.train()
    total_loss = 0.0
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = 0

    for batch_idx, (images, labels, coords) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Для GeoCLIP потрібні координати; GeoDataset вже повертає тензор (N, 2).
        if architecture == "geoclip":
            coords = coords.to(device, non_blocking=True).float()
        else:
            coords = None

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if architecture == "geoclip" and coords is not None:
                output = model(images, coords=coords)
                logits = output["logits"]
                cls_loss = criterion(logits, labels)
                # Комбінований loss: класифікація + контрастивний
                contrastive = output.get("contrastive_loss", torch.tensor(0.0, device=device))
                loss = cls_loss + contrastive_weight * contrastive
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss ({loss.item()}) на батчі {batch_idx}. "
                f"Зупинка для діагностики (lr/AMP/дані)."
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        batch_size = images.size(0)
        total_loss    += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum()
        total_samples += batch_size

        if _cb:
            _cb({
                "type":          "batch",
                "stage":         _cb_stage,
                "epoch":         _cb_epoch,
                "total_epochs":  _cb_total_epochs,
                "batch":         batch_idx + 1,
                "total_batches": len(loader),
                "running_loss":  total_loss / max(total_samples, 1),
                "running_acc":   total_correct.item() / max(total_samples, 1),
            })

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_samples
            avg_acc  = total_correct.item() / total_samples
            logger.debug(
                f"  Batch [{batch_idx+1}/{len(loader)}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc":  total_correct.item() / max(total_samples, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    architecture: str = "baseline",
    amp_dtype: torch.dtype = torch.float16,
    contrastive_weight: float = 0.1,
) -> dict[str, float]:
    """
    Валідація моделі на валідаційному або тестовому наборі.

    Аргументи:
        model:       Модель PyTorch.
        loader:      DataLoader валідаційного набору.
        criterion:   Функція втрат.
        device:      Пристрій.
        use_amp:     Чи використовувати mixed precision.
        architecture: Назва архітектури.

    Повертає:
        Словник {'loss': float, 'top1_acc': float, 'top5_acc': float}.
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for images, labels, coords in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if architecture == "geoclip":
                # Той самий складений loss, що і в train — інакше val_loss
                # та early stopping несумісні між стадіями.
                coords = coords.to(device, non_blocking=True).float()
                output = model(images, coords=coords)
                logits = output["logits"]
                contrastive = output.get(
                    "contrastive_loss", torch.tensor(0.0, device=device)
                )
                # InfoNCE з N<2 вироджений (loss≈0) і зміщує val_loss /
                # early stopping — додаємо контрастивний член лише при N≥2.
                if images.size(0) >= 2:
                    loss = criterion(logits, labels) + contrastive_weight * contrastive
                else:
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    top1 = top_k_accuracy_torch(all_logits, all_labels, k=1).item()
    # top-5 безглуздий при малій кількості класів (=1.0); macro-F1 /
    # balanced accuracy інформативні й захищувані для диплома.
    mf1 = macro_f1(all_logits, all_labels)
    bacc = balanced_accuracy(all_logits, all_labels)
    n = len(all_labels)

    return {
        "loss":         total_loss / max(n, 1),
        "top1_acc":     top1,
        "macro_f1":     mf1,
        "balanced_acc": bacc,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Повний цикл навчання
# ──────────────────────────────────────────────────────────────────────────────

def build_warmup_cosine(optimizer, n_epochs: int, eta_min: float):
    """LinearLR-warmup (≈10% епох) → CosineAnnealingLR. Крок раз на епоху."""
    warmup = max(1, int(round(n_epochs * 0.1)))
    if n_epochs <= 1 or warmup >= n_epochs:
        return CosineAnnealingLR(optimizer, T_max=max(1, n_epochs), eta_min=eta_min)
    w = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    c = CosineAnnealingLR(optimizer, T_max=n_epochs - warmup, eta_min=eta_min)
    return SequentialLR(optimizer, [w, c], milestones=[warmup])


def train(config: TrainConfig) -> nn.Module:
    """
    Повний цикл навчання двоетапного тренування моделі.

    Стадія 1: Backbone заморожено, тренується лише голова (stage1_epochs).
    Стадія 2: Розморожуються останні N блоків, навчання з малим lr (stage2_epochs).

    Аргументи:
        config: Об'єкт TrainConfig із параметрами навчання.

    Повертає:
        Навчена модель PyTorch.
    """
    seed_everything(config.seed)
    # cudnn.benchmark підбирає найшвидші згорткові ядра для фіксованого розміру
    # входу (усі зображення 224/260px) — суттєво швидше, ціною повної
    # детермінованості. Вмикаємо опційно після seed_everything.
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("cudnn.benchmark=True (швидше, не повністю детерміновано)")
    device = get_device()

    # ── Дані ─────────────────────────────────────────────────────────────────
    logger.info("Завантаження датасету...")
    norm_mean, norm_std = get_norm_for(config.architecture)
    logger.info(f"Нормалізація для {config.architecture}: mean={norm_mean[0]:.3f}…")
    # Обмежуємо num_workers кількістю CPU (Windows spawn: завеликий
    # num_workers повільно стартує й перевантажує пам'ять).
    n_workers = min(config.num_workers, os.cpu_count() or 1)
    if n_workers != config.num_workers:
        logger.info(f"num_workers {config.num_workers} → {n_workers} (обмежено CPU)")
    dataloaders = create_dataloaders(
        manifest_path=config.manifest_path,
        train_transform=get_train_transforms(config.img_size, mean=norm_mean, std=norm_std),
        val_transform=get_val_transforms(config.img_size, mean=norm_mean, std=norm_std),
        countries=config.countries if config.countries else None,
        quality_threshold=config.quality_threshold,
        image_root=config.image_root if config.image_root else None,
        split_method=config.split_method,
        h3_resolution=config.h3_resolution,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        batch_size=config.batch_size,
        num_workers=n_workers,
        prefetch_factor=config.prefetch_factor,
        seed=config.seed,
    )
    num_classes  = dataloaders["num_classes"]
    class_names  = dataloaders["class_names"]
    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]

    logger.info(f"Датасет: {num_classes} класів, батч={config.batch_size}")

    # ── Модель ───────────────────────────────────────────────────────────────
    logger.info(f"Ініціалізація архітектури: {config.architecture}")
    # freeze_backbone приймає лише BaselineCNN; StreetCLIP/GeoCLIP заморожують
    # backbone за замовчуванням (freeze_vision / freeze_clip) і не мають цього аргументу.
    build_kwargs = {}
    if config.architecture == "baseline":
        build_kwargs["freeze_backbone"] = True
    model = build_model(
        architecture=config.architecture,
        num_classes=num_classes,
        pretrained=config.pretrained,
        **build_kwargs,
    )
    model = model.to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(
        f"Параметри: всього={format_param_count(total_params)}, "
        f"навчальних={format_param_count(trainable_params)}"
    )

    # ── Втрати, оптимізатор ──────────────────────────────────────────────────
    # Збалансовані ваги класів для нерівномірних датасетів
    if hasattr(train_loader.dataset, "get_class_weights"):
        class_weights = train_loader.dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_cuda  = device.type == "cuda"
    use_amp   = config.mixed_precision and use_cuda
    # Blackwell: bf16 стабільніший і не потребує GradScaler; fp16 — як фоллбек.
    amp_dtype = (torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = GradScaler() if (use_amp and amp_dtype == torch.float16) else None
    if use_amp:
        logger.info(
            f"AMP: dtype={amp_dtype}, "
            f"GradScaler={'on' if scaler is not None else 'off (bf16)'}"
        )
    cw = config.contrastive_loss_weight
    exp_logger = Logger(config)
    ckpt_manager = CheckpointManager(config.checkpoint_dir, save_top_k=config.save_top_k)
    early_stop   = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    global_step = 0
    total_epochs = config.stage1_epochs + config.stage2_epochs

    # ── Стадія 1: Заморожений backbone ───────────────────────────────────────
    if config.stage1_epochs > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"СТАДІЯ 1: Тренування голови ({config.stage1_epochs} епох, lr={config.stage1_lr})")
        logger.info(f"{'='*60}")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.stage1_lr,
            weight_decay=config.weight_decay,
        )
        scheduler = build_warmup_cosine(
            optimizer, config.stage1_epochs, config.stage1_lr * 0.01
        )

        for epoch in range(1, config.stage1_epochs + 1):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, config.grad_clip,
                use_amp, config.architecture, amp_dtype, cw,
                _cb=config.progress_callback,
                _cb_epoch=epoch,
                _cb_stage=1,
                _cb_total_epochs=total_epochs,
            )
            val_metrics = validate(
                model, val_loader, criterion,
                device, use_amp, config.architecture, amp_dtype, cw,
            )
            scheduler.step()

            elapsed = time.time() - t0
            lr_head     = optimizer.param_groups[0]["lr"]
            lr_backbone = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
            gpu_gb = 0.0
            if use_cuda:
                gpu_gb = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()

            logger.info(
                f"Епоха [{epoch:3d}/{config.stage1_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1_acc']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f} "
                f"val_bacc={val_metrics['balanced_acc']:.4f} | "
                f"lr_head={lr_head:.2e} lr_bb={lr_backbone:.2e} gpu={gpu_gb:.1f}GB | {elapsed:.1f}s"
            )

            global_step += 1
            exp_logger.log({
                "train/loss":       train_metrics["loss"],
                "train/acc":        train_metrics["acc"],
                "val/loss":         val_metrics["loss"],
                "val/top1_acc":     val_metrics["top1_acc"],
                "val/macro_f1":     val_metrics["macro_f1"],
                "val/balanced_acc": val_metrics["balanced_acc"],
                "lr/head":          lr_head,
                "lr/backbone":      lr_backbone,
                "gpu_gb":           gpu_gb,
                "stage":            1,
            }, step=global_step)

            ckpt_manager.save(
                model, optimizer, scheduler,
                epoch, val_metrics["loss"], val_metrics["top1_acc"],
                config, class_names,
            )

            _stop = early_stop(val_metrics["loss"])
            if config.progress_callback:
                config.progress_callback({
                    "type": "epoch", "stage": 1,
                    "epoch": epoch, "total_epochs": total_epochs,
                    "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"],
                    "val_loss": val_metrics["loss"], "val_top1": val_metrics["top1_acc"],
                    "val_macro_f1": val_metrics["macro_f1"], "val_bal_acc": val_metrics["balanced_acc"],
                    "best_val_loss": early_stop.best_value, "early_stop_counter": early_stop.counter,
                    "lr_head": lr_head, "lr_backbone": lr_backbone, "elapsed_sec": elapsed,
                })
            if _stop:
                logger.info("Early stopping спрацював на Стадії 1")
                break

        # Скидання стану early stop між стадіями
        early_stop.reset()

    # ── Стадія 2: Розморожування шарів ───────────────────────────────────────
    if config.stage2_epochs > 0:
        logger.info(f"\n{'='*60}")
        logger.info(
            f"СТАДІЯ 2: Тонке налаштування "
            f"(unfreezing {config.stage2_unfreeze_n} blocks/layers, "
            f"{config.stage2_epochs} епох, lr={config.stage2_lr})"
        )
        logger.info(f"{'='*60}")

        # Розморожуємо шари залежно від архітектури
        if config.stage2_unfreeze_n > 0:
            if hasattr(model, "unfreeze_last_n_blocks"):
                model.unfreeze_last_n_blocks(config.stage2_unfreeze_n)
            elif hasattr(model, "unfreeze_last_n_layers"):
                model.unfreeze_last_n_layers(config.stage2_unfreeze_n)

        # Диференціальні lr: предобучений backbone — нижчий lr, нові
        # модулі (gps_encoder, classifier/head, img_proj, log_temperature) —
        # повний stage2_lr. "backbone" визначаємо за іменами шарів
        # предобучених енкодерів (CLIP ViT / EfficientNet).
        BACKBONE_KEYS = ("vision_model", "visual_projection", "features")
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in BACKBONE_KEYS):
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": config.stage2_lr})
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": config.stage2_lr * 0.1}
            )

        optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
        scheduler = build_warmup_cosine(
            optimizer, config.stage2_epochs, config.stage2_lr * 0.001
        )

        for epoch in range(1, config.stage2_epochs + 1):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, config.grad_clip,
                use_amp, config.architecture, amp_dtype, cw,
                _cb=config.progress_callback,
                _cb_epoch=config.stage1_epochs + epoch,
                _cb_stage=2,
                _cb_total_epochs=total_epochs,
            )
            val_metrics = validate(
                model, val_loader, criterion,
                device, use_amp, config.architecture, amp_dtype, cw,
            )
            scheduler.step()

            elapsed = time.time() - t0
            lr_head     = optimizer.param_groups[0]["lr"]
            lr_backbone = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_head
            gpu_gb = 0.0
            if use_cuda:
                gpu_gb = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()

            logger.info(
                f"Епоха [{epoch:3d}/{config.stage2_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1_acc']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f} "
                f"val_bacc={val_metrics['balanced_acc']:.4f} | "
                f"lr_head={lr_head:.2e} lr_bb={lr_backbone:.2e} gpu={gpu_gb:.1f}GB | {elapsed:.1f}s"
            )

            global_step += 1
            exp_logger.log({
                "train/loss":        train_metrics["loss"],
                "train/acc":         train_metrics["acc"],
                "val/loss":          val_metrics["loss"],
                "val/top1_acc":      val_metrics["top1_acc"],
                "val/macro_f1":      val_metrics["macro_f1"],
                "val/balanced_acc":  val_metrics["balanced_acc"],
                "lr/head":           lr_head,
                "lr/backbone":       lr_backbone,
                "gpu_gb":            gpu_gb,
                "stage":             2,
            }, step=global_step)

            ckpt_manager.save(
                model, optimizer, scheduler,
                config.stage1_epochs + epoch,
                val_metrics["loss"], val_metrics["top1_acc"],
                config, class_names,
            )

            _stop = early_stop(val_metrics["loss"])
            if config.progress_callback:
                config.progress_callback({
                    "type": "epoch", "stage": 2,
                    "epoch": config.stage1_epochs + epoch, "total_epochs": total_epochs,
                    "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"],
                    "val_loss": val_metrics["loss"], "val_top1": val_metrics["top1_acc"],
                    "val_macro_f1": val_metrics["macro_f1"], "val_bal_acc": val_metrics["balanced_acc"],
                    "best_val_loss": early_stop.best_value, "early_stop_counter": early_stop.counter,
                    "lr_head": lr_head, "lr_backbone": lr_backbone, "elapsed_sec": elapsed,
                })
            if _stop:
                logger.info("Early stopping спрацював на Стадії 2")
                break

    exp_logger.finish()
    best_ckpt = Path(config.checkpoint_dir) / "best_model.pth"
    logger.info(f"\nНавчання завершено! Найкращий чекпоінт: {best_ckpt}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Навчання моделі геолокації вуличних зображень",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     type=str, default=None,
                        help="Шлях до YAML/JSON конфіг-файлу")
    parser.add_argument("--manifest",   type=str, default="dataset/manifests/train.csv",
                        help="Шлях до CSV-маніфесту")
    parser.add_argument("--image-root", type=str, default="dataset/raw/osv5m/images",
                        help="Корінь шляхів до зображень")
    parser.add_argument("--arch",       type=str, default="baseline",
                        choices=["baseline", "streetclip", "geoclip"],
                        help="Архітектура моделі")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Загальна кількість епох (розподіл 1/3 + 2/3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Початковий lr (Стадія 1)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-wandb",   action="store_true",
                        help="Вимкнути W&B логування")
    parser.add_argument("--use-mlflow", action="store_true",
                        help="Увімкнути MLflow логування")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-amp",     action="store_true",
                        help="Вимкнути mixed precision (AMP)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        from utils import load_config
        cfg_dict = load_config(args.config)
        # Використовуємо __dataclass_fields__ для надійної фільтрації ключів
        valid_keys = TrainConfig.__dataclass_fields__.keys()
        config = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    else:
        total_epochs = args.epochs
        stage1_ep = max(1, total_epochs // 3)
        stage2_ep = total_epochs - stage1_ep

        config = TrainConfig(
            manifest_path=args.manifest,
            image_root=args.image_root,
            architecture=args.arch,
            stage1_epochs=stage1_ep,
            stage2_epochs=stage2_ep,
            stage1_lr=args.lr,
            stage2_lr=args.lr * 0.1,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=not args.no_wandb,
            use_mlflow=args.use_mlflow,
            seed=args.seed,
            mixed_precision=not args.no_amp,
        )

    logger.info("Конфігурація навчання:")
    for k, v in asdict(config).items():
        logger.info(f"  {k}: {v}")

    train(config)


if __name__ == "__main__":
    main()
