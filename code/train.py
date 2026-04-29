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

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from augmentations import get_train_transforms, get_val_transforms
from dataset import GeoDataset, create_dataloaders
from metrics import top_k_accuracy_torch
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
    manifest_path:     str   = "data/manifest.csv"
    image_root:        str   = "data/images"
    countries:         list  = field(default_factory=lambda: ["UA"])
    quality_threshold: float = 0.4
    split_method:      str   = "h3"       # 'h3' або 'kmeans'
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

    # Early stopping
    patience:          int   = 7
    min_delta:         float = 1e-4

    # Логування
    use_wandb:         bool  = False
    use_mlflow:        bool  = False
    wandb_project:     str   = "ua-street-geolocation"
    wandb_run_name:    str   = ""
    mlflow_uri:        str   = "mlruns"
    experiment_name:   str   = "geolocation"

    # Збереження
    checkpoint_dir:    str   = "checkpoints"
    save_top_k:        int   = 3      # Зберігати K найкращих чекпоінтів

    # Відтворюваність
    seed:              int   = 42
    mixed_precision:   bool  = True


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
        self.counter   = 0
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
        scheduler: torch.optim.lr_scheduler._LRScheduler,
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
        torch.save(checkpoint, path)

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

        # Символічне посилання на найкращий чекпоінт
        best_path = self.checkpoint_dir / "best_model.pth"
        best_actual = self._checkpoints[0][1] if self.mode == "min" else self._checkpoints[-1][1]
        # Для 'min' найкращий — з найменшим loss, тобто в кінці після sort(reverse=False)
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
        checkpoint = torch.load(path, map_location=device, weights_only=False)
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
                reinit=True,
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
    scaler: GradScaler,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    architecture: str = "baseline",
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
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels, coords_tuple) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Для GeoCLIP потрібні координати
        if architecture == "geoclip":
            lat = torch.tensor([c[0] for c in coords_tuple[0]], dtype=torch.float32)
            lon = torch.tensor([c[1] for c in coords_tuple[1]], dtype=torch.float32)
            coords = torch.stack([lat, lon], dim=1).to(device, non_blocking=True)
        else:
            coords = None

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            if architecture == "geoclip" and coords is not None:
                output = model(images, coords=coords)
                logits = output["logits"]
                cls_loss = criterion(logits, labels)
                # Комбінований loss: класифікація + контрастивний
                contrastive = output.get("contrastive_loss", torch.tensor(0.0, device=device))
                loss = cls_loss + 0.1 * contrastive
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss    += loss.item() * batch_size
        total_correct += top_k_accuracy_torch(logits, labels, k=1).item() * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_samples
            avg_acc  = total_correct / total_samples
            logger.debug(
                f"  Batch [{batch_idx+1}/{len(loader)}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc":  total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    architecture: str = "baseline",
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

    for images, labels, coords_tuple in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            if architecture == "geoclip":
                output = model(images)
                logits = output["logits"]
            else:
                logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    top1 = top_k_accuracy_torch(all_logits, all_labels, k=1).item()
    top5 = top_k_accuracy_torch(all_logits, all_labels, k=min(5, all_logits.shape[1])).item()
    n = len(all_labels)

    return {
        "loss":     total_loss / max(n, 1),
        "top1_acc": top1,
        "top5_acc": top5,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Повний цикл навчання
# ──────────────────────────────────────────────────────────────────────────────

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
    device = get_device()

    # ── Дані ─────────────────────────────────────────────────────────────────
    logger.info("Завантаження датасету...")
    dataloaders = create_dataloaders(
        manifest_path=config.manifest_path,
        train_transform=get_train_transforms(config.img_size),
        val_transform=get_val_transforms(config.img_size),
        countries=config.countries if config.countries else None,
        quality_threshold=config.quality_threshold,
        image_root=config.image_root if config.image_root else None,
        split_method=config.split_method,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    num_classes  = dataloaders["num_classes"]
    class_names  = dataloaders["class_names"]
    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]

    logger.info(f"Датасет: {num_classes} класів, батч={config.batch_size}")

    # ── Модель ───────────────────────────────────────────────────────────────
    logger.info(f"Ініціалізація архітектури: {config.architecture}")
    model = build_model(
        architecture=config.architecture,
        num_classes=num_classes,
        pretrained=config.pretrained,
        freeze_backbone=(config.architecture == "baseline"),
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

    scaler    = GradScaler(enabled=config.mixed_precision and torch.cuda.is_available())
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
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.stage1_epochs,
            eta_min=config.stage1_lr * 0.01,
        )

        for epoch in range(1, config.stage1_epochs + 1):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, config.grad_clip,
                config.mixed_precision, config.architecture,
            )
            val_metrics = validate(
                model, val_loader, criterion,
                device, config.mixed_precision, config.architecture,
            )
            scheduler.step()

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Епоха [{epoch:3d}/{config.stage1_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1_acc']:.4f} "
                f"val_top5={val_metrics['top5_acc']:.4f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            global_step += 1
            exp_logger.log({
                "train/loss":     train_metrics["loss"],
                "train/acc":      train_metrics["acc"],
                "val/loss":       val_metrics["loss"],
                "val/top1_acc":   val_metrics["top1_acc"],
                "val/top5_acc":   val_metrics["top5_acc"],
                "lr":             current_lr,
                "stage":          1,
            }, step=global_step)

            ckpt_manager.save(
                model, optimizer, scheduler,
                epoch, val_metrics["loss"], val_metrics["top1_acc"],
                config, class_names,
            )

            if early_stop(val_metrics["loss"]):
                logger.info("Early stopping спрацював на Стадії 1")
                break

        # Скидання лічильника early stop між стадіями
        early_stop.counter = 0
        early_stop.should_stop = False

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

        # Диференціальні lr: backbone отримує нижчий lr
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in ["classifier", "head", "fc"]):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": head_params,     "lr": config.stage2_lr},
            {"params": backbone_params, "lr": config.stage2_lr * 0.1},
        ]

        optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.stage2_epochs,
            eta_min=config.stage2_lr * 0.001,
        )

        for epoch in range(1, config.stage2_epochs + 1):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, config.grad_clip,
                config.mixed_precision, config.architecture,
            )
            val_metrics = validate(
                model, val_loader, criterion,
                device, config.mixed_precision, config.architecture,
            )
            scheduler.step()

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Епоха [{epoch:3d}/{config.stage2_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1_acc']:.4f} "
                f"val_top5={val_metrics['top5_acc']:.4f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            global_step += 1
            exp_logger.log({
                "train/loss":     train_metrics["loss"],
                "train/acc":      train_metrics["acc"],
                "val/loss":       val_metrics["loss"],
                "val/top1_acc":   val_metrics["top1_acc"],
                "val/top5_acc":   val_metrics["top5_acc"],
                "lr":             current_lr,
                "stage":          2,
            }, step=global_step)

            ckpt_manager.save(
                model, optimizer, scheduler,
                config.stage1_epochs + epoch,
                val_metrics["loss"], val_metrics["top1_acc"],
                config, class_names,
            )

            if early_stop(val_metrics["loss"]):
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
    parser.add_argument("--manifest",   type=str, default="data/manifest.csv",
                        help="Шлях до CSV-маніфесту")
    parser.add_argument("--image-root", type=str, default="data/images",
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
        config = TrainConfig(**{k: v for k, v in cfg_dict.items() if hasattr(TrainConfig, k)})
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
