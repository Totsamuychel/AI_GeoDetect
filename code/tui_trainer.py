"""
tui_trainer.py — Terminal UI для управління навчанням AI GeoDetect.

Запуск:
    python code/tui_trainer.py          (з кореня проекту)
    python tui_trainer.py               (з папки code/)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from train import TrainConfig, train as _train_model

console = Console()

# ─── Константи ────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("configs/tui_config.json")

ARCH_INFO: dict[str, tuple[str, str]] = {
    "baseline":   ("EfficientNet-B2",        "configs/baseline.yaml"),
    "streetclip": ("StreetCLIP (ViT-L/14)",  "configs/streetclip.yaml"),
    "geoclip":    ("GeoCLIP + GPS encoder",  "configs/geoclip.yaml"),
}

EDITABLE_FIELDS = [
    ("stage1_epochs",   "Stage 1 epochs",           int),
    ("stage2_epochs",   "Stage 2 epochs",            int),
    ("stage1_lr",       "Stage 1 LR",                float),
    ("stage2_lr",       "Stage 2 LR",                float),
    ("batch_size",      "Batch size",                int),
    ("patience",        "Patience",                  int),
    ("mixed_precision", "Mixed precision (true/false)", "bool"),
    ("countries",       "Countries  (comma-separated)", "list"),
    ("manifest_path",   "Manifest path",             str),
    ("checkpoint_dir",  "Checkpoint dir",            str),
    ("use_wandb",       "Use W&B (true/false)",      "bool"),
    ("use_mlflow",      "Use MLflow (true/false)",   "bool"),
]

# ─── Конфіг ───────────────────────────────────────────────────────────────────

def _load_yaml_config(yaml_path: str) -> TrainConfig:
    try:
        import yaml
    except ImportError:
        console.print("[bold red]PyYAML не встановлено. Виконайте: pip install pyyaml[/bold red]")
        raise
    with open(yaml_path, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    valid = TrainConfig.__dataclass_fields__.keys()
    return TrainConfig(**{k: v for k, v in d.items() if k in valid})


def _load_tui_config(arch: str) -> TrainConfig:
    _, yaml_path = ARCH_INFO[arch]
    cfg = _load_yaml_config(yaml_path)
    if CONFIG_PATH.exists():
        saved: dict = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for k, v in saved.get(arch, {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def _save_tui_config(arch: str, config: TrainConfig) -> None:
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    saved: dict = {}
    if CONFIG_PATH.exists():
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    arch_saved = saved.get(arch, {})
    for f, *_ in EDITABLE_FIELDS:
        arch_saved[f] = getattr(config, f)
    saved[arch] = arch_saved
    CONFIG_PATH.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")

# ─── Головне меню ─────────────────────────────────────────────────────────────

def _show_main_menu() -> str:
    import sys as _sys
    console.clear()

    ASCII_LOGO = (
        "    ____  __________  __  __________  __________",
        "   / ___\\/ ____/ __ \\/ / / / ____/ / / /_  __/  ",
        "  / (_ // __/ / / / / /_/ / /   / /_/ / / /     ",
        "  \\___// /___/ /_/ / __  / /___/ __  / / /      ",
        "      /_____/\\____/_/ /_/\\____/_/ /_/ /_/       ",
        "                                                  ",
        "       [ Street-Level  GeoLocation  AI ]         ",
    )

    # ── footer info ───────────────────────────────────────────────────────────
    py_ver = f"Python {_sys.version_info.major}.{_sys.version_info.minor}"
    torch_ver = f"PyTorch {torch.__version__.split('+')[0]}"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_str  = f"GPU: {gpu_name}"
    else:
        gpu_str = "CPU  (no CUDA)"

    sep = "[orange1]" + "━" * 42 + "[/orange1]"

    lines: list[str] = []

    # logo
    for row in ASCII_LOGO:
        lines.append(f"[bold orange1]{row}[/bold orange1]")

    lines.append("")
    lines.append(f"[bold orange1 on black]{'AI GeoDetect — Training Manager':^50}[/bold orange1 on black]")
    lines.append("")

    # ── TRAIN section ─────────────────────────────────────────────────────────
    lines.append(f"[orange1]  ━━━━━  TRAIN  {'━' * 29}[/orange1]")
    lines.append("")
    lines.append(
        "  [orange1]▶[/orange1]  [bold orange1][1][/bold orange1]"
        "  [bright_white]Baseline[/bright_white]"
        "    [white]EfficientNet-B2[/white]"
    )
    lines.append(
        "  [orange1]▶[/orange1]  [bold orange1][2][/bold orange1]"
        "  [bright_white]StreetCLIP[/bright_white]"
        "  [white]ViT-L/14  Transfer Learning[/white]"
    )
    lines.append(
        "  [orange1]▶[/orange1]  [bold orange1][3][/bold orange1]"
        "  [bright_white]GeoCLIP[/bright_white]"
        "     [white]Contrastive + GPS Encoder[/white]"
    )
    lines.append(
        "  [orange1]▶[/orange1]  [bold orange1][4][/bold orange1]"
        "  [bright_white]ALL THREE[/bright_white]"
        "   [white]Train all models sequentially[/white]"
    )
    lines.append("")

    # ── TOOLS section ─────────────────────────────────────────────────────────
    lines.append(f"[orange1]  ━━━━━  TOOLS  {'━' * 29}[/orange1]")
    lines.append("")
    lines.append(
        "  ⚙  [bold orange1][5][/bold orange1]"
        "  [bright_white]Configure[/bright_white]  [white]Edit training settings[/white]"
    )
    lines.append(
        "  💾  [bold orange1][6][/bold orange1]"
        "  [bright_white]Checkpoints[/bright_white]  [white]Browse saved models[/white]"
    )
    lines.append(
        "  📊  [bold orange1][7][/bold orange1]"
        "  [bright_white]Evaluate[/bright_white]   [white]Run evaluation on test set[/white]"
    )
    lines.append("")

    # ── quit ─────────────────────────────────────────────────────────────────
    lines.append(f"[orange1]  {'─' * 42}[/orange1]")
    lines.append("  [bold red][q][/bold red]  [white]Quit[/white]")
    lines.append("")

    # ── footer ────────────────────────────────────────────────────────────────
    lines.append(f"  [dim orange1]{gpu_str}[/dim orange1]")
    lines.append(f"  [dim]{py_ver}  │  {torch_ver}[/dim]")

    body = "\n".join(lines)
    console.print(
        Panel(
            Text.from_markup(body),
            box=box.HEAVY,
            border_style="orange1",
            expand=False,
            padding=(0, 1),
        )
    )
    return console.input("[bold orange1]  ›[/bold orange1] ").strip().lower()

# ─── Запит шляхів ────────────────────────────────────────────────────────────

def _prompt_paths(arch: str, config: TrainConfig) -> TrainConfig:
    """Запитує шляхи до YAML-конфігу і датасету перед запуском навчання."""
    _, default_yaml = ARCH_INFO[arch]
    console.print(
        f"\n[bold orange1]Шляхи до файлів[/bold orange1]"
        f"  [dim](Enter = залишити поточне)[/dim]\n"
    )

    # ── YAML-конфіг ──────────────────────────────────────────────────────────
    yaml_inp = console.input(
        f"  YAML конфіг    [[dim]{default_yaml}[/dim]]: "
    ).strip()
    if yaml_inp:
        p = Path(yaml_inp)
        if p.exists():
            try:
                new_cfg = _load_yaml_config(str(p))
                # Зберігаємо поля, що були перевизначені через tui_config.json
                for fname, *_ in EDITABLE_FIELDS:
                    setattr(new_cfg, fname, getattr(config, fname))
                config = new_cfg
                console.print(f"  [green]✓ Завантажено: {yaml_inp}[/green]")
            except Exception as exc:
                console.print(f"  [yellow]Не вдалося прочитати YAML ({exc}). Використовується поточний конфіг.[/yellow]")
        else:
            console.print(f"  [yellow]Файл не знайдено: {yaml_inp}[/yellow]")

    # ── Manifest CSV ──────────────────────────────────────────────────────────
    manifest_inp = console.input(
        f"  Manifest CSV   [[dim]{config.manifest_path}[/dim]]: "
    ).strip()
    if manifest_inp:
        config.manifest_path = manifest_inp

    # ── Image root ────────────────────────────────────────────────────────────
    image_root_inp = console.input(
        f"  Image root     [[dim]{config.image_root}[/dim]]: "
    ).strip()
    if image_root_inp:
        config.image_root = image_root_inp

    return config


# ─── Таблиця параметрів ────────────────────────────────────────────────────────

def _show_config_table(arch: str, config: TrainConfig) -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device_str += f" ({torch.cuda.get_device_name(0)})"
    total = config.stage1_epochs + config.stage2_epochs

    t = Table(title=f"Config: [bold cyan]{arch}[/bold cyan]", box=box.ROUNDED, expand=False)
    t.add_column("Parameter", style="bold", min_width=22)
    t.add_column("Value")
    t.add_row("Architecture", arch)
    t.add_row("Total epochs", f"{total}  (stage1={config.stage1_epochs}, stage2={config.stage2_epochs})")
    t.add_row("LR  stage1 / stage2", f"{config.stage1_lr}  /  {config.stage2_lr}")
    t.add_row("Batch size", str(config.batch_size))
    t.add_row("Weight decay", str(config.weight_decay))
    t.add_row("Patience", str(config.patience))
    t.add_row("AMP / mixed precision", "on" if config.mixed_precision else "off")
    t.add_row("Device", device_str)
    t.add_row("Countries", ", ".join(config.countries or []))
    t.add_row("Manifest", config.manifest_path)
    t.add_row("Checkpoint dir", config.checkpoint_dir)
    t.add_row("W&B / MLflow", f"{'on' if config.use_wandb else 'off'}  /  {'on' if config.use_mlflow else 'off'}")
    console.print(t)

# ─── Live-панель навчання ─────────────────────────────────────────────────────

def _fmt_time(secs: float) -> str:
    s = int(max(0, secs))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _bar(frac: float, width: int = 28) -> str:
    filled = max(0, min(width, int(frac * width)))
    return "█" * filled + "░" * (width - filled)


def _build_live_panel(state: dict, arch: str) -> Panel:
    epoch         = state.get("epoch", 0)
    total_epochs  = state.get("total_epochs", 1)
    stage         = state.get("stage", 1)
    batch         = state.get("batch", 0)
    total_batches = state.get("total_batches", 1)
    train_loss    = state.get("train_loss")
    train_acc     = state.get("train_acc")
    val_loss      = state.get("val_loss")
    val_top1      = state.get("val_top1")
    val_f1        = state.get("val_macro_f1")
    val_bacc      = state.get("val_bal_acc")
    best_loss     = state.get("best_val_loss")
    es_counter    = state.get("early_stop_counter", 0)
    patience      = state.get("patience", 7)
    lr_head       = state.get("lr_head")
    lr_backbone   = state.get("lr_backbone")
    elapsed       = state.get("elapsed_sec", 0.0)
    eta           = state.get("eta_sec")
    status        = state.get("status", "running")

    lines: list[str] = []

    stage_name = "frozen backbone" if stage == 1 else "fine-tuning"
    lines.append(f"  Stage: [bold]{stage}[/bold] ({stage_name})")

    frac = batch / max(total_batches, 1)
    lines.append(f"  {_bar(frac)}  Batch {batch}/{total_batches}")
    lines.append("")

    tl = f"{train_loss:.4f}" if train_loss is not None else "—"
    ta = f"{train_acc:.4f}"  if train_acc  is not None else "—"
    lines.append(f"  [bold]Train[/bold]  │  loss: {tl}  │  acc:  {ta}")

    if val_loss is not None:
        vl  = f"{val_loss:.4f}"
        vt1 = f"{val_top1:.4f}" if val_top1 is not None else "—"
        vf  = f"{val_f1:.4f}"   if val_f1   is not None else "—"
        vb  = f"{val_bacc:.4f}" if val_bacc  is not None else "—"
        lines.append(f"  [bold]Val  [/bold]  │  loss: {vl}  │  top1: {vt1}")
        lines.append(f"           │  macro_f1: {vf}  │  bal_acc: {vb}")
    else:
        lines.append("  [dim]Val    │  — (waiting for epoch end)[/dim]")
    lines.append("")

    bl = f"[bold green]{best_loss:.4f}[/bold green]" if best_loss is not None else "—"
    lines.append(f"  Best val_loss: {bl}")

    es_style = "bold yellow" if es_counter > patience // 2 else "dim"
    lines.append(f"  Early stop counter: [{es_style}]{es_counter} / {patience}[/{es_style}]")

    lh = f"{lr_head:.2e}"     if lr_head     is not None else "—"
    lb = f"{lr_backbone:.2e}" if lr_backbone  is not None else "—"
    lines.append(f"  LR head: {lh}  │  LR backbone: {lb}")

    eta_str = _fmt_time(eta) if eta is not None else "—"
    lines.append(f"  Elapsed: {_fmt_time(elapsed)}  │  ETA: {eta_str}")
    lines.append("")

    if status == "done":
        footer = "[bold green]✓ Навчання завершено![/bold green]"
    elif status == "error":
        footer = f"[bold red]✗ Помилка: {state.get('error', '')}[/bold red]"
    elif status == "stopped":
        footer = "[bold yellow]⏹ Зупинено користувачем[/bold yellow]"
    else:
        footer = "[dim]Ctrl+C → зупинити навчання[/dim]"
    lines.append(f"  {footer}")

    body = Text.from_markup("\n".join(lines))
    title = f"Training: [bold cyan]{arch}[/bold cyan]  Epoch {epoch}/{total_epochs}"
    return Panel(body, title=title, border_style="cyan", expand=False)

# ─── Потік навчання ───────────────────────────────────────────────────────────

def _train_worker(
    config: TrainConfig,
    result_queue: queue.Queue,
) -> None:
    try:
        _train_model(config)
        result_queue.put({"type": "done"})
    except KeyboardInterrupt:
        result_queue.put({"type": "stopped"})
    except Exception as exc:
        result_queue.put({"type": "error", "error": str(exc)})


def _apply_info(state: dict, info: dict, history: list[dict]) -> None:
    t = info.get("type")
    if t == "batch":
        state.update({
            "epoch":         info.get("epoch", state.get("epoch", 0)),
            "total_epochs":  info.get("total_epochs", state.get("total_epochs", 1)),
            "stage":         info.get("stage", 1),
            "batch":         info.get("batch", 0),
            "total_batches": info.get("total_batches", 1),
            "train_loss":    info.get("running_loss"),
            "train_acc":     info.get("running_acc"),
        })
    elif t == "epoch":
        state.update({
            "epoch":               info.get("epoch", state.get("epoch", 0)),
            "total_epochs":        info.get("total_epochs", state.get("total_epochs", 1)),
            "stage":               info.get("stage", 1),
            "batch":               state.get("total_batches", 1),
            "train_loss":          info.get("train_loss"),
            "train_acc":           info.get("train_acc"),
            "val_loss":            info.get("val_loss"),
            "val_top1":            info.get("val_top1"),
            "val_macro_f1":        info.get("val_macro_f1"),
            "val_bal_acc":         info.get("val_bal_acc"),
            "best_val_loss":       info.get("best_val_loss"),
            "early_stop_counter":  info.get("early_stop_counter", 0),
            "lr_head":             info.get("lr_head"),
            "lr_backbone":         info.get("lr_backbone"),
            "elapsed_sec":         info.get("elapsed_sec", state.get("elapsed_sec", 0.0)),
        })
        history.append({k: info.get(k) for k in (
            "stage", "epoch", "train_loss", "train_acc",
            "val_loss", "val_top1", "val_macro_f1", "val_bal_acc",
        )})
    elif t in ("done", "stopped", "error"):
        state["status"] = t
        if t == "error":
            state["error"] = info.get("error", "")


def _run_training_live(arch: str, config: TrainConfig, transient: bool = True) -> list[dict]:
    _transient = transient
    cb_queue:     queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    history: list[dict] = []

    start_time = time.time()
    state: dict = {
        "epoch": 0,
        "total_epochs": config.stage1_epochs + config.stage2_epochs,
        "stage": 1,
        "batch": 0,
        "total_batches": 1,
        "patience": config.patience,
        "status": "running",
        "elapsed_sec": 0.0,
    }

    def _callback(info: dict) -> None:
        if stop_event.is_set():
            raise KeyboardInterrupt("Зупинено через TUI")
        cb_queue.put(info)

    config.progress_callback = _callback

    train_thread = threading.Thread(
        target=_train_worker, args=(config, result_queue), daemon=True
    )
    train_thread.start()

    try:
        with Live(console=console, refresh_per_second=5, transient=_transient) as live:
            while train_thread.is_alive() or not cb_queue.empty():
                while not cb_queue.empty():
                    _apply_info(state, cb_queue.get_nowait(), history)
                while not result_queue.empty():
                    _apply_info(state, result_queue.get_nowait(), history)

                state["elapsed_sec"] = time.time() - start_time
                ep            = state.get("epoch", 0)
                tot           = state.get("total_epochs", 1)
                batch         = state.get("batch", 0)
                total_batches = state.get("total_batches", 1)
                # Fractional completed epochs: (ep-1) full + current batch progress.
                # Avoids overestimating speed when ep=1 and only seconds have passed.
                completed = (ep - 1) + batch / max(total_batches, 1)
                if completed > 0 and state["elapsed_sec"] > 0:
                    spe = state["elapsed_sec"] / completed
                    state["eta_sec"] = spe * max(0, tot - completed)

                live.update(_build_live_panel(state, arch))
                time.sleep(0.12)

        while not cb_queue.empty():
            _apply_info(state, cb_queue.get_nowait(), history)
        while not result_queue.empty():
            _apply_info(state, result_queue.get_nowait(), history)

    except KeyboardInterrupt:
        stop_event.set()
        state["status"] = "stopped"
        console.print("\n[bold yellow]⏹ Очікування завершення поточного батчу...[/bold yellow]")
        train_thread.join(timeout=60)
        if train_thread.is_alive():
            console.print("[bold red]⚠ Потік навчання не завершився за 60с (daemon, буде вбито при виході)[/bold red]")
        # Drain queues so history and final status are not lost after Ctrl+C.
        while not cb_queue.empty():
            _apply_info(state, cb_queue.get_nowait(), history)
        while not result_queue.empty():
            _apply_info(state, result_queue.get_nowait(), history)

    if state.get("status") == "running":
        state["status"] = "done"

    console.print(_build_live_panel(state, arch))
    return history

# ─── Таблиця результатів ──────────────────────────────────────────────────────

def _show_results_table(arch: str, history: list[dict], checkpoint_dir: str) -> None:
    t = Table(title=f"📊 Results: [bold cyan]{arch}[/bold cyan]", box=box.ROUNDED)
    t.add_column("Ep",    justify="right")
    t.add_column("St",    justify="center")
    t.add_column("Train Loss", justify="right")
    t.add_column("Val Loss",   justify="right")
    t.add_column("Top-1",      justify="right")
    t.add_column("Macro F1",   justify="right")
    t.add_column("Bal Acc",    justify="right")

    best_vl = min((r["val_loss"] for r in history if r.get("val_loss") is not None), default=None)

    for row in history:
        vl = row.get("val_loss")
        vl_str = f"[bold green]{vl:.4f}[/bold green]" if vl is not None and vl == best_vl else (f"{vl:.4f}" if vl is not None else "—")
        t.add_row(
            str(row.get("epoch", "?")),
            str(row.get("stage", "?")),
            f"{row['train_loss']:.4f}" if row.get("train_loss") is not None else "—",
            vl_str,
            f"{row['val_top1']:.4f}"    if row.get("val_top1")    is not None else "—",
            f"{row['val_macro_f1']:.4f}" if row.get("val_macro_f1") is not None else "—",
            f"{row['val_bal_acc']:.4f}"  if row.get("val_bal_acc")  is not None else "—",
        )
    console.print(t)
    best_pth = Path(checkpoint_dir) / "best_model.pth"
    console.print(f"\n[bold green]💾 Best checkpoint:[/bold green] {best_pth}")

# ─── Пункти меню ─────────────────────────────────────────────────────────────

def run_training(arch: str) -> None:
    config = _load_tui_config(arch)
    console.clear()
    config = _prompt_paths(arch, config)
    console.print()
    _show_config_table(arch, config)
    ans = console.input("\nПочати навчання? [y/N]: ").strip().lower()
    if ans != "y":
        return
    history = _run_training_live(arch, config)
    if history:
        _show_results_table(arch, history, config.checkpoint_dir)
    console.input("\n[dim]Enter → повернутися в меню...[/dim]")


def _show_all_status(archs: list[str], statuses: dict, results: dict) -> None:
    console.clear()
    t = Table(box=box.ROUNDED, show_header=False, expand=False)
    t.add_column("icon",  width=4)
    t.add_column("arch",  width=12)
    t.add_column("info")
    icons = {"done": "[bold green]✓[/bold green]", "running": "[bold yellow]▶[/bold yellow]",
             "waiting": "[ ]", "stopped": "[bold red]⏹[/bold red]"}
    for a in archs:
        s = statuses[a]
        icon = icons.get(s, "?")
        if s == "done" and a in results:
            r = results[a]
            vl = f"{r['val_loss']:.4f}" if r.get("val_loss") is not None else "?"
            t1 = f"{r['top1']:.4f}"     if r.get("top1")     is not None else "?"
            info = f"DONE  val_loss={vl}  top1={t1}"
        elif s == "running":
            info = "RUNNING..."
        elif s == "stopped":
            info = "STOPPED"
        else:
            info = "WAITING"
        t.add_row(icon, a, info)
    console.print(Panel(t, title="Training ALL models", border_style="cyan"))


def _show_comparison_table(archs: list[str], all_histories: dict[str, list[dict]]) -> None:
    t = Table(title="📊 Model Comparison", box=box.ROUNDED)
    t.add_column("Architecture")
    t.add_column("Epochs", justify="right")
    t.add_column("Best Val Loss", justify="right")
    t.add_column("Best Top-1",   justify="right")
    t.add_column("Best F1",      justify="right")
    for arch in archs:
        hist = all_histories.get(arch, [])
        if not hist:
            t.add_row(arch, "—", "—", "—", "—")
            continue
        valid = [r for r in hist if r.get("val_loss") is not None]
        if not valid:
            t.add_row(arch, str(len(hist)), "—", "—", "—")
            continue
        best = min(valid, key=lambda r: r["val_loss"])
        t.add_row(
            arch,
            str(len(hist)),
            f"{best['val_loss']:.4f}",
            f"{best['val_top1']:.4f}"    if best.get("val_top1")    is not None else "—",
            f"{best['val_macro_f1']:.4f}" if best.get("val_macro_f1") is not None else "—",
        )
    console.print(t)


def run_all_training() -> None:
    archs = ["baseline", "streetclip", "geoclip"]
    configs = {a: _load_tui_config(a) for a in archs}
    console.clear()
    console.print("[bold cyan]Train ALL THREE models (послідовно)[/bold cyan]\n")

    # Спільні шляхи для всіх архітектур
    ref = configs["baseline"]
    console.print("[dim]Шляхи до датасету (однакові для всіх архітектур):[/dim]\n")
    manifest_inp = console.input(
        f"  Manifest CSV  [[dim]{ref.manifest_path}[/dim]]: "
    ).strip()
    image_root_inp = console.input(
        f"  Image root    [[dim]{ref.image_root}[/dim]]: "
    ).strip()
    if manifest_inp or image_root_inp:
        for a in archs:
            if manifest_inp:
                configs[a].manifest_path = manifest_inp
            if image_root_inp:
                configs[a].image_root = image_root_inp

    console.print()
    ans = console.input("Почати? [y/N]: ").strip().lower()
    if ans != "y":
        return

    statuses: dict[str, str] = {a: "waiting" for a in archs}
    results:  dict[str, dict] = {}
    all_histories: dict[str, list[dict]] = {}

    for arch in archs:
        statuses[arch] = "running"
        _show_all_status(archs, statuses, results)
        history = _run_training_live(arch, configs[arch], transient=False)
        all_histories[arch] = history
        valid = [r for r in history if r.get("val_loss") is not None]
        if valid:
            best = min(valid, key=lambda r: r["val_loss"])
            results[arch] = {"val_loss": best["val_loss"], "top1": best.get("val_top1")}
            statuses[arch] = "done"
        else:
            statuses[arch] = "stopped"

    _show_all_status(archs, statuses, results)
    _show_comparison_table(archs, all_histories)
    console.input("\n[dim]Enter → повернутися в меню...[/dim]")


def configure_settings() -> None:
    console.clear()
    console.print("[bold cyan]🔧 Configure Settings[/bold cyan]\n")
    arch = console.input(
        "Архітектура ([bold]baseline[/bold] / streetclip / geoclip): "
    ).strip() or "baseline"
    if arch not in ARCH_INFO:
        console.print("[bold red]Невідома архітектура.[/bold red]")
        console.input("[dim]Enter...[/dim]")
        return

    config = _load_tui_config(arch)
    console.print(f"\n[dim]Залиште поле порожнім, щоб не змінювати значення.[/dim]\n")

    for field_name, label, typ in EDITABLE_FIELDS:
        current = getattr(config, field_name)
        current_str = ", ".join(current) if isinstance(current, list) else str(current)
        val = console.input(f"  {label} [{current_str}]: ").strip()
        if not val:
            continue
        try:
            if typ is int:
                setattr(config, field_name, int(val))
            elif typ is float:
                setattr(config, field_name, float(val))
            elif typ == "bool":
                setattr(config, field_name, val.lower() in ("true", "1", "yes", "on"))
            elif typ == "list":
                setattr(config, field_name, [x.strip() for x in val.split(",")])
            else:
                setattr(config, field_name, val)
        except ValueError:
            console.print(f"  [yellow]Невалідне значення — пропущено.[/yellow]")

    _save_tui_config(arch, config)
    console.print(f"\n[bold green]✓ Збережено: {CONFIG_PATH}[/bold green]")
    console.input("\n[dim]Enter...[/dim]")


def view_checkpoints() -> None:
    console.clear()
    console.print("[bold cyan]💾 Checkpoints[/bold cyan]\n")
    ckpt_root = Path("checkpoints")
    if not ckpt_root.exists():
        console.print("[dim]Директорія checkpoints/ не існує.[/dim]")
        console.input("\n[dim]Enter...[/dim]")
        return

    t = Table(box=box.ROUNDED)
    t.add_column("File")
    t.add_column("Arch",     min_width=10)
    t.add_column("Epoch",    justify="right")
    t.add_column("Val Loss", justify="right")
    t.add_column("Val Acc",  justify="right")

    found = False
    for pth in sorted(ckpt_root.rglob("*.pth")):
        rel = str(pth.relative_to(ckpt_root.parent))
        try:
            ck = torch.load(pth, map_location="cpu", weights_only=True)
            arch  = ck.get("config", {}).get("architecture", "?")
            epoch = str(ck.get("epoch", "?"))
            vl    = f"{ck['val_loss']:.4f}" if "val_loss" in ck else "?"
            va    = f"{ck['val_acc']:.4f}"  if "val_acc"  in ck else "?"
        except Exception:
            arch, epoch, vl, va = "?", "?", "?", "?"
        t.add_row(rel, arch, epoch, vl, va)
        found = True

    if found:
        console.print(t)
    else:
        console.print("[dim]Чекпоінтів не знайдено.[/dim]")
    console.input("\n[dim]Enter...[/dim]")


def evaluate_model() -> None:
    console.clear()
    console.print("[bold cyan]📊 Evaluate Model[/bold cyan]\n")
    ckpt     = console.input("Шлях до checkpoint (.pth): ").strip()
    manifest = console.input("Шлях до маніфесту (val.csv / test.csv): ").strip()
    output   = console.input("Output JSON [results/eval.json]: ").strip() or "results/eval.json"

    if not ckpt or not manifest:
        console.print("[bold red]Шляхи обов'язкові.[/bold red]")
        console.input("[dim]Enter...[/dim]")
        return

    arch = "baseline"
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        arch = ck.get("config", {}).get("architecture", "baseline")
        console.print(f"[dim]Архітектура з чекпоінту: {arch}[/dim]")
    except Exception:
        console.print("[yellow]Не вдалося прочитати архітектуру з чекпоінту, використовується 'baseline'[/yellow]")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    eval_script = Path(__file__).parent / "evaluate.py"
    cmd = [sys.executable, str(eval_script),
           "--checkpoint", ckpt, "--manifest", manifest,
           "--output", output, "--architecture", arch]
    console.print(f"\n[dim]$ {' '.join(cmd)}[/dim]\n")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        console.print(f"\n[bold green]✓ Результати збережено: {output}[/bold green]")
    else:
        console.print(f"\n[bold red]✗ Помилка (exit={result.returncode})[/bold red]")
    console.input("\n[dim]Enter...[/dim]")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ACTIONS = {
        "1": lambda: run_training("baseline"),
        "2": lambda: run_training("streetclip"),
        "3": lambda: run_training("geoclip"),
        "4": run_all_training,
        "5": configure_settings,
        "6": view_checkpoints,
        "7": evaluate_model,
    }
    while True:
        try:
            choice = _show_main_menu()
        except KeyboardInterrupt:
            break
        if choice in ("q", "quit", "exit"):
            break
        action = ACTIONS.get(choice)
        if action:
            try:
                action()
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Перервано. Повернення в меню...[/bold yellow]")
            except Exception as exc:
                console.print(f"\n[bold red]Помилка: {exc}[/bold red]")
                console.input("[dim]Enter...[/dim]")
        else:
            console.print("[dim]Невідома команда.[/dim]")

    console.print("\n[bold cyan]До побачення! 👋[/bold cyan]\n")


if __name__ == "__main__":
    main()
