"""
visualize.py — Візуалізація результатів навчання та оцінки моделі.

Реалізує:
- plot_confusion_matrix:    матриця плутанини (seaborn)
- plot_learning_curves:     криві навчання (matplotlib)
- plot_error_map:           карта помилок (Folium) з позначками
- plot_gradcam:             теплова карта GradCAM (pytorch_grad_cam)
- plot_tsne_embeddings:     t-SNE проєкція embedding-простору (sklearn)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Стиль matplotlib
plt.rcParams.update({
    "figure.dpi":       120,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
})


# ──────────────────────────────────────────────────────────────────────────────
# Матриця плутанини
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: list[str],
    normalize: bool = True,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    title: str = "Матриця плутанини",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує та відображає матрицю плутанини.

    Аргументи:
        y_true:      Масив справжніх міток.
        y_pred:      Масив передбачених міток.
        class_names: Список назв класів.
        normalize:   Якщо True, нормалізує по рядках (відображає recall).
        figsize:     Розмір фігури (ширина, висота).
        cmap:        Колірна карта matplotlib.
        title:       Заголовок графіку.
        save_path:   Шлях для збереження (PNG/SVG/PDF).

    Повертає:
        matplotlib.figure.Figure.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalize:
        # Нормалізація по рядках (recall per class)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
        fmt = ".2f"
        colorbar_label = "Recall (нормалізовано)"
    else:
        cm_display = cm
        fmt = "d"
        colorbar_label = "Кількість зразків"

    fig, ax = plt.subplots(figsize=figsize)

    try:
        import seaborn as sns
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": colorbar_label},
        )
    except ImportError:
        # Fallback без seaborn
        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label=colorbar_label)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
        thresh = cm_display.max() / 2.0
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                val = f"{cm_display[i, j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if cm_display[i, j] > thresh else "black",
                        fontsize=8)

    ax.set_xlabel("Передбачений клас", fontsize=12)
    ax.set_ylabel("Справжній клас", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Матриця плутанини збережена: {save_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Криві навчання
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: Optional[list[float]] = None,
    val_accs: Optional[list[float]] = None,
    title: str = "Криві навчання",
    figsize: tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує криві навчання (loss та accuracy).

    Аргументи:
        train_losses: Loss на тренувальному наборі по епохах.
        val_losses:   Loss на валідаційному наборі.
        train_accs:   Accuracy на тренувальному наборі (необов'язково).
        val_accs:     Accuracy на валідаційному наборі (необов'язково).
        title:        Заголовок.
        figsize:      Розмір фігури.
        save_path:    Шлях для збереження.

    Повертає:
        matplotlib.figure.Figure.
    """
    has_acc = train_accs is not None and val_accs is not None
    n_plots = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = np.arange(1, len(train_losses) + 1)

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax_loss = axes[0]
    ax_loss.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    ax_loss.plot(epochs, val_losses,   label="Val Loss",   color="#F44336", linewidth=2,
                 linestyle="--")

    # Позначення мінімального val_loss
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val   = min(val_losses)
    ax_loss.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7)
    ax_loss.scatter([best_epoch], [best_val], color="#F44336", s=100, zorder=5,
                    label=f"Найкращий (ep {best_epoch}: {best_val:.4f})")

    ax_loss.set_xlabel("Епоха")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    # ── Accuracy ──────────────────────────────────────────────────────────────
    if has_acc:
        ax_acc = axes[1]
        ax_acc.plot(epochs, [a * 100 for a in train_accs],
                    label="Train Acc", color="#4CAF50", linewidth=2)
        ax_acc.plot(epochs, [a * 100 for a in val_accs],
                    label="Val Acc",   color="#FF9800", linewidth=2, linestyle="--")

        best_acc_epoch = int(np.argmax(val_accs)) + 1
        best_acc = max(val_accs) * 100
        ax_acc.axvline(best_acc_epoch, color="gray", linestyle=":", alpha=0.7)
        ax_acc.scatter([best_acc_epoch], [best_acc], color="#FF9800", s=100, zorder=5,
                       label=f"Найкращий (ep {best_acc_epoch}: {best_acc:.1f}%)")
        ax_acc.set_xlabel("Епоха")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy")
        ax_acc.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Криві навчання збережені: {save_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Карта помилок (Folium)
# ──────────────────────────────────────────────────────────────────────────────

def plot_error_map(
    predictions_df: "pandas.DataFrame",
    output_path: Optional[str] = "results/error_map.html",
    true_lat_col: str = "true_lat",
    true_lon_col: str = "true_lon",
    pred_lat_col: str = "pred_lat",
    pred_lon_col: str = "pred_lon",
    correct_col: str = "correct",
    city_col: Optional[str] = "true_city",
    max_points: int = 2000,
) -> "folium.Map":
    """
    Будує інтерактивну Folium-карту з правильними та помилковими передбаченнями.

    Зелені маркери = правильно класифіковано.
    Червоні маркери = помилка класифікації.
    Для помилкових передбачень малюється лінія від реальної до передбаченої точки.

    Аргументи:
        predictions_df: DataFrame із колонками true_lat, true_lon,
                        pred_lat, pred_lon, correct (bool/int).
        output_path:    Шлях для збереження HTML.
        true_lat_col:   Назва колонки реальної широти.
        true_lon_col:   Назва колонки реальної довготи.
        pred_lat_col:   Назва колонки передбаченої широти.
        pred_lon_col:   Назва колонки передбаченої довготи.
        correct_col:    Назва колонки коректності (True/False або 1/0).
        city_col:       Назва колонки назви міста.
        max_points:     Максимальна кількість точок (для продуктивності).

    Повертає:
        folium.Map.
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        raise ImportError("Встановіть folium: pip install folium")

    import pandas as pd

    df = predictions_df.copy()

    # Обмеження кількості точок
    if len(df) > max_points:
        logger.warning(f"Відображаємо {max_points} з {len(df)} точок")
        df_correct   = df[df[correct_col].astype(bool)].sample(
            min(max_points // 2, df[df[correct_col].astype(bool)].shape[0]), random_state=42
        )
        df_incorrect = df[~df[correct_col].astype(bool)].sample(
            min(max_points // 2, df[~df[correct_col].astype(bool)].shape[0]), random_state=42
        )
        df = pd.concat([df_correct, df_incorrect])

    # Центр карти
    center_lat = df[true_lat_col].mean()
    center_lon = df[true_lon_col].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    # Кластери для великої кількості точок
    correct_cluster   = MarkerCluster(name="Правильні (зелені)").add_to(m)
    incorrect_cluster = MarkerCluster(name="Помилкові (червоні)").add_to(m)

    for _, row in df.iterrows():
        t_lat = row[true_lat_col]
        t_lon = row[true_lon_col]
        p_lat = row[pred_lat_col]
        p_lon = row[pred_lon_col]
        is_correct = bool(row[correct_col])
        city = row[city_col] if city_col and city_col in row else ""

        color = "green" if is_correct else "red"
        icon_name = "check" if is_correct else "times"

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>{'✓ Правильно' if is_correct else '✗ Помилка'}</b><br/>
            Місто: <b>{city}</b><br/>
            Реальні: {t_lat:.4f}°N, {t_lon:.4f}°E<br/>
            Передбачені: {p_lat:.4f}°N, {p_lon:.4f}°E
        </div>
        """

        marker = folium.Marker(
            location=[t_lat, t_lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{'✓' if is_correct else '✗'} {city}",
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        )

        if is_correct:
            marker.add_to(correct_cluster)
        else:
            marker.add_to(incorrect_cluster)
            # Лінія від реального до передбаченого місця
            folium.PolyLine(
                locations=[[t_lat, t_lon], [p_lat, p_lon]],
                color="red",
                weight=1,
                opacity=0.4,
            ).add_to(m)

    # Статистика у вигляді HTML-блоку
    n_correct   = df[correct_col].astype(bool).sum()
    n_total     = len(df)
    accuracy    = n_correct / n_total * 100 if n_total > 0 else 0

    stats_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; z-index: 1000;
                background: rgba(255,255,255,0.95); padding: 12px 18px;
                border-radius: 8px; border: 1px solid #ddd;
                font-family: Arial; font-size: 13px;">
        <b>Статистика</b><br/>
        Всього: {n_total}<br/>
        Правильно: {n_correct} ({accuracy:.1f}%)<br/>
        Помилок: {n_total - n_correct}
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))
    folium.LayerControl(collapsed=False).add_to(m)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        logger.info(f"Карта помилок збережена: {output_path}")

    return m


# ──────────────────────────────────────────────────────────────────────────────
# GradCAM
# ──────────────────────────────────────────────────────────────────────────────

def plot_gradcam(
    model: "torch.nn.Module",
    image: "torch.Tensor",
    target_layer: "torch.nn.Module",
    target_class: Optional[int] = None,
    figsize: tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
    alpha: float = 0.5,
) -> plt.Figure:
    """
    Візуалізує GradCAM теплову карту для зображення.

    Показує, які частини зображення модель вважає важливими
    для передбачення.

    Аргументи:
        model:        PyTorch модель.
        image:        Тензор зображення (3, H, W) або (1, 3, H, W).
        target_layer: Цільовий шар для обчислення GradCAM.
        target_class: Цільовий клас (None = топ передбачення).
        figsize:      Розмір фігури.
        save_path:    Шлях для збереження.
        alpha:        Прозорість теплової карти (0.0–1.0).

    Повертає:
        matplotlib.figure.Figure.

    Примітки:
        Потребує: pip install grad-cam
        Типовий виклик:
            target_layer = model.features[-1]  # Для EfficientNet
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError("Встановіть grad-cam: pip install grad-cam")

    import torch
    from augmentations import denormalize

    if image.ndim == 3:
        image = image.unsqueeze(0)

    # Денормалізація для відображення
    img_display = denormalize(image[0].cpu()).permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1).astype(np.float32)

    # GradCAM
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=image,
            targets=targets,
        )
        grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True, image_weight=1 - alpha)

    # Передбачення
    import torch.nn.functional as F
    with torch.no_grad():
        logits = model(image)
        probs  = F.softmax(logits, dim=1)[0]
        top_prob, top_class = probs.max(dim=0)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(img_display)
    axes[0].set_title("Оригінальне зображення")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("GradCAM (теплова карта)")
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title(
        f"Накладення\nКлас {top_class.item()}: {top_prob.item()*100:.1f}%"
    )
    axes[2].axis("off")

    plt.suptitle("GradCAM Visualization", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"GradCAM збережено: {save_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# t-SNE embedding
# ──────────────────────────────────────────────────────────────────────────────

def plot_tsne_embeddings(
    embeddings: Union[np.ndarray, "torch.Tensor"],
    labels: Union[np.ndarray, list],
    class_names: list[str],
    title: str = "t-SNE проєкція embedding-простору",
    figsize: tuple[int, int] = (12, 10),
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    max_samples: int = 5000,
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    point_size: int = 20,
) -> plt.Figure:
    """
    Будує 2D t-SNE візуалізацію embedding-простору.

    Аргументи:
        embeddings:   Масив embedding-векторів (N, D).
        labels:       Мітки класів (N,).
        class_names:  Список назв класів.
        title:        Заголовок.
        figsize:      Розмір фігури.
        perplexity:   Параметр perplexity для t-SNE.
        n_iter:       Кількість ітерацій t-SNE.
        random_state: Seed.
        max_samples:  Максимальна кількість точок (>5000 значно уповільнює роботу).
        save_path:    Шлях для збереження.
        alpha:        Прозорість точок.
        point_size:   Розмір точок.

    Повертає:
        matplotlib.figure.Figure.
    """
    from sklearn.manifold import TSNE

    import torch
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.detach().cpu().numpy()
    else:
        emb_np = np.asarray(embeddings)

    labels_np = np.asarray(labels)

    # Субвибірка при великій кількості зразків
    n_samples = len(emb_np)
    if n_samples > max_samples:
        logger.info(f"t-SNE: субвибірка {max_samples} з {n_samples} зразків")
        rng = np.random.default_rng(random_state)

        # Стратифікована вибірка: рівна кількість з кожного класу
        selected_idx = []
        unique_labels = np.unique(labels_np)
        per_class = max(1, max_samples // len(unique_labels))

        for label in unique_labels:
            class_idx = np.where(labels_np == label)[0]
            n_select  = min(per_class, len(class_idx))
            selected  = rng.choice(class_idx, n_select, replace=False)
            selected_idx.extend(selected.tolist())

        selected_idx = np.array(selected_idx)
        emb_np    = emb_np[selected_idx]
        labels_np = labels_np[selected_idx]

    # t-SNE проєкція
    logger.info(f"Обчислення t-SNE для {len(emb_np)} зразків...")
    import sklearn
    from packaging import version as pkg_version

    tsne_kwargs = dict(
        n_components=2,
        perplexity=min(perplexity, len(emb_np) - 1),
        random_state=random_state,
        verbose=0,
    )
    # sklearn >=1.4 перейменувало n_iter → max_iter; мінімальне значення = 250
    n_iter_safe = max(250, n_iter)
    if pkg_version.parse(sklearn.__version__) >= pkg_version.parse("1.4"):
        tsne_kwargs["max_iter"] = n_iter_safe
    else:
        tsne_kwargs["n_iter"] = n_iter_safe

    tsne = TSNE(**tsne_kwargs)
    emb_2d = tsne.fit_transform(emb_np)  # (N, 2)

    # Побудова графіку
    num_classes = len(class_names)
    colors = plt.cm.get_cmap("tab20", num_classes) if num_classes <= 20 else \
             plt.cm.get_cmap("hsv", num_classes)

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx, class_name in enumerate(class_names):
        mask = labels_np == class_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            label=class_name,
            color=colors(class_idx),
            alpha=alpha,
            s=point_size,
            linewidths=0,
        )

    ax.set_xlabel("t-SNE вісь 1")
    ax.set_ylabel("t-SNE вісь 2")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Легенда (поза графіком для великої кількості класів)
    if num_classes <= 20:
        ax.legend(
            markerscale=2,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=True,
            fontsize=9,
        )
    else:
        logger.warning(f"Більше 20 класів ({num_classes}): легенда прихована")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"t-SNE збережено: {save_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Розподіл помилок
# ──────────────────────────────────────────────────────────────────────────────

def plot_distance_distribution(
    distances_km: Union[np.ndarray, list[float]],
    title: str = "Розподіл відстані помилок",
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Гістограма та CDF розподілу помилок геолокації.

    Аргументи:
        distances_km: Масив відстаней у км.
        title:        Заголовок.
        figsize:      Розмір фігури.
        save_path:    Шлях для збереження.

    Повертає:
        matplotlib.figure.Figure.
    """
    distances = np.asarray(distances_km)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Гістограма
    ax1.hist(
        distances,
        bins=50,
        color="#2196F3",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.axvline(np.mean(distances),   color="red",    linestyle="--", linewidth=2,
                label=f"Середня: {np.mean(distances):.1f} км")
    ax1.axvline(np.median(distances), color="orange", linestyle="-",  linewidth=2,
                label=f"Медіана: {np.median(distances):.1f} км")
    ax1.set_xlabel("Відстань (км)")
    ax1.set_ylabel("Кількість зразків")
    ax1.set_title("Гістограма відстаней")
    ax1.legend()

    # CDF
    sorted_d = np.sort(distances)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)

    ax2.plot(sorted_d, cdf * 100, color="#4CAF50", linewidth=2)

    # Горизонтальні лінії для ключових порогів
    for threshold, label in [(25, "25 км"), (200, "200 км"), (750, "750 км")]:
        pct = np.mean(distances <= threshold) * 100
        ax2.axvline(threshold, color="gray", linestyle=":", alpha=0.7)
        ax2.text(threshold, pct + 2, f"{pct:.1f}%", ha="center", fontsize=9, color="gray")

    ax2.set_xlabel("Відстань (км)")
    ax2.set_ylabel("CDF (%)")
    ax2.set_title("CDF (накопичений розподіл)")
    ax2.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Розподіл відстаней збережено: {save_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Порівняння моделей
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results: dict[str, dict],
    metrics: list[str] = ("top1_accuracy", "top5_accuracy", "mean_geoscore"),
    figsize: tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Порівняльна діаграма кількох моделей за метриками.

    Аргументи:
        results:  Словник {model_name: {metric_name: value, ...}}.
        metrics:  Список метрик для порівняння.
        figsize:  Розмір фігури.
        save_path: Шлях для збереження.

    Повертає:
        matplotlib.figure.Figure.

    Приклад:
        results = {
            "BaselineCNN":  {"top1_accuracy": 0.72, "mean_geoscore": 3200},
            "StreetCLIP":   {"top1_accuracy": 0.81, "mean_geoscore": 3800},
            "GeoCLIP":      {"top1_accuracy": 0.85, "mean_geoscore": 4100},
        }
    """
    model_names = list(results.keys())
    n_metrics = len(metrics)

    metric_labels = {
        "top1_accuracy":      "Top-1 Accuracy",
        "top5_accuracy":      "Top-5 Accuracy",
        "mean_geoscore":      "Mean GeoScore",
        "mean_distance_km":   "Mean Distance (km)",
        "median_distance_km": "Median Distance (km)",
    }

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        values = [results[m].get(metric, 0) for m in model_names]

        bars = ax.bar(model_names, values, color=colors)

        # Підписи значень
        for bar, val in zip(bars, values):
            label = f"{val*100:.1f}%" if metric.endswith("accuracy") else f"{val:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                label,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title(metric_labels.get(metric, metric), fontsize=12)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_ylim(0, max(values) * 1.15)

        # Обертання підписів для зручності
        ax.set_xticklabels(model_names, rotation=15, ha="right")

    fig.suptitle("Порівняння моделей", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Порівняння моделей збережено: {save_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Тест при прямому запуску
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    print("=== Тест візуалізації ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Матриця плутанини
        np.random.seed(42)
        n, c = 200, 5
        y_true = np.random.randint(0, c, n)
        y_pred = np.random.randint(0, c, n)
        class_names = ["Київ", "Харків", "Одеса", "Дніпро", "Львів"]

        fig = plot_confusion_matrix(
            y_true, y_pred, class_names,
            save_path=str(tmpdir / "confusion.png")
        )
        plt.close(fig)
        print("  Матриця плутанини — OK")

        # Криві навчання
        epochs = 30
        train_losses = np.random.exponential(1.5, epochs).tolist()
        val_losses   = (np.array(train_losses) + np.random.normal(0.1, 0.05, epochs)).tolist()
        train_accs   = np.linspace(0.3, 0.85, epochs).tolist()
        val_accs     = (np.array(train_accs) - 0.05).tolist()

        fig = plot_learning_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_path=str(tmpdir / "learning_curves.png")
        )
        plt.close(fig)
        print("  Криві навчання — OK")

        # t-SNE
        embeddings = np.random.randn(500, 64)
        labels = np.random.randint(0, c, 500)

        fig = plot_tsne_embeddings(
            embeddings, labels, class_names,
            n_iter=250,   # Зменшено для тесту
            max_samples=200,
            save_path=str(tmpdir / "tsne.png")
        )
        plt.close(fig)
        print("  t-SNE — OK")

        # Розподіл відстаней
        distances = np.concatenate([
            np.random.exponential(50, 300),
            np.random.exponential(300, 100),
        ])
        fig = plot_distance_distribution(
            distances,
            save_path=str(tmpdir / "distance_dist.png")
        )
        plt.close(fig)
        print("  Розподіл відстаней — OK")

        # Порівняння моделей
        comparison = {
            "BaselineCNN":  {"top1_accuracy": 0.72, "top5_accuracy": 0.91, "mean_geoscore": 3200},
            "StreetCLIP":   {"top1_accuracy": 0.81, "top5_accuracy": 0.95, "mean_geoscore": 3800},
            "GeoCLIP":      {"top1_accuracy": 0.85, "top5_accuracy": 0.97, "mean_geoscore": 4100},
        }
        fig = plot_model_comparison(
            comparison,
            save_path=str(tmpdir / "comparison.png")
        )
        plt.close(fig)
        print("  Порівняння моделей — OK")

    print("Всі тести візуалізації пройдено!")
