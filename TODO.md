# 📋 AI_GeoDetect — Project TODO

> Стан на **квітень 2026** · Готовність: ~85% (інфраструктура + баги виправлені)
> Залишилось: дані, навчання, результати, демо
> Датасет: див. `TODO_DATASET.md`

---

## ✅ Реалізовано

### Архітектури (`code/models.py`)
- [x] **BaselineCNN** — EfficientNet-B2 + кастомна голова
- [x] **StreetCLIPModel** — `geolocal/StreetCLIP` + frozen encoder + лінійний пробник
- [x] **GeoCLIPModel** — CLIP ViT + RandomFourierGPSEncoder + InfoNCE loss
- [x] `build_model()`, `predict()`, `get_embeddings()`, `retrieve_gps()`

### Pipeline (`code/train.py`)
- [x] Двостадійне навчання + диференціальні lr
- [x] EarlyStopping, CheckpointManager (top-K + `best_model.pth`)
- [x] W&B + MLflow + AMP + gradient clipping
- [x] CLI (`--config`, `--arch`, `--epochs`)

### Інше
- [x] `code/dataset.py` — GeoDataset, H3/K-means split, class weights
- [x] `code/evaluate.py` — Top-1/5, Haversine, GeoScore, per-class table
- [x] `code/download_data.py` — OSV-5M + Mapillary + manifest
- [x] `code/augmentations.py`, `code/metrics.py`, `code/utils.py`
- [x] `requirements.txt`, `environment.yml`, `.gitignore`
- [x] `dataset_assembly_guide.md`, `TODO_DATASET.md`

### Баги — всі виправлено ✅
- [x] `dataset.py` — `h3.geo_to_h3()` → `h3.latlng_to_cell()` + fallback
- [x] `train.py` — `coords_tuple` → чистий `torch.Tensor(N, 2)`
- [x] `models.py` — `get_embeddings()`: прибрано `torch.no_grad()` всереди
- [x] `train.py` — мертвий код `best_actual` видалено

---

## 🔴 КРИТИЧНО (без цього захист неможливий)

#### 1. Зібрати датасет → див. `TODO_DATASET.md`
- [ ] Запустити RunPod Pod (PyTorch 2.7 + CUDA 12.8, Volume 20GB)
- [ ] Завантажити OSV-5M для UA/PL/CZ/HU (~3–6GB)
- [ ] H3-розбиття 70/15/15 → `data/manifests/train.csv`, `val.csv`, `test.csv`

#### 2. Створити YAML конфіги (`configs/` порожня!)
- [ ] `configs/baseline.yaml`
- [ ] `configs/streetclip.yaml`
- [ ] `configs/geoclip.yaml`

<details>
<summary>Приклад configs/baseline.yaml</summary>

```yaml
architecture: baseline
pretrained: true
stage1_epochs: 10
stage1_lr: 0.001
stage2_epochs: 20
stage2_lr: 0.0001
stage2_unfreeze_n: 3
batch_size: 32
num_workers: 4
weight_decay: 0.01
patience: 7
img_size: 224
split_method: h3
mixed_precision: true
use_wandb: false
seed: 42
```
</details>

#### 3. Запустити навчання
```bash
# Baseline (~2-3 год на RTX 5090)
python code/train.py --config configs/baseline.yaml

# StreetCLIP
python code/train.py --config configs/streetclip.yaml

# GeoCLIP
python code/train.py --config configs/geoclip.yaml
```
- [ ] Baseline навчання завершено, `checkpoints/best_model.pth` існує
- [ ] StreetCLIP навчання завершено
- [ ] GeoCLIP навчання завершено

#### 4. Отримати метрики для диплому
```bash
python code/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --manifest data/manifests/test.csv \
    --output results/baseline_eval.json
```
- [ ] Top-1 / Top-5 accuracy для всіх 3 архітектур
- [ ] Haversine distance (mean / median)
- [ ] GeoScore
- [ ] Per-class accuracy (8 міст)
- [ ] `results/comparison_table.md` — порівняння архітектур

---

## 🟡 БАЖАНО (підвищить оцінку)

#### 5. Gradio демо для захисту
- [ ] `demo/app.py` — завантажити фото → топ-5 міст + карта Folium

```python
import gradio as gr
from code.inference import predict_city

def predict(image):
    return predict_city(image, top_k=5)

gr.Interface(fn=predict, inputs="image", outputs="label").launch()
```

#### 6. Notebooks
- [ ] `notebooks/01_data_exploration.ipynb` — розподіл датасету + bbox на карті
- [ ] `notebooks/02_training_curves.ipynb` — loss/accuracy криві
- [ ] `notebooks/03_error_analysis.ipynb` — t-SNE, confusion matrix

#### 7. Тести
- [ ] `tests/test_models.py` — forward pass кожної архітектури
- [ ] `tests/test_dataset.py` — split, фільтрація, class_weights
- [ ] `tests/test_metrics.py` — haversine, geoscore

---

## 🗓️ Порядок дій до захисту

```
1. configs/baseline.yaml + streetclip.yaml + geoclip.yaml   [30 хв]
2. Зібрати датасет (TODO_DATASET.md)                      [3-5 год]
3. Baseline навчання + evaluate.py                         [3-4 год GPU]
4. StreetCLIP + GeoCLIP навчання                          [5-8 год GPU]
5. results/comparison_table.md                             [30 хв]
6. Gradio демо                                              [2 год]
7. Notebooks (візуалізація)                                [2 год]
```

---

*Останнє оновлення: квітень 2026*
*Готовність: ~85% — інфраструктура готова, баги виправлені, залишаються дані + навчання*
