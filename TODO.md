# 📋 AI_GeoDetect — Project TODO

> Аналіз стану проекту станом на **квітень 2026**.
> Зроблено: інфраструктура ML-пайплайну (~75%). Залишилось: дані, навчання, результати.

---

## ✅ Вже реалізовано (не чіпати)

### Архітектури моделей (`code/models.py`)
- [x] **BaselineCNN** — EfficientNet-B2 + кастомна голова (Linear→BN→ReLU→Dropout→Linear)
- [x] **StreetCLIPModel** — `geolocal/StreetCLIP` з HuggingFace + frozen encoder + лінійний пробник
- [x] **GeoCLIPModel** — CLIP ViT + RandomFourierGPSEncoder + InfoNCE контрастивний loss
- [x] `build_model()` фабрична функція
- [x] `predict()`, `get_embeddings()`, `retrieve_gps()` для всіх архітектур
- [x] GPS Gallery (build_gallery, retrieve_gps)

### Pipeline навчання (`code/train.py`)
- [x] Двостадійне навчання (frozen backbone → unfreeze last N layers)
- [x] Диференціальні learning rates (голова ×10 від backbone)
- [x] `EarlyStopping` (patience, min_delta)
- [x] `CheckpointManager` (top-K збереження + `best_model.pth`)
- [x] W&B + MLflow логування через уніфікований `Logger`
- [x] Mixed Precision AMP + gradient clipping
- [x] CLI з `argparse` (`--config`, `--arch`, `--epochs`, `--batch-size`)

### Dataset (`code/dataset.py`)
- [x] `GeoDataset` — зчитує CSV-маніфест, фільтр за країною/містом/якістю
- [x] H3-гексагональний split (без data leakage між train/val/test)
- [x] K-means split (fallback якщо h3 не встановлено)
- [x] `get_class_weights()` для збалансованого навчання
- [x] `create_dummy_manifest()` — синтетичний маніфест для тестування
- [x] `create_dataloaders()` фабрична функція

### Evaluation (`code/evaluate.py`)
- [x] Top-1 / Top-5 accuracy
- [x] Haversine distance (mean, median, p25/p75/p90)
- [x] GeoScore (5000 · exp(−d/1492.7))
- [x] Per-class accuracy table
- [x] Збереження результатів у JSON
- [x] CLI (`--checkpoint`, `--manifest`, `--output`)

### Завантаження даних (`code/download_data.py`)
- [x] `download_osv5m_subset()` — HuggingFace datasets API + парквет fallback
- [x] `download_mapillary()` — ZenSVI API + прямий Mapillary Graph API
- [x] `create_manifest()` — маніфест з директорії зображень (з EXIF GPS)
- [x] `print_dataset_stats()` — статистика датасету
- [x] Паралельне завантаження через `ThreadPoolExecutor`

### Аугментації (`code/augmentations.py`)
- [x] `get_train_transforms()` — RandomHorizontalFlip, ColorJitter, RandomRotation тощо
- [x] `get_val_transforms()` — Resize + CenterCrop + Normalize

### Метрики (`code/metrics.py`)
- [x] `top_k_accuracy_torch()` та `top_k_accuracy()`
- [x] `haversine_distance()`
- [x] `geoscore()`
- [x] `compute_all_metrics()`

### Утиліти (`code/utils.py`)
- [x] `get_device()`, `seed_everything()`, `count_parameters()`
- [x] `load_config()` (YAML/JSON)
- [x] `format_param_count()`

### Інфраструктура
- [x] `requirements.txt` — повний список залежностей
- [x] `environment.yml` — conda environment
- [x] `.gitignore`
- [x] `dataset_assembly_guide.md` — інструкція зі збору датасету

---

## 🐛 Відомі баги — потребують виправлення

### КРИТИЧНІ
- [ ] **`dataset.py` L61** — `h3.geo_to_h3()` deprecated у h3 >= 4.x
  ```python
  # ЗЛАМАНО:
  h3.geo_to_h3(row["lat"], row["lon"], h3_resolution)
  # ВИПРАВЛЕННЯ:
  h3.latlng_to_cell(row["lat"], row["lon"], h3_resolution)
  ```

- [ ] **`train.py` — coords_tuple десеріалізація** крихка при `num_workers > 0`
  ```python
  # ПРОБЛЕМА (геть незрозуміла структура при collation):
  lat = torch.tensor([c[0] for c in coords_tuple[0]], ...)
  # ВИПРАВЛЕННЯ: у GeoDataset.__getitem__ повернути dict або окремі тензори
  ```

### НЕ КРИТИЧНІ
- [ ] **`train.py` CheckpointManager** — змінна `best_actual` оголошена і не використана (мертвий код)
- [ ] **`models.py` BaselineCNN.get_embeddings()** — `torch.no_grad()` всередині методу перериває backprop якщо викликається з train-режиму. Прибрати no_grad або задокументувати обмеження
- [ ] **`evaluate.py`** — подвійне завантаження чекпоінту (перший раз у `load_checkpoint()`, другий наприкінці `evaluate()`) — марнотратно для великих моделей

---

## ❌ Що ще потрібно зробити

### 🔴 КРИТИЧНО (без цього захист неможливий)

#### 1. Зібрати реальний датасет
> Слідувати інструкції `dataset_assembly_guide.md`

```bash
# Крок 1: Встановити залежності
pip install huggingface_hub datasets zensvi h3 geopy imagehash \
    scipy pandas pillow tqdm geopandas shapely pyarrow requests

# Крок 2: Отримати API ключі
# HuggingFace: https://huggingface.co/settings/tokens
# Mapillary:   https://www.mapillary.com/dashboard/developers

export HF_TOKEN="hf_xxxxxxxx"
export MAPILLARY_API_KEY="MLY|xxxxxxxx"

# Крок 3: Завантажити OSV-5M (тільки потрібні шарди ~5-7GB)
python code/download_data.py osv5m --countries UA PL CZ HU --output data/raw/osv5m

# Крок 4: Завантажити Mapillary через ZenSVI (по 10k на місто)
python code/download_data.py mapillary \
    --bbox 50.30 30.20 50.60 30.85 \
    --api-key $MAPILLARY_API_KEY \
    --output data/raw/mapillary/kyiv --max-images 10000

# Крок 5: Створити єдиний маніфест
python code/download_data.py manifest \
    --image-dir data/raw --output data/manifest.csv --stats
```

**Очікуваний результат:** ~35 000 зображень, `data/manifest.csv`

#### 2. Запустити навчання хоча б однієї моделі
```bash
# Baseline (найшвидший старт, ~2-3 год на GPU)
python code/train.py \
    --config configs/baseline.yaml \
    --manifest data/manifest.csv \
    --arch baseline \
    --epochs 30 \
    --batch-size 32

# Перевірити що навчання працює на dummy даних:
python -c "
from code.dataset import create_dummy_manifest
create_dummy_manifest('data/test_manifest.csv', n_samples=500)
"
python code/train.py --manifest data/test_manifest.csv --epochs 3 --no-wandb
```

#### 3. Отримати реальні метрики для диплому
```bash
python code/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --manifest data/manifest.csv \
    --output results/eval_results.json
```

**Потрібні для диплому:**
- [ ] Top-1 accuracy по всіх 3 архітектурах
- [ ] Haversine distance (mean/median)
- [ ] GeoScore (порівняти з бейзлайном)
- [ ] Per-class accuracy table (8 міст)

---

### 🟡 ВАЖЛИВО (потрібно для повноцінного репозиторію)

#### 4. Створити YAML конфіги (`configs/` порожня!)
```bash
mkdir -p configs
```

Потрібні файли:
- [ ] `configs/baseline.yaml`
- [ ] `configs/streetclip.yaml`
- [ ] `configs/geoclip.yaml`

Приклад `configs/baseline.yaml`:
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

#### 5. Заповнити `data/` директорію
- [ ] Покласти хоча б `data/manifest.csv` із реальними даними
- [ ] Або зробити `data/sample_manifest.csv` (100 зразків) для демонстрації

#### 6. Виправити баг з h3 API у `dataset.py`
- [ ] Замінити `h3.geo_to_h3()` → `h3.latlng_to_cell()` (рядок ~61)
- [ ] Перевірити: `pip install h3>=4.0` і запустити тест

#### 7. Додати `results/` з хоча б baseline результатами
- [ ] `results/baseline_eval.json`
- [ ] `results/comparison_table.md` (порівняння архітектур)

---

### 🟠 БАЖАНО (підвищить оцінку)

#### 8. Gradio/Streamlit демо для захисту
```bash
pip install gradio
```
- [ ] `demo/app.py` — завантаження фото → передбачення міста + карта
- [ ] Показати топ-5 передбачень з ймовірностями
- [ ] Відобразити на Folium/Leaflet карті

Скелет:
```python
import gradio as gr
from code.inference import predict_city

def predict(image):
    results = predict_city(image, top_k=5)
    return results

demo = gr.Interface(fn=predict, inputs="image", outputs="label")
demo.launch()
```

#### 9. Notebook для візуалізації (`notebooks/`)
- [ ] `notebooks/01_data_exploration.ipynb` — розподіл датасету, bboxes на карті
- [ ] `notebooks/02_training_curves.ipynb` — loss/accuracy криві з W&B
- [ ] `notebooks/03_error_analysis.ipynb` — t-SNE embeddings, confusion matrix

#### 10. Юніт-тести
- [ ] `tests/test_models.py` — перевірити forward pass кожної архітектури
- [ ] `tests/test_dataset.py` — перевірити split, фільтрацію, class_weights
- [ ] `tests/test_metrics.py` — перевірити haversine, geoscore
```bash
pip install pytest
pytest tests/ -v
```

---

## 📦 Що ще потрібно встановити / скачати

| Пакет | Навіщо | Команда |
|---|---|---|
| `h3>=4.0` | Виправлення бага geo_to_h3 | `pip install h3>=4.0` |
| `zensvi` | Завантаження Mapillary | `pip install zensvi` |
| `deface` | Розмиття облич (GDPR) | `pip install deface` |
| `wandb` | Логування навчання | `pip install wandb` |
| `gradio` | Демо для захисту | `pip install gradio` |
| `folium` | Карти у notebook | `pip install folium` |
| `pytest` | Юніт-тести | `pip install pytest` |
| `cv2` | Фільтр розмитих фото | `pip install opencv-python` |

---

## 🗓️ Рекомендований порядок дій до захисту

```
1. Виправити баг h3.geo_to_h3()                     [30 хв]
2. Отримати API ключі (HF + Mapillary)               [15 хв]
3. Завантажити датасет (OSV-5M + Mapillary)          [3-5 год]
4. Дедупліковати + фільтрувати якість               [1 год]
5. Створити configs/baseline.yaml                    [15 хв]
6. Запустити baseline навчання                       [2-4 год GPU]
7. evaluate.py → results/baseline_eval.json          [30 хв]
8. Повторити для StreetCLIP                          [3-5 год GPU]
9. Зробити Gradio демо                               [2 год]
10. Оновити README з реальними метриками             [1 год]
```

---

*Останнє оновлення: квітень 2026*
*Поточна готовність: ~75% (інфраструктура) | ~20% (дані та результати)*
