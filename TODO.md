# 📋 AI_GeoDetect — Project TODO

> Стан на **квітень 2026** · Готовність: ~90% (інфраструктура + баги виправлені)
> Залишилось: дані, навчання, результати
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
- [x] `configs/` — `baseline.yaml`, `streetclip.yaml`, `geoclip.yaml`
- [x] `demo/app.py` — Gradio демо для захисту
- [x] `tests/` — юніт-тести для моделей, датасету та метрик

### Баги — всі виправлено ✅
- [x] `dataset.py` — `h3.geo_to_h3()` → `h3.latlng_to_cell()` + fallback
- [x] `train.py` — `coords_tuple` → чистий `torch.Tensor(N, 2)`
- [x] `models.py` — `get_embeddings()`: прибрано `torch.no_grad()` всереди
- [x] `train.py` — мертвий код `best_actual` видалено

---

## 🔴 КРИТИЧНО (без цього захист неможливий)

#### 1. Зібрати датасет → див. `TODO_DATASET.md`
- [x] Запустити RunPod Pod (PyTorch 2.7 + CUDA 12.8, Volume 20GB) — *Виконано локально (win32)*
- [x] Завантажити OSV-5M для UA/PL/RO (~7700 фото)
- [x] H3-розбиття 70/15/15 → `dataset/manifests/train.csv`, `val.csv`, `test.csv`

#### 2. Запустити навчання
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

#### 3. Отримати метрики для диплому
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

#### 4. Notebooks
- [ ] `notebooks/01_data_exploration.ipynb` — розподіл датасету + bbox на карті
- [ ] `notebooks/02_training_curves.ipynb` — loss/accuracy криві
- [ ] `notebooks/03_error_analysis.ipynb` — t-SNE, confusion matrix

---

## 🗓️ Порядок дій до захисту

```
1. Зібрати датасет (TODO_DATASET.md)                      [3-5 год]
2. Baseline навчання + evaluate.py                         [3-4 год GPU]
3. StreetCLIP + GeoCLIP навчання                          [5-8 год GPU]
4. results/comparison_table.md                             [30 хв]
5. Notebooks (візуалізація)                                [2 год]
```

---

*Останнє оновлення: квітень 2026*
*Готовність: ~90% — інфраструктура готова, баги виправлені, залишаються дані + навчання*
