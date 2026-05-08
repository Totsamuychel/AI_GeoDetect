# Work Notes - AI_GeoDetect Diploma Project

## ✅ ИСПРАВЛЕНО (2026-05-08)

### 1. Data Quality Issue - City Labels Fixed
**Проблема:** Манифесты содержали коды стран ("PL", "UA") вместо названий городов.

**Решение:** Создан скрипт `scripts/fix_city_labels.py`:
- Использует Nominatim reverse geocoding для определения городов по координатам
- Нормализует названия (Київ → Kyiv, Warszawa → Warsaw)
- Фильтрует по целевым городам (Kyiv, Lviv, Warsaw, Prague, Budapest и др.)
- Кэширует результаты в `dataset/.cache/geocode_cache.csv`

**Использование:**
```bash
python scripts/fix_city_labels.py \
    --input dataset/raw/osv5m/manifest.csv \
    --output dataset/raw/osv5m/manifest_fixed.csv
```

### 2. Missing Analysis Notebooks - Created
**Проблема:** Отсутствовали ноутбуки для анализа результатов обучения.

**Решение:** Созданы два новых ноутбука:

#### `notebooks/02_training_curves.ipynb`
- Визуализация кривых обучения (loss, accuracy)
- Geospatial metrics (Haversine, GeoScore)
- Learning rate schedule
- Сравнительная таблица моделей

#### `notebooks/03_error_analysis.ipynb`
- Confusion matrix
- Per-class performance analysis
- Geographic error analysis
- **t-SNE visualization of embeddings** ⭐
- Error heatmap on interactive map
- Example predictions (success/failure)

**Генерируемые файлы для диплома:**
- `results/plots/loss_curves.png`
- `results/plots/accuracy_curves.png`
- `results/plots/confusion_matrix.png`
- `results/plots/tsne_embeddings.png`
- `results/plots/error_heatmap.html`
- `results/model_comparison.csv`

### 3. Documentation - Usage Guide
Создан `USAGE_GUIDE.md` с подробными инструкциями:
- Как исправить метки городов
- Как запустить ноутбуки для анализа
- Типичный workflow для диплома
- Устранение проблем
- Полезные команды

---

## ⚠️ ТЕХНИЧЕСКИЕ ЗАМЕТКИ (TODO)

### 1. train.py — sys.path не добавлен (критично)
train.py импортирует from augmentations import ... без sys.path . Добавь в самое начало файла, сразу после from __future__ import annotations:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### 2. train.py — дефолтные пути в TrainConfig и argparse всё ещё data/ (важно)
Строки 74–75 и ~430 — конфиги YAML ты обновил, но если кто-то запустит без --config или через argparse напрямую, возьмутся старые дефолты:

```python
# В dataclass TrainConfig:
manifest_path: str = "dataset/manifests/train.csv"   # было data/manifest.csv
image_root:    str = "dataset/raw/osv5m/images"       # было data/images

# В argparse:
parser.add_argument("--manifest", default="dataset/manifests/train.csv")
parser.add_argument("--image-root", default="dataset/raw/osv5m/images")
```

### 3. train.py — train_frac / val_frac не передаются в create_dataloaders (логика)
В функции train() вызов create_dataloaders() не передаёт эти параметры, используются дефолты 0.7/0.15. Если в конфиге нет этих полей — это ок, но лучше явно:

```python
dataloaders = create_dataloaders(
    ...
    train_frac=getattr(config, 'train_frac', 0.7),
    val_frac=getattr(config, 'val_frac', 0.15),
)
```

### 4. Создать пустой code/__init__.py (опционально, но хорошая практика)
```bash
touch code/__init__.py
```

---

## 🎯 NEXT STEPS

### Критические задачи для защиты диплома:

1. **Исправить манифесты с правильными метками городов:**
   ```bash
   python scripts/fix_city_labels.py \
       --input dataset/raw/osv5m/manifest.csv \
       --output dataset/raw/osv5m/manifest_fixed.csv

   python scripts/generate_manifests.py \
       --input dataset/raw/osv5m/manifest_fixed.csv \
       --output-dir dataset/manifests
   ```

2. **Обучить все три модели:**
   ```bash
   python code/train.py --config configs/baseline.yaml
   python code/train.py --config configs/streetclip.yaml
   python code/train.py --config configs/geoclip.yaml
   ```

3. **Провести анализ в ноутбуках:**
   ```bash
   jupyter notebook notebooks/02_training_curves.ipynb
   jupyter notebook notebooks/03_error_analysis.ipynb
   ```

4. **Экспортировать результаты для диплома:**
   - Графики из `results/plots/`
   - Таблицу сравнения из `results/model_comparison.csv`
   - t-SNE визуализацию для главы "Анализ результатов"