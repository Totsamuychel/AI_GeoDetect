# AI_GeoDetect - Руководство по использованию

## 1. Исправление меток городов

### Проблема
Манифесты содержат коды стран ("PL", "UA") вместо названий городов в колонке `city`.

### Решение

```bash
# Исправить метки в основном манифесте
python scripts/fix_city_labels.py \
    --input dataset/raw/osv5m/manifest.csv \
    --output dataset/raw/osv5m/manifest_fixed.csv \
    --delay 1.0

# После исправления - пересоздать train/val/test splits
python scripts/generate_manifests.py \
    --input dataset/raw/osv5m/manifest_fixed.csv \
    --output-dir dataset/manifests
```

### Параметры скрипта `fix_city_labels.py`:

- `--input`: Входной CSV файл с манифестом
- `--output`: Выходной CSV файл с исправленными метками
- `--delay`: Задержка между API запросами (по умолчанию 1.0 сек для Nominatim)
- `--no-filter`: Не фильтровать по целевым городам (оставить все)
- `--no-cache`: Не сохранять кэш geocoding

### Как работает:

1. **Reverse Geocoding**: Использует Nominatim API (OpenStreetMap) для определения города по координатам
2. **Кэширование**: Сохраняет результаты в `dataset/.cache/geocode_cache.csv` для повторного использования
3. **Нормализация**: Приводит названия к стандартным формам (Київ → Kyiv, Warszawa → Warsaw)
4. **Фильтрация**: Оставляет только целевые города:
   - UA: Kyiv, Lviv, Odesa, Kharkiv, Dnipro
   - PL: Warsaw, Kraków, Gdańsk
   - CZ: Prague
   - HU: Budapest
   - RO: Bucharest

### Важно:

- Nominatim API требует 1 запрос/секунду (параметр `--delay`)
- Для 7650 изображений потребуется ~2 часа
- Кэш значительно ускоряет повторные запуски
- **User-Agent** должен быть уникальным (по умолчанию: "AI_GeoDetect_Diploma_Project")

---

## 2. Анализ результатов обучения

### Ноутбуки для анализа:

#### `notebooks/02_training_curves.ipynb`

**Визуализация кривых обучения:**
- Loss curves (train/val)
- Accuracy curves (top-1, top-3, top-5)
- Geospatial metrics (Haversine, GeoScore)
- Learning rate schedule
- Сравнительная таблица моделей

**Запуск:**
```bash
jupyter notebook notebooks/02_training_curves.ipynb
```

**Требования:**
- Обученные модели в `results/{baseline,streetclip,geoclip}/`
- Файл `training_history.json` или `metrics.json` в директории результатов

**Генерируемые файлы:**
- `results/plots/loss_curves.png`
- `results/plots/accuracy_curves.png`
- `results/plots/geospatial_metrics.png`
- `results/plots/learning_rate.png`
- `results/model_comparison.csv`

---

#### `notebooks/03_error_analysis.ipynb`

**Анализ ошибок и эмбеддингов:**
- Confusion matrix по городам
- Per-class performance (precision, recall, F1)
- Geographic error analysis (распределение расстояний)
- **t-SNE визуализация** эмбеддингов
- Heatmap ошибок на интерактивной карте
- Примеры успешных и неуспешных предсказаний

**Запуск:**
```bash
jupyter notebook notebooks/03_error_analysis.ipynb
```

**Требования:**
- Обученная модель (checkpoint): `checkpoints/{model}/best_model.pth`
- Тестовый манифест: `dataset/manifests/test.csv`
- GPU рекомендуется (для быстрой генерации эмбеддингов)

**Генерируемые файлы:**
- `results/plots/confusion_matrix.png`
- `results/plots/error_distance_distribution.png`
- `results/plots/tsne_embeddings.png` ⭐
- `results/plots/error_heatmap.html`
- `results/plots/example_predictions.png`
- `results/classification_report.csv`

---

## 3. Типичный workflow для диплома

### Шаг 1: Подготовка данных

```bash
# 1. Исправить метки городов
python scripts/fix_city_labels.py \
    --input dataset/raw/osv5m/manifest.csv \
    --output dataset/raw/osv5m/manifest_fixed.csv

# 2. Пересоздать splits с исправленными метками
python scripts/generate_manifests.py \
    --input dataset/raw/osv5m/manifest_fixed.csv \
    --output-dir dataset/manifests

# 3. Проверить статистику
python code/dataset.py --manifest dataset/manifests/train.csv
```

### Шаг 2: Обучение моделей

```bash
# Baseline (EfficientNet-B2)
python code/train.py --config configs/baseline.yaml

# StreetCLIP
python code/train.py --config configs/streetclip.yaml

# GeoCLIP
python code/train.py --config configs/geoclip.yaml
```

### Шаг 3: Оценка моделей

```bash
# Оценка на тестовом наборе
python code/evaluate.py \
    --checkpoint checkpoints/baseline/best_model.pth \
    --manifest dataset/manifests/test.csv \
    --output results/baseline/test_metrics.json
```

### Шаг 4: Анализ результатов

```bash
# Запуск Jupyter для анализа
jupyter notebook

# Открыть ноутбуки:
# - notebooks/02_training_curves.ipynb (кривые обучения)
# - notebooks/03_error_analysis.ipynb (ошибки + t-SNE)
```

### Шаг 5: Экспорт для диплома

Все графики сохраняются в `results/plots/` в высоком разрешении (300 DPI):

**Для главы "Эксперименты":**
- `loss_curves.png`
- `accuracy_curves.png`
- `geospatial_metrics.png`
- `model_comparison.csv` (таблица сравнения)

**Для главы "Анализ ошибок":**
- `confusion_matrix.png`
- `tsne_embeddings.png` ⭐ (кластеризация городов)
- `error_distance_distribution.png`
- `example_predictions.png`

**Интерактивные:**
- `error_heatmap.html` (открыть в браузере)

---

## 4. Устранение проблем

### Ошибка: "No training history found"

**Решение:**
- Убедитесь, что модели обучены
- Проверьте наличие `training_history.json` в `results/{model}/`
- Если используете W&B, экспортируйте историю в JSON

### Ошибка: "GeocoderTimedOut"

**Решение:**
- Увеличьте `--delay` до 2.0 секунд
- Используйте кэш из предыдущих запусков
- Разбейте манифест на части и обрабатывайте по частям

### Ошибка: "CUDA out of memory" при t-SNE

**Решение:**
- Уменьшите `n_samples` в ноутбуке (по умолчанию 2000)
- Используйте CPU вместо GPU для inference
- Уменьшите `batch_size` в DataLoader

### Медленная работа t-SNE

**Решение:**
- Используйте меньше образцов (`n_samples=1000`)
- Уменьшите `n_iter` (по умолчанию 1000 → 500)
- Используйте `perplexity=15-30` для баланса скорости/качества

---

## 5. Полезные команды

```bash
# Проверить количество изображений по городам
python -c "import pandas as pd; df=pd.read_csv('dataset/manifests/train.csv'); print(df['city'].value_counts())"

# Визуализировать примеры из датасета
python code/visualize.py --manifest dataset/manifests/train.csv --n-samples 16

# Запустить demo приложение
python demo/app.py --checkpoint checkpoints/baseline/best_model.pth --share

# Экспортировать модель в ONNX
python code/export_onnx.py --checkpoint checkpoints/baseline/best_model.pth
```

---

## 6. Требования к окружению

### Для исправления меток:
```bash
pip install pandas geopy tqdm
```

### Для ноутбуков:
```bash
pip install jupyter matplotlib seaborn scikit-learn folium
```

### Полный список:
```bash
pip install -r requirements.txt
```

---

## Контакты

При возникновении вопросов обращайтесь к документации:
- `README.md` - общее описание проекта
- `writing/chapters_3_5.md` - детали реализации
- `TODO.md` - список задач
