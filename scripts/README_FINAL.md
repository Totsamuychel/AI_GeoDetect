# Dataset Download Guide (Final Version)

Финальное руководство по скачиванию датасета для диплома.

## 📊 Стратегия скачивания данных

После тестирования различных источников, оптимальная стратегия:

### ✅ Основной источник: Global Streetscapes
- **Репозиторий:** https://huggingface.co/datasets/NUS-UAL/global-streetscapes
- **Покрытие:** Хорошее для всех трех городов
- **Преимущества:**
  - Streaming mode (экономия места)
  - Высокое качество изображений
  - Координаты в метаданных

### ⚠️ OSV-5M - НЕ используется
- **Причина:** Очень плохое покрытие для Восточной Европы
- **Результаты тестирования:**
  - Kyiv: 0 фото ❌
  - Warsaw: ~50 фото
  - Prague: ~50 фото
  - **Итого: ~100 из требуемых 7500** ❌

### ✅ Дополнительно: Mapillary (опционально)
- Если нужно дополнить данные
- Требует API ключ

---

## 🚀 Быстрый старт

### Вариант 1: Автоматический (рекомендуется)

```bash
# Один скрипт для всего процесса
python scripts/download_dataset_simple.py
```

### Вариант 2: Поэтапный

```bash
# Этап 1: Получить полигоны городов
python scripts/01_get_city_polygons.py

# Этап 2: Скачать Global Streetscapes
python scripts/04_download_global_streetscapes.py

# Этап 3 (опционально): Объединить с Mapillary
python scripts/05_merge_manifests.py
```

---

## 📋 Требования

### 1. Python библиотеки

```bash
pip install datasets osmnx geopandas shapely pandas tqdm
```

### 2. HuggingFace Token

```bash
# Получить токен: https://huggingface.co/settings/tokens

# Windows PowerShell
$env:HF_TOKEN="hf_your_token_here"

# Windows CMD
set HF_TOKEN=hf_your_token_here

# Linux/Mac
export HF_TOKEN="hf_your_token_here"
```

---

## 📁 Структура скриптов

| Скрипт | Описание | Статус |
|--------|----------|--------|
| `01_get_city_polygons.py` | Получение границ городов из OSM | ✅ Обязательно |
| `04_download_global_streetscapes.py` | Скачивание из Global Streetscapes | ✅ Обязательно |
| `05_merge_manifests.py` | Объединение с Mapillary | ⚠️ Опционально |
| `download_dataset_simple.py` | Master-скрипт (все в одном) | ✅ Рекомендуется |

### ❌ Устаревшие скрипты (не используются)

- `02_filter_osv5m.py` - OSV-5M имеет плохое покрытие
- `03_download_osv5m_images.py` - Не нужен
- `download_dataset.py` - Старый master-скрипт с OSV-5M

---

## ⏱️ Ожидаемое время

| Этап | Время |
|------|-------|
| Получение полигонов | 30 сек |
| Global Streetscapes (7500 фото) | 2-5 часов |
| Объединение манифестов | 1 мин |
| **ИТОГО** | **2-5 часов** |

---

## 📊 Ожидаемые результаты

```
dataset/
└── raw/
    ├── boundaries/           # Полигоны городов
    │   ├── kyiv.geojson
    │   ├── warsaw.geojson
    │   └── prague.geojson
    │
    └── global_streetscapes/ # Основные данные
        ├── images/
        │   ├── kyiv/        # ~2500 фото
        │   ├── warsaw/      # ~2500 фото
        │   └── prague/      # ~2500 фото
        └── manifest.csv     # ~7500 записей
```

---

## 🔍 Проверка результатов

### После скачивания проверьте:

```bash
# Количество файлов
python -c "
from pathlib import Path
for city in ['kyiv', 'warsaw', 'prague']:
    path = Path(f'dataset/raw/global_streetscapes/images/{city}')
    count = len(list(path.glob('*.jpg'))) if path.exists() else 0
    print(f'{city}: {count} files')
"
```

### Ожидаемый вывод:
```
kyiv: 2500 files
warsaw: 2500 files
prague: 2500 files
```

---

## ❓ Устранение проблем

### Проблема: Мало изображений для города

**Причина:** В Global Streetscapes может быть недостаточно данных для конкретного города.

**Решение:** Дополните данными из Mapillary:

```bash
# Для Киева (пример)
python code/download_data.py mapillary \
    --bbox 50.21 30.24 50.59 30.83 \
    --api-key YOUR_MAPILLARY_KEY \
    --max-images 2000

# Объедините
python scripts/05_merge_manifests.py
```

### Проблема: Очень медленно

**Решение:**
- Запустите на ночь
- Используйте более быстрый интернет
- Проверьте VPN если есть блокировки HuggingFace

### Проблема: "HF_TOKEN not found"

**Решение:**
```bash
# Проверьте
python -c "import os; print(os.getenv('HF_TOKEN'))"

# Установите
export HF_TOKEN="hf_..."
```

---

## 📝 Следующие шаги после скачивания

### 1. Создать train/val/test splits

```bash
python scripts/generate_manifests.py \
    --input dataset/raw/global_streetscapes/manifest.csv \
    --output-dir dataset/manifests
```

### 2. Проверить данные

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Начать обучение

```bash
# Baseline
python code/train.py --config configs/baseline.yaml

# StreetCLIP
python code/train.py --config configs/streetclip.yaml

# GeoCLIP
python code/train.py --config configs/geoclip.yaml
```

---

## 📈 Сравнение источников данных

| Источник | Покрытие | Качество | Скорость | Итого |
|----------|----------|----------|----------|-------|
| **Global Streetscapes** | ✅ Отлично | ✅ Высокое | ⚠️ Средняя | **Рекомендуется** |
| OSV-5M | ❌ Очень плохо | ✅ Хорошее | ✅ Быстро | **Не использовать** |
| Mapillary | ✅ Хорошо | ✅ Высокое | ⚠️ Требует API | **Для дополнения** |

---

## 🎯 Итоговая рекомендация

1. **Используйте только Global Streetscapes** как основной источник
2. **OSV-5M пропустите** - тратить время не стоит (покрытие <1%)
3. **Mapillary добавьте** если нужно дополнить конкретный город

**Команда:**
```bash
python scripts/download_dataset_simple.py
```

И вы получите **~7500 качественных изображений** для обучения! 🚀

---

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте HF_TOKEN
2. Убедитесь в наличии полигонов (step 1)
3. Проверьте логи скрипта
4. См. `README_GLOBAL_STREETSCAPES.md` для деталей
