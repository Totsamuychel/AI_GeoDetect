# OSV-5M Dataset Download Scripts

Скрипты для скачивания и подготовки датасета OSV-5M с точной географической фильтрацией по границам городов.

## 🎯 Цель

Скачать **~2000-2500 фотографий на город** из OSV-5M для трех городов:
- **Kyiv** (Киев, Украина)
- **Warsaw** (Варшава, Польша)
- **Prague** (Прага, Чехия)

**Критерий качества:** Только городская застройка, **БЕЗ трасс, пригородов и сельских территорий**.

---

## 📋 Требования

### Системные требования
- Python 3.10+
- ~5-10 GB свободного места на диске
- Стабильное интернет-соединение

### Python библиотеки

```bash
pip install osmnx geopandas shapely pandas huggingface_hub tqdm
```

Или установите полный набор зависимостей:
```bash
pip install -r ../requirements.txt
```

### HuggingFace Token

OSV-5M требует аутентификации. Получите токен:

1. Зарегистрируйтесь на https://huggingface.co/
2. Создайте токен: https://huggingface.co/settings/tokens
3. Установите переменную окружения:

```bash
# Linux/Mac
export HF_TOKEN="your_token_here"

# Windows CMD
set HF_TOKEN=your_token_here

# Windows PowerShell
$env:HF_TOKEN="your_token_here"
```

---

## 🚀 Быстрый старт

### Вариант 1: Запустить все этапы сразу

```bash
python scripts/download_dataset.py
```

Этот мастер-скрипт автоматически выполнит все 3 этапа:
1. Получение полигонов городов
2. Фильтрация метаданных OSV-5M
3. Скачивание изображений

### Вариант 2: Поэтапный запуск

Если хотите контролировать каждый этап:

```bash
# Этап 1: Получить полигоны городов
python scripts/01_get_city_polygons.py

# Этап 2: Отфильтровать метаданные
python scripts/02_filter_osv5m.py

# Этап 3: Скачать изображения
python scripts/03_download_osv5m_images.py
```

---

## 📁 Структура скриптов

### 01_get_city_polygons.py

**Цель:** Получить точные административные границы городов через OpenStreetMap.

**Что делает:**
- Использует библиотеку `osmnx` для запроса границ городов
- Сохраняет полигоны в формате GeoJSON
- Это позволяет отфильтровать фото на трассах и в пригородах

**Выход:**
```
dataset/raw/boundaries/
├── kyiv.geojson
├── warsaw.geojson
└── prague.geojson
```

**Время выполнения:** ~30 секунд

---

### 02_filter_osv5m.py

**Цель:** Отфильтровать метаданные OSV-5M, оставив только фото внутри городских границ.

**Что делает:**
1. Скачивает метаданные OSV-5M (train.csv ~1.5 GB)
2. Для каждого города:
   - Предварительная фильтрация по bounding box (быстро)
   - Точная проверка попадания координат в полигон
3. Сохраняет отфильтрованные метаданные

**Выход:**
```
dataset/raw/osv5m/filtered_cities.parquet
```

**Время выполнения:** ~5-10 минут (зависит от скорости интернета)

**Ожидаемый результат:**
```
kyiv:   ~800-1500 фото
warsaw: ~600-1200 фото
prague: ~500-1000 фото
```

---

### 03_download_osv5m_images.py

**Цель:** Скачать изображения для отфильтрованных записей.

**Что делает:**
1. Читает `filtered_cities.parquet`
2. Определяет нужные ZIP-шарды (OSV-5M разбит на архивы по 50К фото)
3. Скачивает только нужные шарды
4. Извлекает изображения в директории по городам
5. Генерирует итоговый манифест `manifest.csv`

**Выход:**
```
dataset/raw/osv5m/
├── images/
│   ├── kyiv/
│   │   ├── 123456789.jpg
│   │   └── ...
│   ├── warsaw/
│   │   └── ...
│   └── prague/
│       └── ...
└── manifest.csv
```

**Время выполнения:** ~30-60 минут (зависит от количества шардов и скорости интернета)

**Особенности:**
- **Идемпотентность:** можно прервать и перезапустить - не будет скачивать повторно
- **Организация:** изображения автоматически сортируются по папкам городов
- **Экономия места:** скачивает только нужные шарды, а не весь OSV-5M (5M фото)

---

### download_dataset.py (Master Script)

**Цель:** Запустить весь процесс одной командой.

**Опции:**
```bash
# Пропустить этап получения полигонов (если уже выполнен)
python scripts/download_dataset.py --skip-polygons

# Пропустить фильтрацию метаданных (если уже выполнена)
python scripts/download_dataset.py --skip-filter

# Только скачать изображения (все предыдущие этапы пропущены)
python scripts/download_dataset.py --skip-polygons --skip-filter
```

---

## 🔍 Проверка результатов

После завершения скачивания проверьте:

### 1. Количество изображений

```bash
# Linux/Mac
ls dataset/raw/osv5m/images/kyiv/ | wc -l
ls dataset/raw/osv5m/images/warsaw/ | wc -l
ls dataset/raw/osv5m/images/prague/ | wc -l

# Windows PowerShell
(Get-ChildItem dataset/raw/osv5m/images/kyiv/).Count
(Get-ChildItem dataset/raw/osv5m/images/warsaw/).Count
(Get-ChildItem dataset/raw/osv5m/images/prague/).Count
```

### 2. Манифест

```python
import pandas as pd

df = pd.read_csv("dataset/raw/osv5m/manifest.csv")
print(df["city"].value_counts())
print(f"\nTotal: {len(df)} images")
```

### 3. Визуализация распределения

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## ❓ Устранение проблем

### Ошибка: "HF_TOKEN not found"

**Решение:**
```bash
# Проверьте, что токен установлен
echo $HF_TOKEN  # Linux/Mac
echo %HF_TOKEN%  # Windows CMD

# Если пустой, установите:
export HF_TOKEN="hf_..."
```

### Ошибка: "ModuleNotFoundError: No module named 'osmnx'"

**Решение:**
```bash
pip install osmnx geopandas shapely
```

### Ошибка: "ConnectionError" или "TimeoutError"

**Причина:** Нестабильное интернет-соединение.

**Решение:**
- Перезапустите скрипт (он пропустит уже скачанные файлы)
- Используйте VPN если доступ к HuggingFace заблокирован

### Мало изображений после фильтрации

Если получилось меньше ожидаемого количества (например, <500 на город):

**Возможные причины:**
1. OSM полигоны слишком строгие (охватывают только центр)
2. В OSV-5M мало покрытия для этого города

**Решение:**
- Проверьте полигоны визуально: откройте `.geojson` на https://geojson.io/
- Можно расширить границы вручную в GeoJSON
- Добавьте данные из Mapillary (см. `code/download_data.py`)

---

## 📊 Ожидаемая структура данных после скачивания

```
dataset/
├── raw/
│   ├── boundaries/                 # Полигоны городов
│   │   ├── kyiv.geojson
│   │   ├── warsaw.geojson
│   │   └── prague.geojson
│   └── osv5m/
│       ├── images/                 # Изображения по городам
│       │   ├── kyiv/               # ~800-1500 фото
│       │   ├── warsaw/             # ~600-1200 фото
│       │   └── prague/             # ~500-1000 фото
│       ├── manifest.csv            # Итоговый манифест
│       ├── filtered_cities.parquet # Промежуточные метаданные
│       └── .cache/                 # Кэш HuggingFace (можно удалить)
└── manifests/                      # Создается следующим шагом
    ├── train.csv                   # 70% данных
    ├── val.csv                     # 15% данных
    └── test.csv                    # 15% данных
```

---

## 🔄 Следующие шаги после скачивания

1. **Создать train/val/test splits:**
   ```bash
   python scripts/generate_manifests.py \
       --input dataset/raw/osv5m/manifest.csv \
       --output-dir dataset/manifests
   ```

2. **Проверить данные:**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

3. **Начать обучение:**
   ```bash
   python code/train.py --config configs/baseline.yaml
   ```

---

## 💡 Оптимизация и дополнительные опции

### Если нужно больше данных

Можно дополнить данные из Mapillary:

```bash
python code/download_data.py mapillary \
    --bbox 50.2 30.2 50.6 30.8 \
    --api-key YOUR_MAPILLARY_TOKEN \
    --output data/mapillary \
    --max-images 2000
```

### Если нужно изменить города

Отредактируйте константу `CITIES` в скриптах:

```python
CITIES = {
    "budapest": "Budapest, Hungary",
    "vienna": "Vienna, Austria",
    # ...
}
```

---

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи выполнения скриптов
2. Убедитесь, что HF_TOKEN установлен корректно
3. См. основную документацию: `USAGE_GUIDE.md`
4. Проверьте интернет-соединение и свободное место на диске

---

**Удачи в скачивании данных! 🚀**
