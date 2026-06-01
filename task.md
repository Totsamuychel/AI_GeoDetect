# 📋 План: Скрипт сбора датасета городских фото через Google Street View Static API

## 🎯 Цель
Написать Python-скрипт `scripts/collect_streetview.py` для автоматического сбора
городских фото Києва, Варшави, Праги та Будапешта через Google Street View Static API.
Зберегти метадані сумісні з форматом `data/dataset_manifest_example.csv` проекту.

---

## ⚠️ Жёсткие ограничения (обязательно соблюдать)

1. **Никогда не скачивать фото без предварительной проверки через Metadata API**
   - Metadata API — бесплатный, не тратит кредиты
   - Только после статуса `"status": "OK"` делать реальный запрос фото
   
2. **Бюджетный лимит** — добавить аргумент `--max-budget-usd` (default: 1.90)
   - Стоимость одного фото: $0.007
   - При 1.90$ максимум = 271 фото (safety margin для тестов)
   - Для полного сбора пользователь явно указывает `--max-budget-usd 19.0` и т.д.
   - Счётчик потраченного бюджета инкрементировать ПОСЛЕ каждого успешного скачивания
   - При достижении лимита — СТОП, вывести сводку и выйти

3. **Счётчик запросов** — логировать каждые 100 запросов: сколько потрачено, сколько скачано

---

## 🏙️ Приоритет городов и количество фото

| Приоритет | Город | Целевое кол-во фото | BBox |
|-----------|-------|---------------------|------|
| 1 (главный) | Київ | 10 000 | `[50.213, 30.239, 50.590, 30.825]` |
| 2 | Варшава | 5 000 | `[52.200, 20.950, 52.290, 21.080]` |
| 3 | Прага | 5 000 | `[50.040, 14.360, 50.110, 14.480]` |
| 4 | Будапешт | 5 000 | `[47.460, 19.010, 47.540, 19.100]` |

Київ качати першим повністю, потім Варшава, Прага, Будапешт.

---

## 🏘️ Фильтрация: только городская местность (ВАЖНО)

### Метод 1 — OSM Overpass API (основной)
Перед генерацией сетки координат получить из OpenStreetMap полигоны
**только городской застройки** через Overpass API (бесплатно):

```python
# Запрашивать теги которые означают город:
OSM_URBAN_TAGS = [
    "landuse=residential",
    "landuse=commercial", 
    "landuse=industrial",
    "landuse=retail",
    "highway=residential",
    "highway=primary",
    "highway=secondary",
    "highway=tertiary",
]
# Исключить теги:
OSM_EXCLUDE_TAGS = [
    "landuse=forest",
    "landuse=meadow", 
    "landuse=farmland",
    "landuse=grass",
    "natural=wood",
    "natural=water",
    "leisure=park",  # большие парки — исключить
]
```

Функция `is_urban_point(lat, lon) -> bool` должна проверять попадает ли точка
в городской полигон из OSM. Кешировать полигоны локально в `data/osm_cache/`.

### Метод 2 — Street View Metadata дополнительная проверка
В ответе Metadata API есть поле `"description"` и `"date"` — если панорама
старше 2018 года, пропускать (старые фото низкого качества).

---

## 📐 Генерация сетки координат

```python
def generate_urban_grid(bbox, step_meters=120):
    """
    Генерирует сетку точек с шагом step_meters.
    Фильтрует точки через is_urban_point().
    Возвращает только городские точки.
    """
```

Шаг сетки: **120 метров** — оптимально для городской застройки.

---

## 📸 Параметры скачивания фото

Для каждой городской точки качать **4 направления**:
- `heading`: 0, 90, 180, 270 градусов
- `size`: `640x640`
- `fov`: 90
- `pitch`: 0
- `return_error_code`: true

URL запроса:
https://maps.googleapis.com/maps/api/streetview?
size=640x640&location={lat},{lon}&heading={heading}
&fov=90&pitch=0&key={API_KEY}&return_error_code=true

text

---

## 💾 Структура сохранения файлов
data/
├── images/
│ ├── kyiv/
│ │ ├── 50.45123_30.52341_h0.jpg
│ │ ├── 50.45123_30.52341_h90.jpg
│ │ └── ...
│ ├── warsaw/
│ ├── prague/
│ └── budapest/
├── osm_cache/
│ ├── kyiv_urban_polygons.pkl
│ ├── warsaw_urban_polygons.pkl
│ ├── prague_urban_polygons.pkl
│ └── budapest_urban_polygons.pkl
└── splits/
└── streetview_manifest.csv # совместимо с dataset_manifest_example.csv

text

Формат `streetview_manifest.csv`:
filename,city,lat,lon,heading,date_collected,source
data/images/kyiv/50.45123_30.52341_h0.jpg,kyiv,50.45123,30.52341,0,2026-05-31,streetview

text

---

## 🔄 Алгоритм работы скрипта (по шагам)
Парсинг аргументов CLI

Загрузка OSM полигонов для всех городов (или из кеша)

Для каждого города в порядке приоритета:
a. Сгенерировать сетку координат в bbox
b. Отфильтровать только urban точки через OSM
c. Перемешать точки (random.shuffle) для равномерного покрытия
d. Для каждой точки:
i. GET Metadata API → проверить status == "OK" (бесплатно)
ii. Проверить дату панорамы >= 2018
iii.Проверить бюджетный лимит
iv. Скачать 4 фото (4 heading)
v. Записать в manifest CSV
vi. Обновить счётчик бюджета и прогресс

По завершению — вывести итоговую статистику

text

---

## 🖥️ CLI интерфейс

```bash
# Тестовый запуск (очень дешёвый, только Київ, 50 фото)
python scripts/collect_streetview.py \
  --api-key YOUR_KEY \
  --max-budget-usd 0.50 \
  --cities kyiv \
  --output data/images/

# Полный сбор всех трёх городов
python scripts/collect_streetview.py \
  --api-key YOUR_KEY \
  --max-budget-usd 19.0 \
  --cities kyiv warsaw prague budapest \
  --step-meters 120 \
  --output data/images/

# Аргументы:
# --api-key         Google API ключ (или env: STREETVIEW_API_KEY)
# --max-budget-usd  Максимальный бюджет в USD (default: 1.90)
# --cities          Список городов через пробел
# --step-meters     Шаг сетки в метрах (default: 120)
# --output          Папка для сохранения
# --resume          Продолжить прерванный сбор (пропускать уже скачанные)
# --dry-run         Только посчитать сколько точек без скачивания
```

---

## 📊 Логирование и прогресс

- Использовать `tqdm` для прогресс-бара
- Каждые 100 фото выводить: `[Київ] 450/8000 фото | Потрачено: $3.15 | Осталось бюджета: $15.85`
- Сохранять лог в `results/logs/streetview_collection.log`
- При прерывании (Ctrl+C) — сохранить прогресс и вывести статистику

---

## 📦 Зависимости (добавить в requirements.txt)
requests>=2.31.0 # уже есть
tqdm>=4.65.0 # уже есть
shapely>=2.0.0 # для работы с OSM полигонами
overpy>=0.7 # Overpass API клиент для OSM
pandas>=2.0.0 # уже есть

text

---

## ✅ Definition of Done

- [ ] Скрипт запускается командой из раздела CLI
- [ ] `--dry-run` корректно считает точки без скачивания
- [ ] `--resume` пропускает уже скачанные файлы
- [ ] Бюджетный лимит работает и скрипт останавливается при достижении
- [ ] Все скачанные фото — только городская застройка (нет лесов, полей)
- [ ] `streetview_manifest.csv` совместим с форматом проекта
- [ ] OSM кеш сохраняется локально и переиспользуется