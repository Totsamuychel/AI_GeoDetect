# AI_GeoDetect: Нейромережна модель геолокації за фото на основі вуличної фотографії України

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Опис проєкту

Дипломна робота присвячена розробці та порівнянню нейромережних архітектур для автоматичної геолокації вуличних фотографій з акцентом на територію України та суміжних країн Центральної і Східної Європи.

**Задача:** за вхідним зображенням вулиці передбачити географічні координати (широту і довготу) місця зйомки, а також класифікувати знімок за країною та містом.

**Три досліджувані архітектури:**

| Архітектура | Базова модель | Підхід |
|---|---|---|
| `BaselineCNN` | EfficientNet-B2 | Supervised classification + regression |
| `StreetCLIP` | CLIP (ViT-L/14) | Fine-tuning з текстовими підказками |
| `GeoCLIP` | CLIP + GPS encoder | Contrastive learning з GPS-галереєю |

**Цільові регіони:** Україна (Київ, Львів, Одеса, Харків, Дніпро) та сусідні країни — Польща, Чехія, Угорщина, Австрія, Румунія, Словаччина.

---

## Структура репозиторію

```
diploma/
├── README.md
├── environment.yml               # Conda environment
├── requirements.txt              # Pip dependencies
├── .gitignore
│
├── configs/
│   ├── config.yaml               # Основна конфігурація
│   ├── baseline_config.yaml      # EfficientNet-B2 baseline
│   └── geoclip_config.yaml       # GeoCLIP contrastive
│
├── data/
│   ├── dataset_manifest_example.csv   # Приклад маніфесту датасету
│   ├── images/                        # Вхідні зображення (не у git)
│   └── splits/                        # train/val/test CSV-файли
│
├── code/
│   ├── models/                   # Окремі архітектури моделей
│   ├── notebooks/                # Jupyter notebooks для EDA та експериментів
│   ├── augmentations.py          # Аугментації зображень
│   ├── dataset.py                # Класи Dataset та DataLoader
│   ├── download_data.py          # Завантаження зображень (Mapillary/OSM)
│   ├── evaluate.py               # Оцінка на тестовій вибірці
│   ├── inference.py              # Інференс для одного/кількох фото
│   ├── metrics.py                # Метрики (GCD, Accuracy@km тощо)
│   ├── models.py                 # Визначення моделей
│   ├── train.py                  # Головний скрипт тренування
│   ├── utils.py                  # Допоміжні функції
│   └── visualize.py              # Побудова карт та графіків
│
├── writing/                      # Текстова частина дипломної роботи
│
└── results/
    ├── checkpoints/              # Збережені ваги моделей (не у git)
    ├── logs/                     # WandB / MLflow / TensorBoard логи
    └── plots/                    # Збережені графіки та карти
```

---

## Встановлення залежностей

### Варіант 1: Conda (рекомендовано)

```bash
# Клонуємо репозиторій
git clone https://github.com/Totsamuychel/AI_GeoDetect.git
cd AI_GeoDetect

# Створюємо та активуємо conda-оточення
conda env create -f environment.yml
conda activate geo-photo

# Перевіряємо встановлення
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Варіант 2: pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# або
.venv\Scripts\activate          # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Підготовка даних

### 1. Завантаження зображень

Скрипт підтримує джерела Mapillary API та локальні директорії з фотографіями.

```bash
# Завантажити зображення для заданих міст через Mapillary API
python code/download_data.py \
    --source mapillary \
    --cities Kyiv Lviv Odesa Kharkiv Dnipro Warsaw Prague Budapest \
    --output data/images/ \
    --max-per-city 5000 \
    --mapillary-token YOUR_TOKEN

# Або імпортувати з локальної директорії
python code/download_data.py \
    --source local \
    --input /path/to/raw/photos \
    --output data/images/
```

---

## Тренування моделей

### BaselineCNN (EfficientNet-B2)

```bash
python code/train.py \
    --config configs/baseline_config.yaml \
    --model baseline \
    --experiment baseline_efficientnet_b2
```

### StreetCLIP fine-tuning

```bash
python code/train.py \
    --config configs/config.yaml \
    --model streetclip \
    --experiment streetclip_finetune_01 \
    --pretrained openai/clip-vit-large-patch14
```

### GeoCLIP (contrastive з GPS-галереєю)

```bash
python code/train.py \
    --config configs/geoclip_config.yaml \
    --model geoclip \
    --experiment geoclip_contrast_01 \
    --gallery-size 1000
```

---

## Оцінка результатів

```bash
# Оцінити одну модель
python code/evaluate.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/streetclip_best.pth \
    --model streetclip \
    --output results/metrics/streetclip_test_metrics.json
```

---

## Інференс

```bash
python code/inference.py \
    --image path/to/photo.jpg \
    --model streetclip \
    --checkpoint results/checkpoints/streetclip_best.pth \
    --config configs/config.yaml
```

---

## Ліцензія

Цей проєкт розповсюджується під ліцензією **MIT**. Дивіться файл [LICENSE](LICENSE) для деталей.

---

## Цитування

```bibtex
@mastersthesis{diploma2026geoloc,
  title     = {Нейромережна модель геолокації за фото на основі вуличної фотографії України},
  author    = {Totsamuychel},
  year      = {2026},
}
```
