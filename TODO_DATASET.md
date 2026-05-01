# 📋 TODO — Збірка датасету

> Платформа: **Локальна машина (win32)**
> Скрипт завантаження: `code/download_data.py`
> Гайд: `dataset_assembly_guide.md`

---

## 1. OSV-5M — основне джерело (обов'язково)

### 1.1 Завантажити та відфільтрувати по bbox міст
- [ ] Запустити завантаження OSV-5M для UA + сусідніх країн:
  ```bash
  python code/download_data.py osv5m \
      --countries UA PL CZ HU \
      --output dataset/raw/osv5m \
      --max-images 6000 \
      --quality 0.3 \
      --workers 8
  ```
  > ⚠️ `--max-images 6000` = ~6000 фото на країну. Загальний розмір ~3–6GB.
  > Парquet-метадані скачуються автоматично, потім тільки потрібні шарди.

### 1.2 Перевірити результат
- [ ] Створити маніфест та перевірити статистику:
  ```bash
  python code/download_data.py manifest \
      --image-dir dataset/raw/osv5m/images \
      --output dataset/raw/osv5m/manifest.csv \
      --stats
  ```

---

## 2. Mapillary — додаткове джерело (опціонально)

> Пропусти якщо OSV-5M дав ≥ 2000 фото на кожне місто

- [ ] Для міст з дефіцитом фото — докачати через Mapillary API:
  ```bash
  # Приклад для Одеси:
  python code/download_data.py mapillary \
      --bbox 46.35 30.60 46.55 30.82 \
      --output dataset/raw/mapillary/odesa \
      --max-images 3000 \
      --start-date 2019-01-01 \
      --end-date 2023-12-31 \
      --no-zensvi
  ```

---

## 3. Дедуплікація та Очистка

- [ ] Видалити дублікати між OSV-5M та Mapillary (якщо використовували обидва).
- [ ] Запустити фільтрацію розмитих та малих фото:
  ```bash
  # Скрипт 07_quality_filter.py з dataset_assembly_guide.md
  ```

---

## 4. H3 Стратифіковане розбиття 70/15/15

- [ ] Запустити H3-розбиття (без витоку між train/val/test):
  ```bash
  # Скрипт 08_h3_split.py з dataset_assembly_guide.md
  ```

---

## 5. Фінальна перевірка та тренування

- [ ] Перевірити фінальні маніфести в `dataset/manifests/`.
- [ ] Запустити baseline (EfficientNet-B2):
  ```bash
  python code/train.py \
      --config configs/baseline.yaml \
      --manifest dataset/manifests/train.csv \
      --image-root dataset/processed/images
  ```

---

*Останнє оновлення: 30.04.2026*
*Stack: Python 3.10 · PyTorch · OSV-5M · Mapillary · H3*
