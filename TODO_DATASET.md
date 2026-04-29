# 📋 TODO — Збірка датасету

> Платформа: **RunPod RTX 5090** · Сховище: **RunPod Volume 20GB**
> Скрипт завантаження: `code/download_data.py` (вже реалізовано)
> Гайд: `dataset_assembly_guide.md`

---

## 0. Підготовка середовища

- [ ] Створити RunPod Pod з шаблоном **PyTorch 2.7 + CUDA 12.8** (для RTX 5090 / sm_120)
- [ ] Підключити **Volume 20GB** до Pod'а (`/workspace`)
- [ ] Встановити залежності:
  ```bash
  pip install torch==2.7.* torchvision==0.22.* --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```
- [ ] Отримати **HuggingFace Token** → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- [ ] Отримати **Mapillary API Key** → [mapillary.com/dashboard/developers](https://www.mapillary.com/dashboard/developers) *(опціонально)*
- [ ] Експортувати ключі:
  ```bash
  export HF_TOKEN="hf_xxxxxxxxxxxxxxx"
  export MAPILLARY_ACCESS_TOKEN="MLY|xxxxxxxxxxxxxxx"   # опціонально
  ```
- [ ] Перевірити GPU:
  ```python
  import torch
  assert torch.cuda.is_available()
  print(torch.cuda.get_device_name(0))   # → NVIDIA GeForce RTX 5090
  ```

---

## 1. OSV-5M — основне джерело (обов'язково)

### 1.1 Завантажити та відфільтрувати по bbox міст
- [ ] Запустити завантаження OSV-5M для UA + сусідніх країн:
  ```bash
  python code/download_data.py osv5m \
      --countries UA PL CZ HU \
      --output /workspace/dataset/raw/osv5m \
      --max-images 6000 \
      --quality 0.3 \
      --workers 8
  ```
  > ⚠️ `--max-images 6000` = ~6000 фото на країну. Загальний розмір ~3–6GB.
  > Парquet-метадані скачуються автоматично, потім тільки потрібні шарди.

### 1.2 Перевірити результат
- [ ] Переконатись що кожне місто має ≥ 2000 фото:
  ```bash
  python code/download_data.py manifest \
      --image-dir /workspace/dataset/raw/osv5m \
      --output /workspace/dataset/raw/osv5m/manifest.csv \
      --stats
  ```
  Очікуваний результат:
  ```
  kyiv:     ~3000-5000
  warsaw:   ~2000-4000
  prague:   ~2000-3500
  ...
  Imbalance Ratio < 3.0  ✅
  ```

---

## 2. Mapillary — додаткове джерело (опціонально)

> Пропусти якщо OSV-5M дав ≥ 2000 фото на кожне місто

- [ ] Для міст з дефіцитом фото — докачати через Mapillary API:
  ```bash
  # Приклад для Одеси (якщо мало фото з OSV-5M):
  python code/download_data.py mapillary \
      --bbox 46.35 30.60 46.55 30.82 \
      --output /workspace/dataset/raw/mapillary/odesa \
      --max-images 3000 \
      --start-date 2019-01-01 \
      --end-date 2023-12-31 \
      --no-zensvi
  ```
  > `--no-zensvi` — використовує пряме Mapillary Graph API (без проблем з zensvi залежностями)

- [ ] Повторити для кожного міста з нестачею даних:
  - [ ] Київ (bbox: 50.30 30.20 50.60 30.85)
  - [ ] Львів (bbox: 49.77 23.90 49.90 24.15)
  - [ ] Одеса (bbox: 46.35 30.60 46.55 30.82)
  - [ ] Харків (bbox: 49.90 36.10 50.10 36.42)
  - [ ] Дніпро (bbox: 48.38 34.90 48.55 35.20)
  - [ ] Варшава (bbox: 52.10 20.80 52.37 21.20)
  - [ ] Прага (bbox: 49.94 14.22 50.18 14.71)
  - [ ] Будапешт (bbox: 47.35 18.90 47.61 19.20)

---

## 3. Дедуплікація (якщо використовували обидва джерела)

- [ ] Видалити Mapillary фото що дублюються з OSV-5M (поріг 50м):
  ```bash
  python dataset_assembly_guide.md   # скрипт 06_deduplication.py
  ```
  > Пропусти якщо використовував тільки OSV-5M

---

## 4. Контроль якості

- [ ] Запустити фільтрацію розмитих та малих фото:
  ```bash
  # Скрипт 07_quality_filter.py з dataset_assembly_guide.md
  # Laplacian variance < 100 → видалити
  # Розмір < 500px → видалити
  python -c "
  import cv2, pandas as pd, os
  from tqdm import tqdm

  df = pd.read_csv('/workspace/dataset/raw/osv5m/manifest.csv')
  valid = []
  for _, row in tqdm(df.iterrows(), total=len(df)):
      path = row['filepath']
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      if img is None: valid.append(False); continue
      h, w = img.shape
      var = cv2.Laplacian(img, cv2.CV_64F).var()
      valid.append(var >= 100 and min(h, w) >= 500)
  df_clean = df[valid]
  print(f'Залишилось: {len(df_clean)} / {len(df)}')
  df_clean.to_csv('/workspace/dataset/cleaned_metadata.csv', index=False)
  "
  ```

- [ ] (Опціонально) Розмити обличчя якщо плануєш публікувати датасет:
  ```bash
  pip install deface
  python -m deface /workspace/dataset/processed/images/ \
      --output /workspace/dataset/processed/images/ \
      --thresh 0.3
  ```

---

## 5. H3 Стратифіковане розбиття 70/15/15

- [ ] Запустити H3-розбиття (без витоку між train/val/test):
  ```bash
  # Скрипт 08_h3_split.py з dataset_assembly_guide.md
  # Результат: dataset/manifests/train.csv, val.csv, test.csv
  ```
  > Використовує `h3.latlng_to_cell()` — вже виправлено в `dataset.py`

- [ ] Перевірити фінальні маніфести:
  ```bash
  python code/download_data.py manifest \
      --image-dir /workspace/dataset/processed/images \
      --output /workspace/dataset/manifests/full.csv \
      --stats
  ```

---

## 6. Фінальна перевірка перед тренуванням

- [ ] Запустити валідацію (скрипт 09 з гайду):
  ```bash
  python -c "
  import pandas as pd
  for split in ['train', 'val', 'test']:
      df = pd.read_csv(f'/workspace/dataset/manifests/{split}.csv')
      ir = df['city'].value_counts().max() / df['city'].value_counts().min()
      print(f'{split}: {len(df):,} фото | IR={ir:.2f}')
  "
  ```
  Очікуваний результат:
  ```
  train: ~24 000 фото | IR < 3.0  ✅
  val:   ~5 000 фото  | IR < 3.0  ✅
  test:  ~5 000 фото  | IR < 3.0  ✅
  ```

- [ ] Переконатись що шляхи в CSV вказують на реальні файли:
  ```python
  import pandas as pd
  from pathlib import Path
  df = pd.read_csv('/workspace/dataset/manifests/train.csv')
  missing = df['filepath'].apply(lambda p: not Path(p).exists()).sum()
  print(f'Відсутніх файлів: {missing}')  # → 0
  ```

- [ ] Запустити тестовий forward pass через `GeoDataset`:
  ```python
  from code.dataset import GeoDataset
  from code.augmentations import get_val_transforms
  ds = GeoDataset('/workspace/dataset/manifests/train.csv', get_val_transforms(224))
  img, label, coords = ds[0]
  print(img.shape, label, coords)   # torch.Size([3,224,224]), int, tensor([lat, lon])
  ```

---

## 7. Тренування

- [ ] Запустити baseline (EfficientNet-B2):
  ```bash
  python code/train.py \
      --config configs/baseline.yaml \
      --manifest /workspace/dataset/manifests/train.csv \
      --image-root /workspace/dataset/processed/images
  ```

- [ ] Запустити StreetCLIP:
  ```bash
  python code/train.py --config configs/streetclip.yaml
  ```

- [ ] Запустити GeoCLIP:
  ```bash
  python code/train.py --config configs/geoclip.yaml
  ```

---

## 📊 Очікуваний розмір датасету

| Метрика | Значення |
|---|---|
| Всього фото | ~25 000–35 000 |
| Розмір на диску | ~3–6 GB (512px) |
| Міст | 8 (UA×5 + PL+CZ+HU) |
| Train / Val / Test | 70% / 15% / 15% |
| Imbalance Ratio | < 3.0 |

---

*Останнє оновлення: автоматично*
*Stack: Python 3.10 · PyTorch 2.7 · CUDA 12.8 · RunPod RTX 5090 · OSV-5M · H3*
