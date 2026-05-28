# RunPod Training — Інструкція для Claude Code агента

## Контекст
Проект геолокації вуличних фотографій. Три архітектури (baseline/streetclip/geoclip), три класи (Warsaw/Prague/Budapest). Датасет вже завантажений в архіві. Потрібно встановити залежності, розпакувати датасет, запустити навчання всіх моделей і зберегти результати.

---

## Твоє завдання (виконай кроки по порядку)

### Крок 1 — Перевірка середовища
```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Очікується: torch ≥ 2.6, CUDA True, GPU = RTX 5090 (або аналог).
Якщо CUDA False — ЗУПИНИСЬ і повідом.

### Крок 2 — Розпакування файлів
Файли мають бути в `/workspace/`:
- `diploma_dataset.tar.gz` — датасет (~2.6 GB)
- `diploma_code.zip` — код і конфіги

```bash
cd /workspace
unzip diploma_code.zip -d diploma
cd diploma
tar -xzf /workspace/diploma_dataset.tar.gz
```

Перевір що розпакувалось:
```bash
ls dataset/manifests/
ls dataset/raw/mapillary/ | head
python -c "import pandas as pd; [print(s, len(pd.read_csv(f'dataset/manifests/{s}.csv'))) for s in ['train','val','test']]"
```
Очікується: train=5271, val=1223, test=1000.

### Крок 3 — Встановлення залежностей
```bash
pip install -r requirements.txt
```
PyTorch НЕ перевстановлювати — він вже є в образі RunPod.
Якщо pip спробує перевстановити torch — додай `--no-deps` або пропусти torch рядки.

Перевір ключові пакети:
```bash
python -c "import pandas, transformers, PIL, sklearn, yaml; print('OK')"
```

### Крок 4 — Попереднє завантаження HuggingFace моделей
(Щоб уникнути таймаутів під час навчання)
```bash
python -c "
from transformers import CLIPModel, CLIPProcessor
print('Downloading StreetCLIP...')
CLIPModel.from_pretrained('geolocal/StreetCLIP')
CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
print('Downloading OpenAI CLIP base...')
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('Done')
"
```

### Крок 5 — Smoke-test (1 епоха перед повним навчанням)
```bash
python code/train.py --config configs/smoke_test.yaml
```
Очікується: без помилок, val_top1 ≈ 0.3+ (рандом = 0.33 для 3 класів).
Якщо падає — ЗУПИНИСЬ, покажи traceback.

### Крок 6 — Повне навчання всіх моделей
```bash
mkdir -p logs
python code/train.py --config configs/baseline.yaml   2>&1 | tee logs/baseline.log
python code/train.py --config configs/streetclip.yaml 2>&1 | tee logs/streetclip.log
python code/train.py --config configs/geoclip.yaml    2>&1 | tee logs/geoclip.log
```

Конфіги налаштовані для RTX 5090 (batch_size 64-128, 30 епох).
Очікуваний час: baseline ~1h, streetclip ~2h, geoclip ~2.5h.

### Крок 7 — Збір результатів і звіт у Markdown

Після завершення навчання зібери результати з логів і чекпоінтів та запиши у `results/training_report.md`.

Структура звіту:

```markdown
# Training Results — Warsaw / Prague / Budapest

**Date:** <дата>
**GPU:** <назва GPU>
**PyTorch:** <версія>

## Dataset Summary
| Split | Budapest | Prague | Warsaw | Total |
|-------|----------|--------|--------|-------|
| train | ... | ... | ... | 5271 |
| val   | ... | ... | ... | 1223 |
| test  | ... | ... | ... | 1000 |

## Results

### Baseline (EfficientNet-B2)
- Epochs: stage1=10 + stage2=20
- Best val_loss: ...
- Best val_top1: ...
- Best val_macro_f1: ...
- Best val_balanced_acc: ...
- Training time: ...

### StreetCLIP (CLIP ViT-L/14)
- Epochs: stage1=8 + stage2=15
- Best val_loss: ...
- Best val_top1: ...
- Best val_macro_f1: ...
- Best val_balanced_acc: ...
- Training time: ...

### GeoCLIP (CLIP + GPS Encoder)
- Epochs: stage1=12 + stage2=18
- Best val_loss: ...
- Best val_top1: ...
- Best val_macro_f1: ...
- Best val_balanced_acc: ...
- Training time: ...

## Best Model
<яка архітектура показала найкращий val_top1>

## Checkpoints
- checkpoints/baseline/best_model.pth
- checkpoints/streetclip/best_model.pth
- checkpoints/geoclip/best_model.pth
```

Для отримання метрик з логів використай grep:
```bash
grep "Епоха" logs/baseline.log | tail -5
grep "val_top1" logs/baseline.log | sort -t= -k2 -rn | head -1
```

### Крок 8 — Завантаження результатів
Запакуй чекпоінти і логи для завантаження:
```bash
tar -czf results_diploma.tar.gz checkpoints/ logs/ results/training_report.md
ls -lh results_diploma.tar.gz
```

---

## Якщо щось пішло не так

**`CUDA out of memory`** → зменши batch_size у yaml (baseline: 32, streetclip: 32, geoclip: 64)

**`Non-finite loss`** → зменши lr вдвічі і перезапусти

**`FileNotFoundError: manifest`** → перевір що tar розпакувався правильно:
```bash
ls dataset/manifests/train.csv dataset/raw/mapillary/warsaw/images/ | head
```

**`transformers` version error** → перевір версію:
```bash
python -c "import transformers; print(transformers.__version__)"
```
Потрібно ≥ 5.0. Якщо менше: `pip install transformers>=5.0`

---

## Структура проекту (довідка)
```
diploma/
├── code/           — train.py, models.py, dataset.py, metrics.py
├── configs/        — baseline.yaml, streetclip.yaml, geoclip.yaml, smoke_test.yaml
├── dataset/
│   ├── manifests/  — train.csv (5271), val.csv (1223), test.csv (1000)
│   └── raw/mapillary/{warsaw,prague,budapest}/images/*.jpg
├── scripts/        — runpod_setup.sh, runpod_train_all.sh
├── checkpoints/    — з'явиться після навчання
└── logs/           — з'явиться після навчання
```
