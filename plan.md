# План: подготовка проекта к обучению на RunPod (RTX 5090)

## Context

Дипломный проект — классификация города по уличному фото (3 класса). Код (`code/`,
3 архитектуры: `baseline` EfficientNet-B2, `streetclip`, `geoclip`) написан и работает
концептуально, но **не запустится на арендованной RTX 5090** и содержит баги, из-за
которых метрики будут невалидны для защиты. Обучение пойдёт на одной RTX 5090 на
RunPod (Linux, 32 GB VRAM).

Зафиксированные решения:
- **Классы:** Warsaw / Prague / Budapest — данные уже собраны (`dataset/manifests/*.csv`,
  ~7.5k фото Mapillary, идеальный баланс ~33%/город, Киева нет — это ОК). Kyiv не добавляем.
- **Объём:** только исправление багов + улучшение пайплайна обучения. Расширение
  датасета (пустые OSV-5M / Global Streetscapes) — вне рамок.
- **Сплит:** корректный, без утечки соседних кадров, со стратификацией по городу.

Цель: после выполнения можно арендовать под, склонировать репо, запустить 3
обучения и получить честные, защищаемые метрики.

---

## Главные проблемы (проверено чтением кода)

1. **BLOCKER — окружение несовместимо с Blackwell.** `environment.yml:13-17` и
   `requirements.txt:5-7` фиксируют `pytorch=2.1.* / cu118`. RTX 5090 = Blackwell
   (sm_120), требует CUDA 12.8 и PyTorch с wheels `cu128` (≥2.7). На 5090 любой CUDA-kernel
   с torch 2.1+cu118 упадёт сразу.
2. **BLOCKER — невалидный сплит данных.** Все конфиги указывают
   `manifest_path: dataset/raw/mapillary/manifest.csv` + `split_method: h3`, поэтому
   `train.py` **игнорирует** готовые `dataset/manifests/{train,val,test}.csv` и делает
   собственный H3-сплит `res=4` в `dataset.py:_split_by_h3` (`dataset.py:35-100`,
   вызывается из `create_dataloaders` `dataset.py:530-536`). H3 res-4 ячейка ≈ 1770 км² —
   каждый из 3 близких городов укладывается в 1–4 ячейки; шифл ячеек и срез 70/15/15
   (`dataset.py:81-94`) с лёгкостью отправляет целый город целиком в один сплит →
   в val/test может не быть класса, `num_classes` (из полного df) ≠ классам в val/test,
   метрики и per-city разбивка бессмысленны. Плюс кадры Mapillary идут
   последовательностями (соседние почти-дубли) → при любом «случайном» сплите утечка
   train↔test и завышенная точность.
3. **Неверная нормализация для CLIP.** `code/augmentations.py:24-25` задаёт ImageNet
   mean/std для **всех** моделей; `train.py:540-541` зовёт `get_train/val_transforms`
   с дефолтом. StreetCLIP/GeoCLIP (OpenAI CLIP) ожидают CLIP mean/std
   (`(0.4815,0.4578,0.4082)/(0.2686,0.2613,0.2758)`). HF `CLIPModel` не нормализует
   сырой тензор сам — фичи деградируют.

Менее критичное, но реальное (войдёт в фазу 2): отсутствие LR-warmup перед
`CosineAnnealingLR` (`train.py:604,696`), маленький `batch_size:32` для InfoNCE GeoCLIP
(`geoclip.yaml:30`) при 32 GB VRAM, `num_workers:4` занижен, нет bf16 (Blackwell это
любит), нет guard на NaN-loss, `validate()` для geoclip идёт другим forward-путём
(`train.py:491-493`), а `evaluate.py` считает центры городов по всему датасету включая
test (утечка в метрику расстояния).

Сознательно НЕ считаем багами (отвергнуто после проверки): `label_smoothing` + class
weights (валидно в PyTorch, данные сбалансированы), сброс EarlyStopping между стадиями
(намеренно), `img_size:260` для baseline (нативный для EfficientNet-B2, transforms
корректно его обрабатывают).

---

## Фаза 0 — Окружение под RunPod/5090 (BLOCKER, делать первым)

- **Стратегия:** не пинить `torch` в репозитории. Использовать на RunPod готовый
  шаблон с PyTorch ≥2.7 + CUDA 12.8 (cu128), остальное доставить через pip.
- `requirements.txt`: убрать строку-инструкцию про `cu118` (стр. 5-7), заменить
  комментарием «torch/torchvision ставит базовый образ RunPod (PyTorch 2.7+, CUDA 12.8);
  не устанавливать здесь». Оставить только не-torch зависимости.
- `environment.yml`: пометить как legacy/локальный (CPU/Windows), либо обновить
  `pytorch`/`pytorch-cuda` до 2.7+/12.8; не использовать его на поде.
- Добавить `scripts/runpod_setup.sh`: проверка `nvidia-smi`, версии torch и
  `torch.cuda.get_device_capability()` (ожидаем sm_120 / `(12, 0)`), `pip install -r
  requirements.txt`, экспорт `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:16:8`,
  затем `pytest -q tests/`.
- Проверить, что `code/utils.py:get_device()` корректно возвращает `cuda` и логирует
  имя GPU и `torch.version.cuda`.

## Фаза 1 — Критические баги корректности

### 1.1 Корректный сплит без утечки + train.py реально его использует
- Прочитать `code/models.py` (подтвердить API `unfreeze_last_n_blocks`,
  `geoclip.forward(images)` без coords, что CLIP-backbone не нормализует сам).
- Переписать `scripts/06_build_mapillary_splits.py:split_dataset()`: гео-блочный
  сплит со стратификацией по городу — если есть `sequence_id`, группировать по нему,
  иначе H3 res=7 (~5 км²); внутри каждого города перемешать группы (seed=42) и
  раздать 70/15/15 по *группам*, не по строкам; перегенерировать
  `dataset/manifests/{train,val,test}.csv`, вывести распределение и контроль
  пересечений (0 общих `image_id`).
- Добавить `split_method: "prebuilt"` в `code/dataset.py:create_dataloaders`:
  грузить `train.csv` и сиблингов `val.csv`/`test.csv` напрямую, пропуская
  `get_split_indices`. Обновить 3 конфига: `manifest_path: dataset/manifests/train.csv`,
  `split_method: "prebuilt"`.

### 1.2 Правильная нормализация для CLIP-моделей
- В `code/augmentations.py` добавить `CLIP_MEAN/CLIP_STD` и хелпер
  `get_norm_for(architecture)`.
- В `code/train.py` выбирать mean/std по `config.architecture`
  (`baseline`→ImageNet, `streetclip`/`geoclip`→CLIP).
- Те же константы в `code/evaluate.py` и `code/inference.py`.

## Фаза 2 — Улучшение обучения

- **bf16:** `autocast(dtype=torch.bfloat16)`, `GradScaler` только для fp16; снять
  хардкод `device_type="cuda"`.
- **Batch/workers:** `num_workers: 8`; `batch_size` baseline/streetclip 64–96,
  geoclip ≥128 (InfoNCE).
- **LR warmup:** `LinearLR` перед `CosineAnnealingLR` через `SequentialLR`.
- **GeoCLIP:** в `validate()` передавать `coords` и считать тот же составной loss;
  `contrastive_loss_weight: float = 0.1` в `TrainConfig`.
- **Надёжность:** guard `torch.isfinite(loss)`; лог `max_memory_allocated`.
- **evaluate.py:** центры городов — только по train-сплиту.

## Фаза 3 — Прогон и метрики (на поде)

- 3 обучения: `python code/train.py --config configs/{baseline,streetclip,geoclip}.yaml`.
- `code/evaluate.py` на `dataset/manifests/test.csv` для лучшего чекпоинта каждой
  модели; таблица сравнения (Top-1/5, per-city, Haversine, GeoScore) в
  `results/comparison.md`.

---

## Верификация

1. **Окружение:** `runpod_setup.sh` → `get_device_capability()` == `(12, 0)`,
   `pytest -q tests/` зелёный.
2. **Сплит:** `set(train.image_id) ∩ test == ∅`, все 3 города в каждом сплите,
   доли ~70/15/15, ни одна группа не делится между сплитами.
3. **Нормализация:** для CLIP в трансформациях mean≈0.481; smoke-train — loss падает.
4. **Обучение:** короткий прогон каждой архитектуры без ошибок.
5. **Метрики:** `evaluate.py` отрабатывает на test.csv, таблица сгенерирована.

## Ключевые файлы

- Фаза 0: `requirements.txt`, `environment.yml`, `scripts/runpod_setup.sh` (нов.)
- Фаза 1.1: `scripts/06_build_mapillary_splits.py`, `code/dataset.py`,
  `configs/{baseline,streetclip,geoclip}.yaml`
- Фаза 1.2 / 2: `code/augmentations.py`, `code/train.py`, `code/evaluate.py`,
  `code/inference.py`
- Чтение: `code/models.py`

## Вне рамок (осознанно)

Расширение датасета (докачка Mapillary, пустые OSV-5M/Global Streetscapes),
мульти-GPU/DDP, refactor скачивающих скриптов, хардкод города в
`download_kartaview.py` (не используется в текущем пайплайне).
