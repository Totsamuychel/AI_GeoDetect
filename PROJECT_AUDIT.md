# AI GeoDetect — Аудит проєкту

> Повний аналіз усіх Python-файлів у `code/` + конфіги: `models.py`, `train.py`,
> `dataset.py`, `augmentations.py`, `metrics.py`, `inference.py`, `evaluate.py`,
> `tui_trainer.py`, `utils.py`, `configs/*.yaml`, `requirements.txt`.
> Дата: 2026-05-19
>
> **Легенда статусу:** ✅ **[Виправлено]** — пофікшено в цьому проході ·
> 💡 **[Рекомендація]** — задокументовано, але НЕ застосовано автоматично
> (змінює динаміку навчання / продуктивність, потребує окремих експериментів,
> щоб не зламати відтворюваність дипломних результатів).

---

## Зведена таблиця

| # | Файл | Місце | Severity | Статус | Короткий опис |
|---|------|-------|----------|--------|---------------|
| C-01 | `inference.py` | `GeoLocator.__init__` | 🔴 CRITICAL | ✅ | `UKRAINE_CITY_CENTERS` не визначено — `NameError`, весь CLI інференсу мертвий |
| C-02 | `tui_trainer.py` | `evaluate_model()` | 🔴 CRITICAL | ✅ | передає `--architecture` якого немає в `evaluate.py` → меню [7] падає |
| M-01 | `models.py` | `contrastive_loss` | 🟠 MEDIUM | ✅ | GeoCLIP temperature стартує поза clamp → градієнт=0, не вчиться |
| M-04 | `train.py` | `validate()` | 🟠 MEDIUM | ✅ | InfoNCE на хвостовому батчі N=1 вироджений, зміщує val_loss |
| T-01 | `train.py`/`tui_trainer.py` | callbacks/ETA | 🟡 LOW | ✅ | ETA вибухає на межі Стадія 1→2 (нумерація епох не глобальна) |
| T-02 | `tui_trainer.py` | `run_all_training()` | 🟡 LOW | ✅ | впала архітектура пропускається мовчки |
| T-03 | `tui_trainer.py` | `_show_all_status()` | 🟡 LOW | ✅ | `console.clear()` → блимання + втрата логів |
| D-01 | `train.py` | `CheckpointManager.save` | 🔵 MINOR | ✅ | анотація `_LRScheduler` (застарілий аліас) |
| D-02 | `train.py` | `_init_wandb` | 🔵 MINOR | ✅ | `wandb.init(reinit=True)` — bool reinit застарів у wandb ≥0.16 |
| D-03 | `utils.py` | `reverse_geocode` | 🔵 MINOR | ✅ | Nominatim user_agent з фейковим `research@example.com` |
| P-04 | `train.py` | `train()` | 🔵 MINOR | ✅ | `num_workers` не обмежений CPU (Windows spawn) |
| M-02 | `models.py` | GPS-енкодер | 🟠 MEDIUM | 💡 | слабкий сигнал GPS для 3-країнного датасету |
| M-03 | `dataset.py`/`train.py` | class weights | 🟠 MEDIUM | 💡 | inverse-freq (sum=1) + label_smoothing — подвійна регуляризація |
| P-01 | `dataset.py` | `create_dataloaders` | 🟡 LOW | 💡 | CSV-маніфест читається 3–4× (non-prebuilt) |
| P-02 | `dataset.py` | `_split_by_h3` | 🟡 LOW | 💡 | рядковий `df.apply` — повільно на великих даних |
| P-03 | `utils.py` | `seed_everything` | 🟡 LOW | 💡 | примусовий cudnn-детермінізм уповільнює навчання 20–40% |
| D-04 | `models.py` | `build_gallery`/`retrieve_gps` | 🔵 MINOR | 💡 | мертвий код (retrieval-режим не задіяний у пайплайні) |
| D-05 | `evaluate.py` | distance/GeoScore | 🔵 NOTE | 💡 | метрика відстані = помилка класифікації на центроїдах (методологія) |

---

## ✅ Виправлено в цьому проході

### C-01 🔴 — `inference.py`: `UKRAINE_CITY_CENTERS` не визначено

**Було:** `code/inference.py:66` — `self.city_coords = city_coords or
UKRAINE_CITY_CENTERS.copy()`, але `UKRAINE_CITY_CENTERS` ніде не оголошено.
Будь-який `GeoLocator(checkpoint)` за замовчуванням → `NameError`. Додатково
семантично хибно: реальний датасет — PL/CZ/HU (Варшава/Прага/Будапешт), не
Україна.

**Стало:**
- Додано модульну константу `DEFAULT_CENTER = (50.0, 17.5)` (центр PL/CZ/HU).
- Додано хелпер `_city_centers_from_manifest()` — рахує середні (lat, lon)
  по містах із CSV (аналогічно `evaluate._city_centers_from_df`, без витоку
  якщо подати TRAIN-маніфест).
- `GeoLocator.__init__` отримав `manifest_path: Optional[...]`. Пріоритет:
  `city_coords` → `manifest_path` → `{}` (+ попередження).
- Усі фолбеки `(49.0, 32.0)` → `DEFAULT_CENTER`.
- CLI: додано `--manifest`, прокинуто в `main()`.

**Перевірка:** `python code/inference.py --help` стартує без `NameError`,
`--manifest` присутній; `grep UKRAINE_CITY_CENTERS` — порожньо.

### C-02 🔴 — `tui_trainer.py`: невідомий аргумент `--architecture`

**Було:** `evaluate_model()` будував `cmd = [... "--output", output,
"--architecture", arch]`, але argparser `evaluate.py` не має такого прапорця →
`argparse` завершується з ненульовим кодом, меню **[7] Evaluate** падало.
Архітектура й так читається з чекпоінту всередині `evaluate.py`.

**Стало:** `--architecture arch` прибрано з `cmd`. `arch` досі читається з
чекпоінту, але лише для інформаційного `console.print`.

### M-01 🟠 — `models.py`: GeoCLIP temperature стартує замороженою

**Було:** `models.py` init `log_temperature = log(1/0.07) ≈ 2.659`;
`contrastive_loss` робив `exp(log_temperature).clamp(min=0.01, max=10.0)`.
`exp(2.659) ≈ 14.29` > `max=10.0` → значення затиснуте на межі, градієнт через
clamp = 0, параметр ніколи не оновлюється; ефективний масштаб хибний з кроку 0.

**Стало:** `clamp(min=0.01, max=100.0)` — відповідає конвенції CLIP logit-scale;
init 14.29 тепер у межах діапазону й параметр навчається.

### M-04 🟠 — `train.py`: вироджений InfoNCE на батчі N=1

**Було:** `validate()` для geoclip рахував контрастивний член навіть коли
останній val-батч має розмір 1 (`drop_last=False`). При N=1 `similarity` —
(1,1), `cross_entropy` ≈ 0 → занижений `val_loss`, спотворює early stopping.

**Стало:** контрастивний член додається лише при `images.size(0) >= 2`,
інакше `loss = criterion(logits, labels)`.

### T-01 🟡 — ETA вибухає на межі Стадія 1 → Стадія 2

**Було:** `train.py` передавав у callback `_cb_total_epochs` = розмір
поточної стадії, а `_cb_epoch` у Стадії 2 починався з 1.
`tui_trainer._apply_info` перезаписував `state["total_epochs"]`, тож на старті
Стадії 2 `completed≈0` при великому `elapsed` → `spe = elapsed/completed` →
астрономічний ETA.

**Стало (тільки `train.py`):** скрізь передається глобальне
`total_epochs = stage1+stage2`; у Стадії 2 `_cb_epoch` і `epoch` payload =
`stage1_epochs + epoch`. Нумерація епох тепер монотонна, ETA коректний через
межу стадій. Логіка TUI не змінювалась (вона коректна за коректних входів).

### T-02 🟡 — впала архітектура пропускається мовчки

**Було:** `run_all_training()` при відсутності валідних епох ставив
`statuses[arch]="stopped"` і мовчки йшов далі.

**Стало:** додано червоний банер «⚠ Архітектуру '{arch}' не натреновано…»
+ пауза `Enter → продовжити`, щоб користувач побачив це до наступної арх.

### T-03 🟡 — `_show_all_status()` блимання

**Було:** `console.clear()` на кожен виклик стирав попередні логи навчання.

**Стало:** `console.clear()` прибрано (Panel сам відділяє блок); виклик
безпечний — він і так лише ~3 рази за прогін.

### D-01 🔵 — застаріла анотація `_LRScheduler`

`CheckpointManager.save` анотація `torch.optim.lr_scheduler._LRScheduler` →
`torch.optim.lr_scheduler.LRScheduler` (рядкова анотація через
`from __future__ import annotations`, тож рантайм не зачеплено — лише чистота).

### D-02 🔵 — `wandb.init(reinit=True)`

`reinit=True` (bool) застарів у wandb ≥0.16 → `reinit="finish_previous"`.
Латентно (`use_wandb: false` за замовчуванням), у наявному try/except.

### D-03 🔵 — Nominatim user_agent

Фейковий контакт `research@example.com` (порушує ToS Nominatim, ризик блоку) →
нейтральний `"ai_geodetect_thesis/1.0"`.

### P-04 🔵 — `num_workers` без обмеження CPU

Додано `n_workers = min(config.num_workers, os.cpu_count() or 1)` перед
`create_dataloaders` (Windows spawn: завеликий num_workers повільно стартує й
перевантажує пам'ять), з лог-повідомленням про корекцію.

---

## 💡 Рекомендації (НЕ застосовано — потребують експериментів)

> Свідомо не чіпав автоматично: зміна цих місць тихо змінює динаміку навчання
> або продуктивність і вимагає повторних прогонів/абляцій. Для дипломної
> відтворюваності рішення лишається за автором.

### M-02 🟠 — слабкий сигнал GPS-енкодера для 3-країнного датасету

`GeoCLIPModel.normalize_coords` ділить `lat/90, lon/180`. Для PL/CZ/HU
нормалізований діапазон крихітний (lon ≈ 0.07–0.13). З `B = randn(2,256)*0.1`
Fourier-фічі майже не варіюються між зразками → GPS-embedding погано
розділяє регіони. **Ідея:** стандартизувати координати за bbox / mean-std
датасету (або підняти `sigma` RFF). Потребує перетренування + абляції.

### M-03 🟠 — подвійна регуляризація class weights + label smoothing

`get_class_weights` повертає inverse-freq, нормалізовані до sum=1; `train.py`
додатково `label_smoothing=0.1`. Для ~3 класів це може надмірно згладжувати
сигнал. **Ідея:** нормувати `weights / weights.mean()` та/або прибрати label
smoothing — як окремий tuning-експеримент із порівнянням метрик.

### P-01 🟡 — маніфест читається 3–4× (non-prebuilt)

`create_dataloaders` (non-prebuilt) будує `GeoDataset` для full+train+val+test,
кожен робить `pd.read_csv` по тому самому файлу (`dataset.py:591-647`).
**Ідея:** прочитати раз і передавати DataFrame / кешувати за шляхом. (Для
`split_method: prebuilt` — як зараз у конфігах — проблема не виникає.)

### P-02 🟡 — `_split_by_h3` рядковий `df.apply`

`dataset.py:77-79` — O(N) на рівні Python. **Ідея:** векторизувати через
list-comprehension по `zip(lat, lon)` або numpy. Важливо лише для великих
датасетів і лише при `split_method: h3` (не `prebuilt`).

### P-03 🟡 — примусовий cudnn-детермінізм

`seed_everything` ставить `cudnn.deterministic=True, benchmark=False`
(`utils.py:50-51`) — коректно для відтворюваності (узгоджено з вимогою
дипломної строгості), але сповільнює CNN на 20–40%. **Ідея:** винести
детермінізм у прапорець конфіга, лишивши відтворюваний дефолт; фінальні
заміри швидкості робити з вимкненим детермінізмом.

### D-04 🔵 — мертвий код retrieval у GeoCLIP

`build_gallery`/`retrieve_gps` (`models.py`) не викликаються у
train/eval/inference (пайплайн — класифікаційний). Або задокументувати як
невикористане, або додати окрему retrieval-оцінку для диплома.

### D-05 🔵 — методологічна примітка для тексту диплома

`evaluate.py` як «передбачену» точку бере **центроїд передбаченого міста**,
тож Mean Distance / GeoScore — це помилка *класифікації*, спроєктована на
центроїди (обмежена відстанями між містами), а не регресія координат. Це
**за дизайном** для класифікаційної моделі, не баг — але в тексті диплома
формулювання метрик має це чітко відображати.

---

## Пріоритет виправлень (історично)

### Критично (🔴) — зроблено
| # | Файл | Дія |
|---|------|-----|
| C-01 | `inference.py` | визначити центри міст, прибрати `UKRAINE_CITY_CENTERS` |
| C-02 | `tui_trainer.py` | прибрати `--architecture` з виклику evaluate |

### Важливо (🟠)
| # | Дія | Статус |
|---|-----|--------|
| M-01 | clamp temperature max=100 | ✅ |
| M-04 | guard InfoNCE N≥2 | ✅ |
| M-02 | нормалізація координат GPS | 💡 експеримент |
| M-03 | class weights / label smoothing | 💡 експеримент |

### Решта (🟡/🔵)
T-01, T-02, T-03, D-01, D-02, D-03, P-04 — ✅ ·
P-01, P-02, P-03, D-04, D-05 — 💡 рекомендації.

---

*Аудит згенеровано на основі статичного аналізу репозиторію
[Totsamuychel/AI_GeoDetect](https://github.com/Totsamuychel/AI_GeoDetect).
Доповнює попередній `AI_GeoDetect_BugReport.md` (B-01…B-17).*
