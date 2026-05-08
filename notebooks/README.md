# Analysis Notebooks

Jupyter ноутбуки для анализа результатов обучения моделей геолокации.

## Ноутбуки

### 📊 01_data_exploration.ipynb
Исследовательский анализ датасета OSV-5M:
- Статистика по странам и городам
- Географическое распределение изображений
- Анализ качества данных
- Визуализация примеров

**Статус:** ✅ Готов
**Запуск:** `jupyter notebook 01_data_exploration.ipynb`

---

### 📈 02_training_curves.ipynb
Визуализация кривых обучения для всех моделей:
- Loss curves (train/val)
- Accuracy curves (top-1, top-3, top-5)
- Geospatial metrics (Haversine distance, GeoScore)
- Learning rate schedule
- Сравнительная таблица моделей

**Требования:**
- Обученные модели в `results/{baseline,streetclip,geoclip}/`
- Файлы `training_history.json` или `metrics.json`

**Генерируемые файлы:**
```
results/plots/
├── loss_curves.png
├── accuracy_curves.png
├── geospatial_metrics.png
├── learning_rate.png
└── model_comparison.csv
```

**Статус:** ✅ Создан (2026-05-08)
**Запуск:** `jupyter notebook 02_training_curves.ipynb`

---

### 🔍 03_error_analysis.ipynb
Детальный анализ ошибок и визуализация эмбеддингов:
- **Confusion matrix** по городам
- **Classification report** (precision, recall, F1)
- **Geographic error analysis** (распределение расстояний)
- **t-SNE visualization** эмбеддингов изображений ⭐
- **Interactive error heatmap** на карте
- **Example predictions** (успешные и неуспешные)

**Требования:**
- Обученная модель: `checkpoints/{model}/best_model.pth`
- Тестовый манифест: `dataset/manifests/test.csv`
- GPU рекомендуется (для быстрой генерации эмбеддингов)

**Генерируемые файлы:**
```
results/plots/
├── confusion_matrix.png
├── error_distance_distribution.png
├── tsne_embeddings.png              ⭐ t-SNE визуализация
├── error_heatmap.html               (интерактивная карта)
├── example_predictions.png
└── classification_report.csv
```

**Статус:** ✅ Создан (2026-05-08)
**Запуск:** `jupyter notebook 03_error_analysis.ipynb`

---

## Быстрый старт

### 1. Установка зависимостей

```bash
# Основные библиотеки
pip install jupyter matplotlib seaborn pandas numpy

# Для анализа
pip install scikit-learn folium

# Для работы с моделями
pip install torch torchvision
```

Или полный набор:
```bash
pip install -r ../requirements.txt
```

### 2. Запуск Jupyter

```bash
# Из директории notebooks/
jupyter notebook

# Или из корня проекта
jupyter notebook notebooks/
```

### 3. Порядок выполнения

1. **01_data_exploration.ipynb** - исследование данных
2. Обучить модели (см. `USAGE_GUIDE.md`)
3. **02_training_curves.ipynb** - анализ кривых обучения
4. **03_error_analysis.ipynb** - анализ ошибок + t-SNE

---

## Использование результатов в дипломе

### Глава 3: Эксперименты

Графики из `02_training_curves.ipynb`:
- **Рис. 3.1:** Loss curves - сравнение сходимости моделей
- **Рис. 3.2:** Accuracy curves - динамика точности
- **Рис. 3.3:** Geospatial metrics - географическая точность
- **Табл. 3.1:** `model_comparison.csv` - итоговые метрики

### Глава 4: Анализ результатов

Графики из `03_error_analysis.ipynb`:
- **Рис. 4.1:** Confusion matrix - ошибки между городами
- **Рис. 4.2:** t-SNE embeddings - кластеризация изображений по городам ⭐
- **Рис. 4.3:** Error distribution - распределение географических ошибок
- **Рис. 4.4:** Example predictions - примеры работы модели
- **Табл. 4.1:** `classification_report.csv` - precision/recall по городам

### Приложения

- **Приложение A:** Interactive error heatmap (`error_heatmap.html`)

---

## Настройка ноутбуков

### Изменение модели для анализа

В `03_error_analysis.ipynb`:
```python
# Выберите нужный checkpoint
MODEL_CHECKPOINT = '../checkpoints/baseline/best_model.pth'
# MODEL_CHECKPOINT = '../checkpoints/streetclip/best_model.pth'
# MODEL_CHECKPOINT = '../checkpoints/geoclip/best_model.pth'
```

### Ускорение t-SNE

Если t-SNE работает медленно:
```python
# Уменьшите количество образцов
n_samples = 1000  # вместо 2000

# Уменьшите количество итераций
tsne = TSNE(n_iter=500)  # вместо 1000

# Используйте меньший perplexity
plot_tsne_embeddings(embeddings, labels, class_names, perplexity=15)
```

### Экспорт для диплома

Все графики сохраняются в высоком разрешении (300 DPI):
```python
plt.savefig('../results/plots/figure_name.png', dpi=300, bbox_inches='tight')
```

Для LaTeX используйте формат PNG или PDF.

---

## Устранение проблем

### Ошибка: "No training history found"

**Причина:** Модели еще не обучены или отсутствуют файлы истории.

**Решение:**
1. Обучите модели: `python code/train.py --config configs/baseline.yaml`
2. Проверьте наличие `training_history.json` в `results/{model}/`
3. Если используете W&B, экспортируйте историю в JSON

### Ошибка: "CUDA out of memory"

**Причина:** Недостаточно GPU памяти для inference.

**Решение:**
1. Уменьшите `batch_size` в DataLoader:
   ```python
   test_loader = torch.utils.data.DataLoader(
       test_dataset,
       batch_size=16,  # вместо 32
       ...
   )
   ```
2. Используйте CPU: `device = 'cpu'`
3. Обрабатывайте меньше образцов

### Ошибка: "Kernel died" при t-SNE

**Причина:** Не хватает RAM для t-SNE.

**Решение:**
1. Уменьшите `n_samples`: `n_samples = 500`
2. Используйте `IncrementalPCA` перед t-SNE для уменьшения размерности
3. Закройте другие программы

---

## Дополнительные ресурсы

- **Документация:** `../USAGE_GUIDE.md`
- **Конфигурации:** `../configs/`
- **Обучение:** `../code/train.py`
- **Оценка:** `../code/evaluate.py`

---

## Контрибуция

При создании новых ноутбуков следуйте структуре:
1. Заголовок с описанием цели
2. Импорты и настройки
3. Секции с анализом (markdown + code)
4. Визуализации в высоком разрешении
5. Сохранение результатов в `results/plots/`
6. Итоговая секция с Summary

Все ноутбуки должны работать из коробки после обучения моделей.
