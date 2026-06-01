# Global Streetscapes Dataset Download

Скрипт для скачивания данных из **Global Streetscapes** - крупного датасета уличных фотографий.

Dataset: https://huggingface.co/datasets/NUS-UAL/global-streetscapes

## Преимущества Global Streetscapes

- ✅ **Лучшее покрытие** чем OSV-5M
- ✅ **Высокое качество** изображений
- ✅ **Метаданные** с координатами
- ✅ **Streaming mode** - не нужно скачивать весь датасет

## Быстрый старт

### 1. Установите зависимости

```bash
pip install datasets
```

### 2. Установите HF_TOKEN

```bash
# Windows PowerShell
$env:HF_TOKEN="hf_your_token"

# Windows CMD
set HF_TOKEN=hf_your_token

# Linux/Mac
export HF_TOKEN="hf_your_token"
```

Получить токен: https://huggingface.co/settings/tokens

### 3. Запустите скачивание

```bash
python scripts/04_download_global_streetscapes.py
```

Скрипт будет:
1. Загружать изображения в streaming режиме
2. Фильтровать по полигонам городов (Kyiv, Warsaw, Prague)
3. Скачивать до 2500 фото на город
4. Сохранять в `dataset/raw/global_streetscapes/`

## Ожидаемое время

- **Скорость**: ~10-50 изображений в минуту (зависит от интернета)
- **Для 7500 фото**: ~2-5 часов
- **Объем данных**: ~3-5 GB

## Структура результата

```
dataset/raw/global_streetscapes/
├── images/
│   ├── kyiv/
│   │   ├── gss_00001234.jpg
│   │   └── ...
│   ├── warsaw/
│   │   └── ...
│   └── prague/
│       └── ...
└── manifest.csv
```

## Объединение с другими источниками

После скачивания объедините с OSV-5M:

```bash
python scripts/05_merge_manifests.py
```

Это создаст `dataset/raw/merged_manifest.csv` со всеми изображениями.

## Мониторинг прогресса

Скрипт показывает прогресс каждые 100 изображений:

```
Прогресс: Kyiv=234, Warsaw=456, Prague=178
```

## Остановка и продолжение

- **Ctrl+C** для остановки
- При перезапуске скрипт **НЕ пропустит** уже скачанные файлы
- Нужно вручную проверить и продолжить с нужного места

## Устранение проблем

### Ошибка: "HF_TOKEN not found"

**Решение:**
```bash
export HF_TOKEN="hf_..."
python scripts/04_download_global_streetscapes.py
```

### Ошибка: "No module named 'datasets'"

**Решение:**
```bash
pip install datasets
```

### Слишком медленно

**Причины:**
- Медленный интернет
- Перегружен HuggingFace CDN

**Решение:**
- Запустите на ночь
- Используйте VPN если есть блокировки

### Мало изображений для города

**Причина:** В датасете может быть мало покрытия для этого города.

**Решение:**
- Дополните данными из Mapillary:
  ```bash
  python code/download_data.py mapillary \
      --bbox 50.21 30.24 50.59 30.83 \
      --api-key YOUR_MAPILLARY_KEY \
      --max-images 2000
  ```

## Сравнение источников

| Источник | Kyiv | Warsaw | Prague | Итого |
|----------|------|--------|--------|-------|
| OSV-5M | 0 | ~50 | ~50 | ~100 |
| **Global Streetscapes** | **?** | **?** | **?** | **~7500** |
| Mapillary | зависит | зависит | зависит | ~2000+ |

**Рекомендация:** Используйте комбинацию всех трех источников.

## Следующие шаги

1. **Скачайте Global Streetscapes:**
   ```bash
   python scripts/04_download_global_streetscapes.py
   ```

2. **Объедините манифесты:**
   ```bash
   python scripts/05_merge_manifests.py
   ```

3. **Создайте train/val/test splits:**
   ```bash
   python scripts/generate_manifests.py \
       --input dataset/raw/merged_manifest.csv \
       --output-dir dataset/manifests
   ```

4. **Начните обучение:**
   ```bash
   python code/train.py --config configs/baseline.yaml
   ```

---

**Успешного скачивания! 🚀**
