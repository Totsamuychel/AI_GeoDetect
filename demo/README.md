# AI_GeoDetect — Gradio-демо

Легке демо для захисту диплому: завантажуєш фото вулиці → модель передбачає топ-5 міст з ймовірностями, показує на карті.

## Встановлення

```bash
pip install gradio folium
# + базові залежності проекту
pip install -r ../requirements.txt
```

## Запуск

```bash
# З навченим чекпоінтом
python demo/app.py --checkpoint checkpoints/best_model.pth

# Без чекпоінту — «заглушковий» режим для перевірки UI
python demo/app.py --checkpoint nonexistent.pth

# З публічним посиланням для онлайн-демо
python demo/app.py --checkpoint checkpoints/best_model.pth --share
```

За замовчуванням інтерфейс доступний на `http://localhost:7860`.
