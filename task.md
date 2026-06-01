2. Confusion Matrix + найгірші пари класів
Зараз є per-class accuracy, але немає інформації про те, куди модель помиляється. Confusion matrix (topN найпоширеніших помилок) — критична для розуміння географічних плутанин (наприклад, Warsaw ↔ Kraków).

python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels.numpy(), all_logits.argmax(1).numpy())
# Топ-10 найпоширеніших помилок
errors = [(class_names[i], class_names[j], cm[i,j]) for i in range(N) for j in range(N) if i != j]
top_errors = sorted(errors, key=lambda x: -x[2])[:10]
3. Confidence-калібрація (ECE/MCE)
Немає жодної метрики калібрування — наскільки впевненість моделі (softmax probability) відповідає реальній точності. Expected Calibration Error (ECE) — стандарт для production-моделей:

python
def expected_calibration_error(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            ece += mask.float().mean() * abs(correct[mask].float().mean() - confidences[mask].mean())
    return float(ece)
4. TTA (Test-Time Augmentation)
Немає підтримки TTA — запуску інференсу кілька разів з різними аугментаціями і усередненням. Для геолокаційних моделей TTA типово дає +1–3% до точності:

python
parser.add_argument("--tta", type=int, default=1, help="Кількість TTA ітерацій")
5. Збереження предикцій (CSV)
Результати зберігаються тільки як JSON-агрегати, але немає CSV з per-sample предикціями. Це унеможливлює post-hoc аналіз помилок:

python
preds_df = pd.DataFrame({
    "true_city": [class_names[l] for l in all_labels.numpy()],
    "pred_city": [class_names[i] for i in all_logits.argmax(1).numpy()],
    "confidence": torch.softmax(all_logits, dim=1).max(1).values.numpy(),
    "distance_km": distances,
    "geoscore": scores,
})
preds_df.to_csv(output_path.replace(".json", "_predictions.csv"), index=False)
6. Метрика: Mean Reciprocal Rank (MRR)
Корисна для задач ranking-по-місту — показує, на якій позиції в топ-K знаходиться правильна відповідь в середньому:

python
def mean_reciprocal_rank(logits, labels):
    ranks = logits.argsort(dim=1, descending=True)
    mrr = 0.0
    for i, label in enumerate(labels):
        rank = (ranks[i] == label).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1.0 / rank
    return mrr / len(labels)
7. --tta-flips і --fp16 для швидкого інференсу
Флаг --fp16 / torch.autocast відсутній — на великих тестових наборах суттєво прискорить оцінку без втрати точності:

python
with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
    logits = model(images)
