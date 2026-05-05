- [x] Единственная проблема — дублирование predict() . Метод практически одинаков во всех трёх классах (BaselineCNN, StreetCLIPModel, GeoCLIPModel). Нужна базовая абстракция:

```python
class GeoModelMixin:
    @torch.no_grad()
    def predict(self, x, class_names=None, top_k=5):
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        logits = self._get_logits(x)  # abstract method
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)
        return [{"class": class_names[i] if class_names else str(i),
                 "index": i, "prob": round(p, 6)}
                for p, i in zip(top_probs[0].tolist(), top_indices[0].tolist())]
```

- [x] StreetCLIPModel хранит self.processor, но нигде не использует его в forward() . Это создаёт путаницу: пользователь не знает, нужно ли ему предобрабатывать изображения самому. Либо добавить process_image() метод, либо убрать processor.

- [x] if __name__ == "__main__": с ручными тестами лучше заменить на pytest в папке tests/ — профессиональный подход.
