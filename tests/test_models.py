"""
test_models.py — Юніт-тести для code/models.py.

Швидкі forward-pass тести (без завантаження CLIP/StreetCLIP з HuggingFace).
Тести для StreetCLIP / GeoCLIP пропускаються, якщо transformers недоступний
або інтернет відсутній.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models import (
    BaselineCNN,
    RandomFourierGPSEncoder,
    build_model,
)


# ──────────────────────────────────────────────────────────────────────────────
# RandomFourierGPSEncoder (чистий PyTorch, без HuggingFace)
# ──────────────────────────────────────────────────────────────────────────────

class TestRandomFourierGPSEncoder:
    def test_output_shape(self) -> None:
        enc = RandomFourierGPSEncoder(embed_dim=512, num_frequencies=256)
        coords = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        out = enc(coords)
        assert out.shape == (2, 512)

    def test_output_is_l2_normalized(self) -> None:
        enc = RandomFourierGPSEncoder(embed_dim=128, num_frequencies=64)
        coords = torch.randn(4, 2)
        out = enc(coords)
        norms = torch.linalg.norm(out, dim=1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_different_coords_different_embeddings(self) -> None:
        enc = RandomFourierGPSEncoder(embed_dim=128, num_frequencies=64)
        enc.eval()
        a = enc(torch.tensor([[0.1, 0.2]]))
        b = enc(torch.tensor([[-0.8, 0.4]]))
        assert not torch.allclose(a, b)


# ──────────────────────────────────────────────────────────────────────────────
# BaselineCNN — важкий, маркуємо slow, бо завантажує EfficientNet-B2
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestBaselineCNN:
    @pytest.fixture(scope="class")
    def model(self) -> BaselineCNN:
        # pretrained=False уникне завантаження ваг із мережі у CI
        return BaselineCNN(num_classes=8, pretrained=False)

    def test_forward_shape(self, model: BaselineCNN) -> None:
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 8)

    def test_embeddings_normalized(self, model: BaselineCNN) -> None:
        x = torch.randn(2, 3, 224, 224)
        model.eval()
        emb = model.get_embeddings(x)
        norms = torch.linalg.norm(emb, dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-4, rtol=1e-4)

    def test_embeddings_allow_gradient(self, model: BaselineCNN) -> None:
        """Перевіряє fix: get_embeddings не повинен блокувати backprop."""
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        model.train()
        emb = model.get_embeddings(x)
        loss = emb.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_freeze_backbone(self, model: BaselineCNN) -> None:
        model.freeze_backbone()
        for p in model.features.parameters():
            assert not p.requires_grad

    def test_unfreeze_last_n_blocks(self, model: BaselineCNN) -> None:
        model.freeze_backbone()
        model.unfreeze_last_n_blocks(n=2)
        blocks = list(model.features.children())
        # Останні 2 блоки мають мати хоча б один trainable parameter
        trainable_in_last = any(
            any(p.requires_grad for p in b.parameters()) for b in blocks[-2:]
        )
        assert trainable_in_last

    def test_predict_returns_top_k(self, model: BaselineCNN) -> None:
        x = torch.randn(3, 224, 224)
        preds = model.predict(x, class_names=[f"C{i}" for i in range(8)], top_k=3)
        assert len(preds) == 3
        assert {"class", "index", "prob"} <= set(preds[0].keys())
        assert all(0.0 <= p["prob"] <= 1.0 for p in preds)


# ──────────────────────────────────────────────────────────────────────────────
# build_model factory
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildModel:
    def test_invalid_arch_raises(self) -> None:
        with pytest.raises(ValueError):
            build_model("nonexistent_arch", num_classes=5)

    @pytest.mark.slow
    def test_baseline_alias(self) -> None:
        m = build_model("baseline", num_classes=5, pretrained=False)
        assert isinstance(m, BaselineCNN)
        assert m.num_classes == 5
