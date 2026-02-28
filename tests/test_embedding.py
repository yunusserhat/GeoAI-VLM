# -*- coding: utf-8 -*-
"""Tests for geoai_vlm.embedding module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestImageEmbedderInit:
    """Verify ImageEmbedder construction and backend resolution."""

    def test_default_backend_is_transformers(self):
        from geoai_vlm.embedding import ImageEmbedder

        embedder = ImageEmbedder.__new__(ImageEmbedder)
        embedder._backend_type = "transformers"
        assert embedder._backend_type == "transformers"

    def test_invalid_backend_raises(self):
        from geoai_vlm.embedding import ImageEmbedder

        embedder = ImageEmbedder(backend="invalid_backend_xyz")
        with pytest.raises(ValueError, match="(?i)backend"):
            _ = embedder.backend  # lazy init triggers the error


class TestTransformersEmbeddingBackend:
    """Unit-level checks (model loading mocked)."""

    @pytest.mark.skipif(
        not _torch_available(), reason="torch not installed"
    )
    def test_last_token_pool_static(self):
        """_last_token_pool extracts correct positions."""
        from geoai_vlm.embedding import TransformersEmbeddingBackend

        batch_size, seq_len, dim = 2, 5, 4
        hidden = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        # attention_mask: first sample all 1s, second sample last 2 padded
        attention_mask = np.array(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=np.int64
        )

        import torch

        hidden_t = torch.tensor(hidden)
        mask_t = torch.tensor(attention_mask)

        pooled = TransformersEmbeddingBackend._last_token_pool(hidden_t, mask_t)
        assert pooled.shape == (batch_size, dim)
        # First sample: last non-pad token is index 4
        np.testing.assert_allclose(pooled[0].numpy(), hidden[0, 4], atol=1e-5)
        # Second sample: last non-pad token is index 2
        np.testing.assert_allclose(pooled[1].numpy(), hidden[1, 2], atol=1e-5)


class TestMockEmbedder:
    """Verify the mock embedder fixture works correctly."""

    def test_embed_texts_shape(self, mock_embedder):
        texts = ["hello world", "test sentence", "another one"]
        emb = mock_embedder.embed_texts(texts)
        from tests.conftest import MOCK_EMBED_DIM

        assert emb.shape == (3, MOCK_EMBED_DIM)

    def test_embed_texts_normalised(self, mock_embedder):
        texts = ["alpha", "beta"]
        emb = mock_embedder.embed_texts(texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_images(self, mock_embedder, sample_images_dir):
        emb = mock_embedder.embed_images(sample_images_dir)
        assert emb.ndim == 2
        assert emb.shape[0] > 0
