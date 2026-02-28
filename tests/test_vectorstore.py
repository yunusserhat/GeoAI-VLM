# -*- coding: utf-8 -*-
"""Tests for geoai_vlm.vectorstore module."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MOCK_EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_embeddings(n: int, dim: int = MOCK_EMBED_DIM) -> np.ndarray:
    rng = np.random.RandomState(0)
    emb = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


# ===========================================================================
# ChromaDB backend
# ===========================================================================
class TestChromaVectorStore:
    """Test ChromaVectorStore add / query / delete / persist round-trip."""

    @pytest.fixture
    def store(self, tmp_path):
        from geoai_vlm.vectorstore import ChromaVectorStore

        return ChromaVectorStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="unit_test",
        )

    def test_add_and_count(self, store):
        embs = _random_embeddings(10)
        ids = [f"id_{i}" for i in range(10)]
        store.add(embeddings=embs, ids=ids)
        assert store.count() == 10

    def test_query_returns_correct_count(self, store):
        embs = _random_embeddings(20)
        ids = [f"id_{i}" for i in range(20)]
        store.add(embeddings=embs, ids=ids)

        results = store.query(query_embedding=embs[0], n_results=5)
        assert len(results["ids"]) == 5

    def test_query_with_metadata(self, store):
        embs = _random_embeddings(5)
        ids = [f"id_{i}" for i in range(5)]
        metadatas = [{"label": f"label_{i}"} for i in range(5)]
        store.add(embeddings=embs, ids=ids, metadatas=metadatas)

        results = store.query(query_embedding=embs[0], n_results=2)
        assert "metadatas" in results
        assert len(results["metadatas"]) == 2

    def test_delete(self, store):
        embs = _random_embeddings(5)
        ids = [f"id_{i}" for i in range(5)]
        store.add(embeddings=embs, ids=ids)
        store.delete(ids=["id_0", "id_1"])
        assert store.count() == 3

    def test_persist_and_reload(self, tmp_path):
        from geoai_vlm.vectorstore import ChromaVectorStore

        path = str(tmp_path / "chroma_persist")
        store1 = ChromaVectorStore(persist_directory=path, collection_name="persist_test")
        embs = _random_embeddings(8)
        ids = [f"id_{i}" for i in range(8)]
        store1.add(embeddings=embs, ids=ids)
        store1.persist()

        store2 = ChromaVectorStore(persist_directory=path, collection_name="persist_test")
        assert store2.count() == 8


# ===========================================================================
# FAISS backend
# ===========================================================================
class TestFAISSVectorStore:
    """Test FAISSVectorStore add / query / delete / persist round-trip."""

    @pytest.fixture
    def store(self):
        from geoai_vlm.vectorstore import FAISSVectorStore

        return FAISSVectorStore(dimension=MOCK_EMBED_DIM)

    def test_add_and_count(self, store):
        embs = _random_embeddings(10)
        ids = [f"id_{i}" for i in range(10)]
        store.add(embeddings=embs, ids=ids)
        assert store.count() == 10

    def test_query_returns_correct_count(self, store):
        embs = _random_embeddings(20)
        ids = [f"id_{i}" for i in range(20)]
        store.add(embeddings=embs, ids=ids)

        results = store.query(query_embedding=embs[0], n_results=5)
        assert len(results["ids"]) == 5

    def test_query_with_metadata(self, store):
        embs = _random_embeddings(5)
        ids = [f"id_{i}" for i in range(5)]
        metadatas = [{"label": f"label_{i}"} for i in range(5)]
        store.add(embeddings=embs, ids=ids, metadatas=metadatas)

        results = store.query(query_embedding=embs[0], n_results=2)
        assert "metadatas" in results

    def test_delete(self, store):
        embs = _random_embeddings(5)
        ids = [f"id_{i}" for i in range(5)]
        store.add(embeddings=embs, ids=ids)
        store.delete(ids=["id_0", "id_1"])
        assert store.count() == 3

    def test_persist_and_reload(self, tmp_path, store):
        embs = _random_embeddings(8)
        ids = [f"id_{i}" for i in range(8)]
        metadatas = [{"x": i} for i in range(8)]
        store.add(embeddings=embs, ids=ids, metadatas=metadatas)

        path = str(tmp_path / "faiss_persist")
        store.persist(path)

        from geoai_vlm.vectorstore import FAISSVectorStore

        store2 = FAISSVectorStore.load(path)
        assert store2.count() == 8


# ===========================================================================
# VectorDB orchestrator
# ===========================================================================
class TestVectorDB:
    """Integration-level tests for the VectorDB helper class."""

    def test_build_from_dataframe(self, sample_gdf, mock_embedder, tmp_path):
        from geoai_vlm.vectorstore import VectorDB

        db = VectorDB(
            embedder=mock_embedder,
            store_backend="faiss",
            dimension=MOCK_EMBED_DIM,
        )
        db.build(
            gdf=sample_gdf,
            text_column="scene_narrative",
            id_column="image_id",
        )
        assert db.store.count() == len(sample_gdf)

    def test_search_returns_results(self, sample_gdf, mock_embedder):
        from geoai_vlm.vectorstore import VectorDB

        db = VectorDB(
            embedder=mock_embedder,
            store_backend="faiss",
            dimension=MOCK_EMBED_DIM,
        )
        db.build(
            gdf=sample_gdf,
            text_column="scene_narrative",
            id_column="image_id",
        )
        results = db.search("busy street", n_results=5)
        assert len(results) == 5
