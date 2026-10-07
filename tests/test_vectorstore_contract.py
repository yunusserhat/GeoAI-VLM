# -*- coding: utf-8 -*-
"""
Vector search contract tests.

The 2026-09-12 review noted that both backends report their scores in a field
called ``distance`` without a stated metric or ordering direction, and that the
existing tests assert result *counts* only -- which cannot detect a reversed
ranking. These tests pin the contract with a hand-computable fixture.

The fixture is six unit vectors whose cosine similarity to the query can be
read off by eye, so the expected ranking is written out by hand rather than
taken from whatever the library happens to return:

    id   vector              cos sim   cosine distance   squared L2
    a    ( 1.0,  0.0, 0, 0)     1.0          0.0             0.0
    b    ( 0.8,  0.6, 0, 0)     0.8          0.2             0.4
    c    ( 0.6,  0.8, 0, 0)     0.6          0.4             0.8
    d    ( 0.0,  1.0, 0, 0)     0.0          1.0             2.0
    e    (-0.6,  0.8, 0, 0)    -0.6          1.6             3.2
    f    (-1.0,  0.0, 0, 0)    -1.0          2.0             4.0

Query is a itself, so the expected order is always a, b, c, d, e, f.
No model, network or API key is used.
"""

from __future__ import annotations

import numpy as np
import pytest


QUERY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

IDS = ["a", "b", "c", "d", "e", "f"]
VECTORS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],
        [0.6, 0.8, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-0.6, 0.8, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
COSINE_SIM = [1.0, 0.8, 0.6, 0.0, -0.6, -1.0]
COSINE_DIST = [0.0, 0.2, 0.4, 1.0, 1.6, 2.0]
SQUARED_L2 = [0.0, 0.4, 0.8, 2.0, 3.2, 4.0]

METADATAS = [
    {"kind": "near"},
    {"kind": "near"},
    {"kind": "near"},
    {"kind": "far"},
    {"kind": "far"},
    {"kind": "far"},
]


def _chroma(tmp_path, name="contract_probe", **kw):
    from geoai_vlm.vectorstore import ChromaVectorStore

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / name), collection_name=name, **kw
    )
    store.add(ids=IDS, embeddings=VECTORS, metadatas=METADATAS)
    return store


def _faiss(metric="ip", **kw):
    from geoai_vlm.vectorstore import FAISSVectorStore

    store = FAISSVectorStore(metric=metric, **kw)
    store.add(ids=IDS, embeddings=VECTORS, metadatas=METADATAS)
    return store


@pytest.fixture
def all_backends(tmp_path):
    """Every (name, store, expected_raw_scores) combination under test."""
    return [
        ("chroma-cosine", _chroma(tmp_path), COSINE_DIST),
        ("faiss-ip", _faiss("ip"), COSINE_SIM),
        ("faiss-l2", _faiss("l2"), SQUARED_L2),
    ]


# ---------------------------------------------------------------------------
# The raw numbers each backend produces
# ---------------------------------------------------------------------------
class TestRawScores:
    """Each backend's own numbers must match the hand-computed values."""

    def test_chroma_reports_cosine_distance(self, tmp_path):
        res = _chroma(tmp_path).query(QUERY, n_results=6)
        assert res["ids"] == IDS
        assert res["distances"] == pytest.approx(COSINE_DIST, abs=1e-5)

    def test_faiss_ip_reports_inner_product(self):
        res = _faiss("ip").query(QUERY, n_results=6)
        assert res["ids"] == IDS
        assert res["distances"] == pytest.approx(COSINE_SIM, abs=1e-5)

    def test_faiss_l2_reports_squared_distance(self):
        res = _faiss("l2").query(QUERY, n_results=6)
        assert res["ids"] == IDS
        assert res["distances"] == pytest.approx(SQUARED_L2, abs=1e-5)


# ---------------------------------------------------------------------------
# The uniform contract layered on top of them
# ---------------------------------------------------------------------------
class TestUniformContract:
    """Callers must be able to act on results without knowing the backend."""

    def test_every_backend_declares_its_metric_and_direction(self, all_backends):
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=6)
            assert "metric" in res, f"{name}: result does not state its metric"
            assert res["direction"] in ("lower_is_closer", "higher_is_closer"), (
                f"{name}: result does not state its ordering direction"
            )

    def test_direction_matches_the_actual_numbers(self, all_backends):
        """The declared direction must describe the raw scores, not contradict them."""
        for name, store, expected in all_backends:
            res = store.query(QUERY, n_results=6)
            raw = list(res["distances"])
            if res["direction"] == "lower_is_closer":
                assert raw == sorted(raw), f"{name}: declared ascending but is not"
            else:
                assert raw == sorted(raw, reverse=True), (
                    f"{name}: declared descending but is not"
                )

    def test_similarity_is_always_higher_is_closer(self, all_backends):
        """This is the field a caller can use without a per-backend branch."""
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=6)
            sims = list(res["similarity"])
            assert sims == sorted(sims, reverse=True), (
                f"{name}: similarity is not ordered best-first: {sims}"
            )
            assert sims[0] > sims[-1], f"{name}: similarity does not discriminate"

    def test_cosine_backends_report_true_cosine_similarity(self, tmp_path):
        """Where the metric allows it, similarity is the real cosine value."""
        for store in (_chroma(tmp_path), _faiss("ip")):
            res = store.query(QUERY, n_results=6)
            assert res["similarity"] == pytest.approx(COSINE_SIM, abs=1e-5)

    def test_rank_is_zero_based_and_best_first(self, all_backends):
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=6)
            assert res["rank"] == list(range(6)), f"{name}: {res['rank']}"
            assert res["ids"] == IDS, f"{name}: {res['ids']}"

    def test_sorting_by_similarity_preserves_returned_order(self, all_backends):
        """The trap: sorting by the raw field reverses FAISS inner-product results."""
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=6)
            paired = sorted(
                zip(res["ids"], res["similarity"]), key=lambda t: -t[1]
            )
            assert [i for i, _ in paired] == res["ids"], (
                f"{name}: sorting by similarity disagrees with the returned order"
            )


# ---------------------------------------------------------------------------
# Same-id update
# ---------------------------------------------------------------------------
class TestDuplicateIds:
    """Re-adding an id must update it, not create a second entry."""

    def test_chroma_updates_in_place(self, tmp_path):
        store = _chroma(tmp_path)
        store.add(ids=["a"], embeddings=np.array([[0.0, 0.0, 1.0, 0.0]], np.float32))
        assert store.count() == 6

    def test_faiss_updates_in_place(self):
        store = _faiss("ip")
        store.add(ids=["a"], embeddings=np.array([[0.0, 0.0, 1.0, 0.0]], np.float32))
        assert store.count() == 6, "re-adding an id appended a duplicate entry"
        res = store.query(QUERY, n_results=6)
        assert res["ids"].count("a") == 1, f"id 'a' appears twice: {res['ids']}"

    def test_updated_vector_is_the_one_searched(self):
        """The replacement must actually take effect, not just keep the count."""
        store = _faiss("ip")
        # Move 'a' far away from the query; it must stop being the top hit.
        store.add(ids=["a"], embeddings=np.array([[0.0, 0.0, 1.0, 0.0]], np.float32))
        res = store.query(QUERY, n_results=6)
        assert res["ids"][0] == "b", f"stale vector still ranked first: {res['ids']}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_faiss_reload_preserves_metric(self, tmp_path):
        """A reloaded store that misreports its metric misreports its direction."""
        from geoai_vlm.vectorstore import FAISSVectorStore

        store = _faiss("l2")
        store.persist(tmp_path / "idx")
        reloaded = FAISSVectorStore.load(tmp_path / "idx")

        assert reloaded.metric == "l2", (
            f"metric became {reloaded.metric!r} on reload; the stored numbers are "
            "still L2, so the declared direction would now be wrong"
        )
        assert reloaded.index_type == store.index_type

    def test_faiss_reload_preserves_results_and_contract(self, tmp_path):
        from geoai_vlm.vectorstore import FAISSVectorStore

        store = _faiss("l2")
        before = store.query(QUERY, n_results=6)
        store.persist(tmp_path / "idx2")
        after = FAISSVectorStore.load(tmp_path / "idx2").query(QUERY, n_results=6)

        assert after["ids"] == before["ids"]
        assert after["distances"] == pytest.approx(before["distances"], abs=1e-5)
        assert after["metric"] == before["metric"]
        assert after["direction"] == before["direction"]
        assert after["similarity"] == pytest.approx(before["similarity"], abs=1e-5)

    def test_faiss_reload_preserves_metadata(self, tmp_path):
        from geoai_vlm.vectorstore import FAISSVectorStore

        _faiss("ip").persist(tmp_path / "idx3")
        reloaded = FAISSVectorStore.load(tmp_path / "idx3")
        res = reloaded.query(QUERY, n_results=3)
        assert [m["kind"] for m in res["metadatas"]] == ["near", "near", "near"]


# ---------------------------------------------------------------------------
# Delete, empty results, filters
# ---------------------------------------------------------------------------
class TestDeleteAndEmpty:
    def test_delete_removes_from_results(self, all_backends):
        for name, store, _ in all_backends:
            store.delete(["a", "b"])
            res = store.query(QUERY, n_results=6)
            assert "a" not in res["ids"], f"{name}: deleted id still returned"
            assert "b" not in res["ids"], f"{name}: deleted id still returned"
            assert store.count() == 4, f"{name}: count after delete is {store.count()}"

    def test_empty_store_returns_empty_result_not_an_error(self, tmp_path):
        from geoai_vlm.vectorstore import ChromaVectorStore, FAISSVectorStore

        empty_faiss = FAISSVectorStore()
        empty_chroma = ChromaVectorStore(
            persist_directory=str(tmp_path / "empty_store"),
            collection_name="empty_store",
        )
        for store in (empty_faiss, empty_chroma):
            res = store.query(QUERY, n_results=5)
            assert res["ids"] == []
            assert res["distances"] == []
            assert res["similarity"] == []
            assert res["rank"] == []

    def test_n_results_larger_than_collection_is_truncated(self, all_backends):
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=50)
            assert len(res["ids"]) == 6, f"{name}"
            assert len(res["distances"]) == 6, f"{name}"
            assert len(res["similarity"]) == 6, f"{name}"


class TestMetadataFilter:
    def test_filter_restricts_results(self, all_backends):
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=6, where={"kind": "far"})
            assert res["ids"] == ["d", "e", "f"], f"{name}: {res['ids']}"
            assert all(m["kind"] == "far" for m in res["metadatas"]), name

    def test_filter_keeps_ranking_and_contract(self, all_backends):
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=6, where={"kind": "near"})
            assert res["ids"] == ["a", "b", "c"], f"{name}: {res['ids']}"
            assert res["rank"] == [0, 1, 2], f"{name}: {res['rank']}"
            sims = list(res["similarity"])
            assert sims == sorted(sims, reverse=True), name

    def test_filter_matching_nothing_returns_empty(self, all_backends):
        for name, store, _ in all_backends:
            res = store.query(QUERY, n_results=6, where={"kind": "nonexistent"})
            assert res["ids"] == [], f"{name}: {res['ids']}"


# ---------------------------------------------------------------------------
# Index types must not be assumed interchangeable
# ---------------------------------------------------------------------------
class TestIndexTypes:
    def test_ivf_with_too_few_vectors_raises_a_clear_error(self):
        """faiss raises a bare clustering RuntimeError; say what to do instead."""
        from geoai_vlm.vectorstore import FAISSVectorStore

        store = FAISSVectorStore(metric="ip", index_type="ivf", nlist=100)
        with pytest.raises(ValueError) as exc:
            store.add(ids=IDS, embeddings=VECTORS)

        message = str(exc.value).lower()
        assert "nlist" in message
        assert "flat" in message, "the error should name a workable alternative"

    def test_ivf_with_enough_vectors_works_and_keeps_the_contract(self):
        from geoai_vlm.vectorstore import FAISSVectorStore

        rng = np.random.RandomState(0)
        many = rng.randn(400, 4).astype(np.float32)
        many /= np.linalg.norm(many, axis=1, keepdims=True)
        ids = [f"v{i}" for i in range(400)]

        store = FAISSVectorStore(metric="ip", index_type="ivf", nlist=4)
        store.add(ids=ids, embeddings=many)
        res = store.query(QUERY, n_results=5)

        assert len(res["ids"]) == 5
        sims = list(res["similarity"])
        assert sims == sorted(sims, reverse=True)
        assert res["direction"] == "higher_is_closer"


# ---------------------------------------------------------------------------
# The high-level search() surface
# ---------------------------------------------------------------------------
class _FixedQueryEmbedder:
    """Embedder stub that always emits the hand-computed query vector.

    The shared mock embedder produces 128-dimensional vectors, which cannot be
    queried against this 4-dimensional fixture; fixing the query also lets the
    expected ranking be asserted exactly.
    """

    instruction = "fixed"

    class _Backend:
        def embed(self, inputs, instruction=""):
            return QUERY.reshape(1, -1).copy()

    def __init__(self):
        self.backend = self._Backend()


class TestSearchDataFrame:
    def _db(self, backend, tmp_path):
        from geoai_vlm.vectorstore import VectorDB

        if backend == "chromadb":
            db = VectorDB(
                embedder=_FixedQueryEmbedder(),
                store_backend="chromadb",
                persist_directory=str(tmp_path / "sdf_store"),
                collection_name="sdf_store",
            )
        else:
            db = VectorDB(embedder=_FixedQueryEmbedder(), store_backend="faiss")
        db.store.add(ids=IDS, embeddings=VECTORS, metadatas=METADATAS)
        return db

    @pytest.mark.parametrize("backend", ["chromadb", "faiss"])
    def test_search_returns_rank_and_similarity(self, backend, tmp_path):
        db = self._db(backend, tmp_path)
        df = db.search(query_text="anything", n_results=4)

        for col in ("id", "rank", "similarity", "distance"):
            assert col in df.columns, f"{backend}: missing column {col}"
        assert list(df["rank"]) == [0, 1, 2, 3]
        assert list(df["id"]) == ["a", "b", "c", "d"], f"{backend}: {list(df['id'])}"
        assert list(df["similarity"]) == sorted(df["similarity"], reverse=True), (
            f"{backend}: rows are not ordered best-first"
        )

    @pytest.mark.parametrize("backend", ["chromadb", "faiss"])
    def test_search_reports_metric_in_attrs(self, backend, tmp_path):
        db = self._db(backend, tmp_path)
        df = db.search(query_text="anything", n_results=3)
        assert df.attrs.get("metric"), f"{backend}: metric not reported"
        assert df.attrs.get("direction") in (
            "lower_is_closer",
            "higher_is_closer",
        ), f"{backend}: direction not reported"

    @pytest.mark.parametrize("backend", ["chromadb", "faiss"])
    def test_search_passes_metadata_filter_through(self, backend, tmp_path):
        db = self._db(backend, tmp_path)
        df = db.search(query_text="anything", n_results=6, where={"kind": "far"})
        assert list(df["id"]) == ["d", "e", "f"], f"{backend}: {list(df['id'])}"

    @pytest.mark.parametrize("backend", ["chromadb", "faiss"])
    def test_search_with_no_matches_returns_empty_frame(self, backend, tmp_path):
        db = self._db(backend, tmp_path)
        df = db.search(query_text="anything", n_results=6, where={"kind": "none"})
        assert len(df) == 0
        for col in ("id", "rank", "similarity", "distance"):
            assert col in df.columns, f"{backend}: empty frame lost column {col}"
