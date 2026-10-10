# -*- coding: utf-8 -*-
"""
ClipEmbeddingBackend with a fake dual encoder, and its VectorDB contract (P0-B).

The fake model projects an image's mean colour and a text's character
statistics into 8 dimensions with fixed matrices, so every expected vector can
be computed by hand. Both output conventions are covered: transformers 4
(``get_*_features`` returns the tensor) and transformers 5 (it returns an
output object whose ``pooler_output`` holds the features). No torch, model or
network is needed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from geoai_vlm.embedding import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    ClipEmbeddingBackend,
    ImageEmbedder,
)

DIM = 8
RNG = np.random.RandomState(0)
W_IMAGE = RNG.randn(3, DIM).astype(np.float32)
W_TEXT = RNG.randn(2, DIM).astype(np.float32)


class FakeBatch(dict):
    def to(self, device, dtype=None):
        return self


class PoolerOutput(dict):
    """Stands in for transformers 5's BaseModelOutputWithPooling."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class FakeClipProcessor:
    def __init__(self):
        self.text_options = []
        self.image_batches = []

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        if images is not None:
            self.image_batches.append(len(images))
            means = [np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0 for img in images]
            return FakeBatch(pixel_values=np.array(means, dtype=np.float32))
        self.text_options.append(kwargs)
        self.texts = list(text)
        stats = [[(sum(map(ord, t)) % 97) / 97.0, len(t) / 50.0] for t in text]
        return FakeBatch(input_ids=np.array(stats, dtype=np.float32))


class FakeClipModel:
    device = "cpu"
    dtype = "float32"

    def __init__(self, model_type="siglip2", v5_outputs=False):
        self.config = types.SimpleNamespace(model_type=model_type)
        self.v5_outputs = v5_outputs

    def _wrap(self, value):
        return PoolerOutput(pooler_output=value) if self.v5_outputs else value

    def get_image_features(self, pixel_values):
        return self._wrap(pixel_values @ W_IMAGE)

    def get_text_features(self, input_ids):
        return self._wrap(input_ids @ W_TEXT)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _expected_image(color):
    mean = np.array(color, dtype=np.float32) / 255.0
    return _unit(mean @ W_IMAGE)


def _expected_text(text):
    stats = np.array([(sum(map(ord, text)) % 97) / 97.0, len(text) / 50.0], dtype=np.float32)
    return _unit(stats @ W_TEXT)


def _backend(model_type="siglip2", v5_outputs=False, **kwargs):
    backend = ClipEmbeddingBackend("fake/clip", **kwargs)
    backend._model = FakeClipModel(model_type, v5_outputs)
    backend._processor = FakeClipProcessor()
    return backend


COLORS = {"red": (200, 30, 30), "green": (30, 180, 40), "blue": (20, 40, 210), "grey": (128, 128, 128)}


@pytest.fixture
def images(tmp_path):
    paths = {}
    for name, color in COLORS.items():
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (12, 12), color).save(p)
        paths[name] = str(p)
    return paths


# ---------------------------------------------------------------------------
# Backend behaviour
# ---------------------------------------------------------------------------
class TestClipBackend:
    @pytest.mark.parametrize("v5_outputs", [False, True], ids=["transformers4", "transformers5"])
    def test_image_and_text_vectors_match_hand_computation(self, images, v5_outputs):
        backend = _backend(v5_outputs=v5_outputs)
        vecs = backend.embed([{"image": images["red"]}, {"text": "a tree-lined street"}])
        assert np.allclose(vecs[0], _expected_image(COLORS["red"]), atol=1e-5)
        assert np.allclose(vecs[1], _expected_text("a tree-lined street"), atol=1e-5)

    def test_all_modalities_share_one_dimension_and_are_normalised(self, images):
        backend = _backend()
        vecs = backend.embed(
            [{"image": images["green"]}, {"text": "cycle lane"}, {"image": images["blue"], "text": "kerb"}]
        )
        assert vecs.shape == (3, DIM)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)

    def test_image_and_text_together_are_fused_by_weighted_mean(self, images):
        backend = _backend(image_weight=0.5)
        fused = backend.embed([{"image": images["blue"], "text": "kerb"}])[0]
        expected = _unit(0.5 * _expected_image(COLORS["blue"]) + 0.5 * _expected_text("kerb"))
        assert np.allclose(fused, expected, atol=1e-5)

    def test_image_weight_changes_the_fusion(self, images):
        heavy = _backend(image_weight=0.9).embed([{"image": images["blue"], "text": "kerb"}])[0]
        expected = _unit(0.9 * _expected_image(COLORS["blue"]) + 0.1 * _expected_text("kerb"))
        assert np.allclose(heavy, expected, atol=1e-5)

    def test_result_is_independent_of_batching(self, images):
        backend = _backend()
        inputs = [{"image": images[c]} for c in COLORS] + [{"text": t} for t in ("a", "bb", "kerb")]
        together = backend.embed(inputs)
        alone = np.vstack([backend.embed([inp]) for inp in inputs])
        assert np.allclose(together, alone, atol=1e-6)

    def test_order_is_preserved_with_mixed_modalities(self, images):
        backend = _backend()
        vecs = backend.embed([{"text": "x"}, {"image": images["red"]}, {"text": "yy"}])
        assert np.allclose(vecs[0], _expected_text("x"), atol=1e-5)
        assert np.allclose(vecs[1], _expected_image(COLORS["red"]), atol=1e-5)
        assert np.allclose(vecs[2], _expected_text("yy"), atol=1e-5)

    def test_instruction_is_ignored(self, images):
        backend = _backend()
        a = backend.embed([{"text": "kerb"}], instruction="one")
        b = backend.embed([{"text": "kerb"}], instruction="another")
        assert np.allclose(a, b)

    def test_empty_input_embeds_empty_text(self):
        backend = _backend()
        vec = backend.embed([{}])
        assert vec.shape == (1, DIM)
        assert backend._processor.texts == [""]

    def test_siglip_text_uses_max_length_padding(self):
        backend = _backend(model_type="siglip2")
        backend.embed([{"text": "kerb"}])
        assert backend._processor.text_options[-1] == {
            "padding": "max_length", "truncation": True, "max_length": 64,
        }

    def test_clip_text_uses_dynamic_padding(self):
        backend = _backend(model_type="clip")
        backend.embed([{"text": "kerb"}])
        assert backend._processor.text_options[-1] == {"padding": True, "truncation": True}

    def test_explicit_padding_overrides_the_default(self):
        backend = _backend(model_type="clip", text_padding="max_length", max_text_length=77)
        backend.embed([{"text": "kerb"}])
        assert backend._processor.text_options[-1]["max_length"] == 77

    def test_pil_images_are_accepted(self):
        vec = _backend().embed([{"image": Image.new("RGB", (4, 4), COLORS["grey"])}])
        assert np.allclose(vec[0], _expected_image(COLORS["grey"]), atol=1e-5)

    def test_several_images_in_one_input_are_rejected(self, images):
        with pytest.raises(ValueError, match="one image per input"):
            _backend().embed([{"image": [images["red"], images["blue"]]}])

    def test_remote_urls_are_not_fetched(self):
        with pytest.raises(ValueError, match="not fetched"):
            _backend().embed([{"image": "https://example.org/x.jpg"}])

    def test_invalid_image_weight(self):
        with pytest.raises(ValueError):
            ClipEmbeddingBackend(image_weight=1.5)

    def test_trust_remote_code_is_off_by_default(self):
        assert ClipEmbeddingBackend().trust_remote_code is False


class TestImageEmbedderClip:
    def test_clip_backend_with_explicit_model(self):
        embedder = ImageEmbedder(backend="clip", model_name="google/siglip2-base-patch16-224")
        assert isinstance(embedder.backend, ClipEmbeddingBackend)
        assert embedder.backend.model_name == "google/siglip2-base-patch16-224"

    def test_clip_backend_defaults_to_a_clip_model(self):
        assert ImageEmbedder(backend="clip").model_name == DEFAULT_CLIP_MODEL

    def test_qwen_default_is_unchanged(self):
        assert ImageEmbedder().model_name == DEFAULT_EMBEDDING_MODEL
        assert ImageEmbedder(backend="transformers").model_name == DEFAULT_EMBEDDING_MODEL

    def test_qwen_backends_do_not_trust_remote_code_by_default(self):
        from geoai_vlm.embedding import TransformersEmbeddingBackend, VLLMEmbeddingBackend

        assert TransformersEmbeddingBackend().trust_remote_code is False
        assert VLLMEmbeddingBackend().trust_remote_code is False

    def test_public_methods_route_through_the_clip_backend(self, images):
        embedder = ImageEmbedder(backend="clip", model_name="fake/clip")
        embedder._backend = _backend()
        assert embedder.embed_images([images["red"], images["blue"]]).shape == (2, DIM)
        assert embedder.embed_texts(["a", "b", "c"], batch_size=2).shape == (3, DIM)
        assert embedder.embed_multimodal([{"image": images["red"], "text": "x"}]).shape == (1, DIM)


class TestLoading:
    def test_non_dual_encoder_is_rejected(self, monkeypatch):
        pytest.importorskip("torch")
        module = types.ModuleType("transformers")

        class AutoProcessor:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return FakeClipProcessor()

        class AutoModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return types.SimpleNamespace(config=types.SimpleNamespace(model_type="bert"))

        module.AutoProcessor = AutoProcessor
        module.AutoModel = AutoModel
        monkeypatch.setitem(sys.modules, "transformers", module)
        with pytest.raises(ValueError, match="not a dual image/text encoder"):
            ClipEmbeddingBackend("fake/bert").load_model()


# ---------------------------------------------------------------------------
# VectorDB contract
# ---------------------------------------------------------------------------
def _clip_embedder():
    embedder = ImageEmbedder(backend="clip", model_name="fake/clip")
    embedder._backend = _backend()
    return embedder


def _frame(images):
    import geopandas as gpd
    from shapely.geometry import Point

    names = list(COLORS)
    return gpd.GeoDataFrame(
        {
            "image_id": names,
            "scene_narrative": [f"a {n} scene" for n in names],
        },
        geometry=[Point(28.9 + i * 0.001, 41.0) for i in range(len(names))],
        crs="EPSG:4326",
    )


def _db(store, tmp_path, embedder):
    from geoai_vlm.vectorstore import VectorDB

    if store == "chromadb":
        pytest.importorskip("chromadb")
        return VectorDB(
            embedder=embedder,
            store_backend="chromadb",
            persist_directory=str(tmp_path / "clip_store"),
            collection_name="clip_store",
        )
    pytest.importorskip("faiss")
    return VectorDB(embedder=embedder, store_backend="faiss", metric="ip")


class TestVectorDBContract:
    @pytest.mark.parametrize("store", ["faiss", "chromadb"])
    def test_an_image_finds_itself_first_with_similarity_one(self, store, tmp_path, images):
        db = _db(store, tmp_path, _clip_embedder())
        gdf = _frame(images)
        db.build(gdf, text_column=None, image_dir=str(tmp_path))
        result = db.search(query_image=images["green"], n_results=4)
        assert result.iloc[0]["id"] == "green"
        assert result.iloc[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
        assert list(result["similarity"]) == sorted(result["similarity"], reverse=True)

    def test_inner_product_and_cosine_rank_identically(self, tmp_path, images):
        pytest.importorskip("faiss")
        pytest.importorskip("chromadb")
        gdf = _frame(images)
        rankings, sims = [], []
        for store in ("faiss", "chromadb"):
            db = _db(store, tmp_path / store, _clip_embedder())
            db.build(gdf, text_column="scene_narrative", image_dir=str(tmp_path))
            res = db.search(query_text="a blue scene", n_results=4)
            rankings.append(list(res["id"]))
            sims.append(np.array(res["similarity"], dtype=float))
        assert rankings[0] == rankings[1]
        assert np.allclose(sims[0], sims[1], atol=1e-5)

    def test_similarity_equals_the_dot_product_of_unit_vectors(self, tmp_path, images):
        pytest.importorskip("faiss")
        embedder = _clip_embedder()
        db = _db("faiss", tmp_path, embedder)
        db.build(_frame(images), text_column=None, image_dir=str(tmp_path))
        res = db.search(query_text="kerb", n_results=4)
        query = _expected_text("kerb")
        for _, row in res.iterrows():
            expected = float(query @ _expected_image(COLORS[row["id"]]))
            assert row["similarity"] == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("batch_size", [1, 3, 32])
    def test_index_content_is_independent_of_batch_size(self, tmp_path, images, batch_size):
        pytest.importorskip("faiss")
        db = _db("faiss", tmp_path, _clip_embedder())
        db.build(_frame(images), text_column="scene_narrative", image_dir=str(tmp_path), batch_size=batch_size)
        res = db.search(query_image=images["red"], n_results=4)
        reference = _db("faiss", tmp_path / "ref", _clip_embedder())
        reference.build(_frame(images), text_column="scene_narrative", image_dir=str(tmp_path), batch_size=4)
        ref = reference.search(query_image=images["red"], n_results=4)
        assert list(res["id"]) == list(ref["id"])
        assert np.allclose(res["similarity"], ref["similarity"], atol=1e-6)

    def test_text_image_and_fused_vectors_fit_one_index(self, tmp_path, images):
        pytest.importorskip("faiss")
        from geoai_vlm.vectorstore import FAISSVectorStore

        backend = _backend()
        vecs = backend.embed(
            [{"text": "kerb"}, {"image": images["red"]}, {"image": images["blue"], "text": "x"}]
        )
        store = FAISSVectorStore(metric="ip")
        store.add(ids=["t", "i", "f"], embeddings=vecs)
        assert store.count() == 3
        result = store.query(vecs[1], n_results=3)
        assert result["ids"][0] == "i"
