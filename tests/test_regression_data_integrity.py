# -*- coding: utf-8 -*-
"""
Phase 0 regression tests: data integrity.

Every test in this module asserts the *fixed* behaviour. Each one corresponds
to a finding from the 2026-09-12 archive review (commit 7f880f70). These are
software-behaviour tests: they never download a model, never contact a network
service and never require an API key.

Finding map
-----------
R1  describe_query ignored a custom output_path on the normal export branch.
R2  describe_query described every image on disk, not the selected subset.
R3  parse_json_response accepted non-object JSON and crashed downstream.
R4  the "simple" prompt schema never reached the summary columns.
R5  a missing image_quality block was silently treated as usable=True.
R6  build_embedding_text raised ValueError on list-valued cells.
R7  merge_metadata_and_descriptions inflated row counts on duplicate ids.
R8  resume was keyed on image id alone, ignoring model/prompt identity.
R9  embed_place promised a joint representation but embedded text only.
R10 importing the package required cv2/torch even for pure data work.
R11 the test embedder was content-independent and batch-size-dependent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeBackend:
    """Minimal VLM backend stub: returns canned responses, records calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_paths = []

    def load_model(self):
        pass

    def is_available(self):
        return True

    def generate(self, image_paths, system_prompt, user_prompt):
        self.seen_paths.extend(image_paths)
        out = []
        for _ in image_paths:
            out.append(self._responses.pop(0) if self._responses else "{}")
        return out


def _make_describer(responses, **kwargs):
    from geoai_vlm.describer import ImageDescriber

    d = ImageDescriber(**kwargs)
    d._backend = _FakeBackend(responses)
    return d


def _write_images(directory: Path, ids) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for img_id in ids:
        p = directory / f"{img_id}.jpg"
        Image.new("RGB", (16, 16), color="gray").save(p)
        paths.append(p)
    return paths


GEOAI_RESPONSE = json.dumps(
    {
        "scene_narrative": "A narrow residential street with parked cars.",
        "semantic_tags": ["residential", "narrow"],
        "land_use_character": {"primary": "residential"},
        "urban_morphology": {"street_type": "local"},
        "place_character": {"dominant_activity": "quiet"},
        "image_quality": {"usable_for_analysis": True},
    }
)


# ---------------------------------------------------------------------------
# R1 / R2 — pipeline output contract
# ---------------------------------------------------------------------------
class TestDescribeQueryOutputContract:
    """describe_query must honour output_path and the selected image set."""

    @pytest.fixture
    def patched_pipeline(self, monkeypatch, tmp_path):
        """Stub the downloader and describer so no network/model is needed."""
        import geopandas as gpd
        from shapely.geometry import Point

        from geoai_vlm import pipeline as pl

        img_dir = tmp_path / "work"
        selected = ["100", "200"]
        _write_images(img_dir, selected + ["999_not_selected"])

        meta = gpd.GeoDataFrame(
            {
                "image_id": selected,
                "image_path": [str(img_dir / f"{i}.jpg") for i in selected],
            },
            geometry=[Point(28.97, 41.0), Point(28.98, 41.01)],
            crs="EPSG:4326",
        )

        class _StubDownloader:
            def __init__(self, *a, **k):
                pass

            def download(self, *a, **k):
                return meta

        monkeypatch.setattr(pl, "MapillaryDownloader", _StubDownloader)

        captured = {}

        class _StubDescriber:
            def __init__(self, *a, **k):
                pass

            def describe(self, image_dir=None, image_paths=None, **k):
                captured["image_dir"] = image_dir
                captured["image_paths"] = image_paths
                if image_paths is not None:
                    ids = [Path(p).stem for p in image_paths]
                else:  # legacy behaviour: glob the whole directory
                    ids = sorted(p.stem for p in Path(image_dir).glob("*.jpg"))
                return pd.DataFrame(
                    {"image_id": ids, "scene_narrative": ["x"] * len(ids)}
                )

        monkeypatch.setattr(pl, "ImageDescriber", _StubDescriber)
        return pl, img_dir, captured

    def test_custom_output_path_is_actually_written(self, patched_pipeline, tmp_path):
        """R1: the file named in output_path must exist after the run."""
        pl, img_dir, _ = patched_pipeline
        custom = tmp_path / "custom" / "fatih_run.parquet"

        pl.describe_query(
            query=object(),
            mly_api_key="dummy",
            output_dir=img_dir,
            output_path=custom,
            verbosity=0,
        )

        assert custom.exists(), f"custom output path was not written: {custom}"

    def test_default_output_path_still_written(self, patched_pipeline, tmp_path):
        """The historical default (output_dir/results.parquet) must keep working."""
        pl, img_dir, _ = patched_pipeline
        pl.describe_query(
            query=object(), mly_api_key="dummy", output_dir=img_dir, verbosity=0
        )
        assert (img_dir / "results.parquet").exists()

    def test_only_selected_images_are_described(self, patched_pipeline, tmp_path):
        """R2: an unrelated image in the work dir must not be described."""
        pl, img_dir, captured = patched_pipeline

        result = pl.describe_query(
            query=object(),
            mly_api_key="dummy",
            output_dir=img_dir,
            output_path=tmp_path / "out.parquet",
            verbosity=0,
        )

        assert captured["image_paths"] is not None, (
            "describe_query must pass the selected image paths to the describer"
        )
        described = {Path(p).stem for p in captured["image_paths"]}
        assert described == {"100", "200"}
        assert "999_not_selected" not in described
        assert len(result) == 2


# ---------------------------------------------------------------------------
# R3 — invalid model responses become controlled errors
# ---------------------------------------------------------------------------
class TestInvalidModelResponses:
    """Non-object JSON must degrade to a recorded error, never an exception."""

    @pytest.mark.parametrize(
        "payload",
        ['[{"scene_narrative": "a"}]', "null", "42", '"a bare string"', "[]"],
        ids=["list", "null", "number", "string", "empty-list"],
    )
    def test_parse_json_response_rejects_non_object(self, payload):
        from geoai_vlm.describer import parse_json_response

        parsed = parse_json_response(payload)
        assert isinstance(parsed, dict), "parser must always return a dict"
        assert "error" in parsed, f"non-object JSON must be an error: {payload!r}"
        assert parsed.get("raw_response") == payload

    def test_parse_json_response_still_accepts_objects(self):
        from geoai_vlm.describer import parse_json_response

        parsed = parse_json_response('{"scene_narrative": "ok"}')
        assert parsed == {"scene_narrative": "ok"}
        assert "error" not in parsed

    def test_parse_json_response_handles_markdown_fence(self):
        from geoai_vlm.describer import parse_json_response

        parsed = parse_json_response('```json\n{"scene_narrative": "ok"}\n```')
        assert parsed.get("scene_narrative") == "ok"

    def test_describe_records_failure_instead_of_raising(self, tmp_path):
        """A JSON list from the model must not abort the batch."""
        img_dir = tmp_path / "img"
        paths = _write_images(img_dir, ["a", "b"])

        describer = _make_describer(['[{"scene_narrative": "x"}]', GEOAI_RESPONSE])
        df = describer.describe(image_paths=paths, batch_size=2)

        assert len(df) == 2
        row_a = df[df["image_id"] == "a"].iloc[0]
        assert bool(row_a["parse_error"]) is True
        assert row_a["raw_response"] == '[{"scene_narrative": "x"}]'
        # A failed record must never look like a successful one.
        assert pd.isna(row_a["usable"])
        assert row_a["quality_status"] == "error"

    def test_failed_record_is_not_counted_usable(self, tmp_path):
        img_dir = tmp_path / "img2"
        paths = _write_images(img_dir, ["bad"])
        describer = _make_describer(["this is not json at all"])
        df = describer.describe(image_paths=paths)
        assert bool(df.iloc[0]["parse_error"]) is True
        assert pd.isna(df.iloc[0]["usable"])


# ---------------------------------------------------------------------------
# R4 / R5 — prompt schema and quality provenance
# ---------------------------------------------------------------------------
class TestPromptSchemaAndQuality:
    def test_simple_schema_populates_summary_columns(self, tmp_path):
        """R4: the simple template's description/tags must reach the summary columns."""
        paths = _write_images(tmp_path / "s", ["s1"])
        response = json.dumps(
            {"description": "A quiet lane.", "tags": ["quiet", "lane"]}
        )
        describer = _make_describer([response], prompt_template="simple")
        df = describer.describe(image_paths=paths)

        row = df.iloc[0]
        assert bool(row["parse_error"]) is False
        assert row["scene_narrative"] == "A quiet lane."
        assert row["semantic_tags"] == "quiet,lane"

    def test_missing_quality_block_is_unknown_not_usable(self, tmp_path):
        """R5: absent quality info must not be recorded as usable=True."""
        paths = _write_images(tmp_path / "q", ["q1"])
        response = json.dumps({"description": "A lane.", "tags": ["lane"]})
        describer = _make_describer([response], prompt_template="simple")
        df = describer.describe(image_paths=paths)

        row = df.iloc[0]
        assert pd.isna(row["usable"]), "unknown quality must not become usable=True"
        assert row["quality_status"] == "unknown"

    def test_reported_quality_is_preserved(self, tmp_path):
        paths = _write_images(tmp_path / "q2", ["q2"])
        describer = _make_describer([GEOAI_RESPONSE])
        df = describer.describe(image_paths=paths)
        row = df.iloc[0]
        assert pd.notna(row["usable"]) and bool(row["usable"]) is True
        assert row["quality_status"] == "reported"

    def test_reported_unusable_is_preserved(self, tmp_path):
        paths = _write_images(tmp_path / "q3", ["q3"])
        response = json.dumps(
            {
                "scene_narrative": "Blurred.",
                "image_quality": {"usable_for_analysis": False},
            }
        )
        describer = _make_describer([response])
        df = describer.describe(image_paths=paths)
        row = df.iloc[0]
        assert pd.notna(row["usable"]) and bool(row["usable"]) is False
        assert row["quality_status"] == "reported"

    def test_error_quality_status_is_distinct(self, tmp_path):
        """Unknown, unusable and errored must be three distinguishable states."""
        paths = _write_images(tmp_path / "q4", ["q4"])
        describer = _make_describer(["not json"])
        df = describer.describe(image_paths=paths)
        assert df.iloc[0]["quality_status"] == "error"


# ---------------------------------------------------------------------------
# R6 — list-valued cells in embedding text
# ---------------------------------------------------------------------------
class TestBuildEmbeddingText:
    def test_multi_element_list_is_joined(self):
        from geoai_vlm.preparation import build_embedding_text

        df = pd.DataFrame(
            {"tags": [["historic", "narrow", "quiet"]], "narrative": ["A street."]}
        )
        out = build_embedding_text(df, ["narrative", "tags"])
        assert out.iloc[0] == "A street. | historic, narrow, quiet"

    def test_single_element_list_is_joined(self):
        from geoai_vlm.preparation import build_embedding_text

        df = pd.DataFrame({"tags": [["historic"]], "narrative": ["A street."]})
        assert build_embedding_text(df, ["narrative", "tags"]).iloc[0] == (
            "A street. | historic"
        )

    def test_empty_list_is_skipped(self):
        from geoai_vlm.preparation import build_embedding_text

        df = pd.DataFrame({"tags": [[]], "narrative": ["A street."]})
        assert build_embedding_text(df, ["narrative", "tags"]).iloc[0] == "A street."

    def test_numpy_array_cell_is_joined(self):
        from geoai_vlm.preparation import build_embedding_text

        df = pd.DataFrame({"narrative": ["A street."]})
        df["tags"] = [np.array(["a", "b"], dtype=object)]
        assert build_embedding_text(df, ["narrative", "tags"]).iloc[0] == (
            "A street. | a, b"
        )

    def test_nan_is_still_skipped(self):
        from geoai_vlm.preparation import build_embedding_text

        df = pd.DataFrame({"a": ["x"], "b": [np.nan]})
        assert build_embedding_text(df, ["a", "b"]).iloc[0] == "x"


# ---------------------------------------------------------------------------
# R7 — duplicate ids must not inflate the sample
# ---------------------------------------------------------------------------
class TestDuplicateIdMerge:
    @pytest.fixture
    def meta(self):
        import geopandas as gpd
        from shapely.geometry import Point

        return gpd.GeoDataFrame(
            {"image_id": ["1", "2"]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )

    def test_duplicate_description_ids_raise_by_default(self, meta):
        from geoai_vlm.io import merge_metadata_and_descriptions

        desc = pd.DataFrame(
            {"image_id": ["1", "1", "2"], "scene_narrative": ["a", "b", "c"]}
        )
        with pytest.raises(ValueError, match="duplicate"):
            merge_metadata_and_descriptions(meta, desc)

    def test_duplicate_ids_can_be_resolved_to_first(self, meta):
        from geoai_vlm.io import merge_metadata_and_descriptions

        desc = pd.DataFrame(
            {"image_id": ["1", "1", "2"], "scene_narrative": ["a", "b", "c"]}
        )
        merged = merge_metadata_and_descriptions(meta, desc, on_duplicate="first")
        assert len(merged) == 2
        assert merged.set_index("image_id").loc["1", "scene_narrative"] == "a"

    def test_unique_ids_are_unaffected(self, meta):
        from geoai_vlm.io import merge_metadata_and_descriptions

        desc = pd.DataFrame({"image_id": ["1", "2"], "scene_narrative": ["a", "b"]})
        merged = merge_metadata_and_descriptions(meta, desc)
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# R8 — resume must be keyed on the processing identity
# ---------------------------------------------------------------------------
class TestResumeBehaviour:
    def test_resume_does_not_duplicate_successful_records(self, tmp_path):
        """R8: re-running with the same model/prompt must not re-add rows."""
        paths = _write_images(tmp_path / "r", ["r1", "r2"])
        out = tmp_path / "desc.parquet"

        d1 = _make_describer([GEOAI_RESPONSE, GEOAI_RESPONSE])
        first = d1.describe(image_paths=paths, output_path=out, resume=True)
        assert len(first) == 2

        d2 = _make_describer([GEOAI_RESPONSE, GEOAI_RESPONSE])
        second = d2.describe(image_paths=paths, output_path=out, resume=True)

        assert len(second) == 2, "resume duplicated already-successful records"
        assert second["image_id"].duplicated().sum() == 0
        assert d2._backend.seen_paths == [], "resume re-described finished images"

    def test_changed_prompt_is_reprocessed(self, tmp_path):
        """A different prompt is a different derived output, not a finished one."""
        paths = _write_images(tmp_path / "r2", ["r1"])
        out = tmp_path / "desc2.parquet"

        d1 = _make_describer([GEOAI_RESPONSE], prompt_template="geoai")
        d1.describe(image_paths=paths, output_path=out, resume=True)

        d2 = _make_describer([GEOAI_RESPONSE], prompt_template="simple")
        d2.describe(image_paths=paths, output_path=out, resume=True)

        assert d2._backend.seen_paths, "prompt change must trigger reprocessing"
        stored = pd.read_parquet(out)
        assert stored["processing_id"].nunique() == 2

    def test_processing_identity_columns_are_recorded(self, tmp_path):
        paths = _write_images(tmp_path / "r3", ["r1"])
        d = _make_describer([GEOAI_RESPONSE], model_name="test/model-x")
        df = d.describe(image_paths=paths)
        for col in ("model_name", "prompt_version", "processing_id"):
            assert col in df.columns, f"missing provenance column: {col}"
        assert df.iloc[0]["model_name"] == "test/model-x"

    def test_failed_records_are_retried_on_resume(self, tmp_path):
        """A parse failure must not be permanently skipped as if it succeeded."""
        paths = _write_images(tmp_path / "r4", ["r1"])
        out = tmp_path / "desc3.parquet"

        d1 = _make_describer(["not json"])
        d1.describe(image_paths=paths, output_path=out, resume=True)

        d2 = _make_describer([GEOAI_RESPONSE])
        d2.describe(image_paths=paths, output_path=out, resume=True)

        assert d2._backend.seen_paths, "failed record was skipped instead of retried"
        stored = pd.read_parquet(out)
        assert len(stored[stored["image_id"] == "r1"]) == 1
        assert bool(stored.iloc[0]["parse_error"]) is False


# ---------------------------------------------------------------------------
# R9 — embedding modality must be explicit
# ---------------------------------------------------------------------------
class TestEmbeddingModality:
    @pytest.fixture
    def stub_place(self, monkeypatch, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Point

        from geoai_vlm import embedding as emb_mod
        from geoai_vlm import pipeline as pl

        img_dir = tmp_path / "imgs"
        _write_images(img_dir, ["1", "2"])
        gdf = gpd.GeoDataFrame(
            {
                "image_id": ["1", "2"],
                "scene_narrative": ["a", "b"],
                "image_path": [str(img_dir / "1.jpg"), str(img_dir / "2.jpg")],
            },
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )
        monkeypatch.setattr(pl, "describe_place", lambda **k: gdf.copy())

        calls = []

        class _StubEmbedder:
            def __init__(self, *a, **k):
                pass

            def embed_texts(self, texts, **k):
                calls.append(("text", list(texts)))
                return np.ones((len(texts), 4), dtype=np.float32)

            def embed_images(self, paths, **k):
                calls.append(("image", list(paths)))
                return np.ones((len(list(paths)), 4), dtype=np.float32)

            def embed_multimodal(self, inputs, **k):
                calls.append(("multimodal", list(inputs)))
                return np.ones((len(list(inputs)), 4), dtype=np.float32)

        monkeypatch.setattr(emb_mod, "ImageEmbedder", _StubEmbedder)
        return pl, calls, gdf

    def test_default_uses_joint_representation(self, stub_place):
        """R9: the documented joint representation must actually use the image."""
        pl, calls, _ = stub_place
        out = pl.embed_place(place_name="X", mly_api_key="k")
        assert calls, "no embedding call was made"
        assert calls[0][0] == "multimodal", (
            f"expected a joint image+text embedding, got {calls[0][0]!r}"
        )
        assert set(out["embedding_modality"]) == {"multimodal"}

    def test_text_only_is_available_explicitly(self, stub_place):
        pl, calls, _ = stub_place
        out = pl.embed_place(
            place_name="X", mly_api_key="k", embedding_modality="text"
        )
        assert calls[0][0] == "text"
        assert set(out["embedding_modality"]) == {"text"}

    def test_image_only_is_available_explicitly(self, stub_place):
        pl, calls, _ = stub_place
        pl.embed_place(place_name="X", mly_api_key="k", embedding_modality="image")
        assert calls[0][0] == "image"

    def test_missing_image_does_not_silently_fall_back_to_text(
        self, stub_place, monkeypatch
    ):
        """A missing file must be reported, not quietly downgraded."""
        pl, calls, gdf = stub_place
        broken = gdf.copy()
        broken["image_path"] = ["/nonexistent/a.jpg", "/nonexistent/b.jpg"]
        monkeypatch.setattr(pl, "describe_place", lambda **k: broken.copy())

        with pytest.raises(FileNotFoundError):
            pl.embed_place(place_name="X", mly_api_key="k")

    def test_missing_image_skip_is_recorded(self, stub_place, monkeypatch):
        pl, calls, gdf = stub_place
        partial = gdf.copy()
        partial.loc[partial.index[1], "image_path"] = "/nonexistent/b.jpg"
        monkeypatch.setattr(pl, "describe_place", lambda **k: partial.copy())

        out = pl.embed_place(
            place_name="X", mly_api_key="k", on_missing_image="skip"
        )
        modalities = set(out["embedding_modality"])
        assert "multimodal" in modalities
        assert "missing_image" in modalities
        assert out.loc[out["image_id"] == "2", "embedding"].isna().all()


# ---------------------------------------------------------------------------
# R10 — core data work must not require the heavy vision stack
# ---------------------------------------------------------------------------
class TestOptionalHeavyDependencies:
    def test_package_imports_without_cv2_or_torch(self):
        """R10: `import geoai_vlm` must work with only the core data stack."""
        # Make the heavy modules genuinely unimportable, rather than stubbing
        # them in sys.modules (a None entry confuses libraries that probe for
        # torch). This keeps the test meaningful on machines that do have them.
        code = (
            "import sys\n"
            "BLOCKED = {'cv2', 'torch', 'transformers', 'zensvi', 'skimage'}\n"
            "class _Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        return None\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ModuleNotFoundError(f'blocked: {name}')\n"
            "        return None\n"
            "for _m in list(sys.modules):\n"
            "    if _m.split('.')[0] in BLOCKED:\n"
            "        del sys.modules[_m]\n"
            "sys.meta_path.insert(0, _Blocker())\n"
            "import geoai_vlm\n"
            "from geoai_vlm import describe_place, build_embedding_text\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, (
            f"core import needs the heavy stack:\n{proc.stderr[-2000:]}"
        )
        assert "OK" in proc.stdout

    def test_vision2slope_attribute_still_reachable(self):
        """The lazy path must not remove the public name from the API."""
        import geoai_vlm

        assert "Vision2SlopePipeline" in geoai_vlm.__all__


# ---------------------------------------------------------------------------
# R11 — the test embedder must model a real embedder's contract
# ---------------------------------------------------------------------------
class TestMockEmbedderContract:
    def test_embedding_depends_on_content(self, mock_embedder):
        a = mock_embedder.embed_texts(["a quiet residential street"])
        b = mock_embedder.embed_texts(["a busy arterial road"])
        assert not np.allclose(a, b), "different text produced identical vectors"

    def test_embedding_is_batch_size_independent(self, mock_embedder):
        texts = [f"street number {i}" for i in range(7)]
        one = mock_embedder.embed_texts(texts, batch_size=1)
        many = mock_embedder.embed_texts(texts, batch_size=4)
        assert np.allclose(one, many, atol=1e-6), (
            "batch size changed the embedding of identical input"
        )

    def test_embedding_is_deterministic(self, mock_embedder):
        a = mock_embedder.embed_texts(["same text"])
        b = mock_embedder.embed_texts(["same text"])
        assert np.allclose(a, b)

    def test_embeddings_are_l2_normalised(self, mock_embedder):
        emb = mock_embedder.embed_texts(["x", "y", "z"])
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
