# -*- coding: utf-8 -*-
"""
Provenance fields and processing_id sensitivity (P0-A item 6).

``processing_id`` must change with anything that can change a model's text
(model, revision, prompt, output-affecting generation settings) and must not
change with execution detail (batch size, device, concurrency). Resume and
upsert key on it, so both directions are tested through ``describe``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from PIL import Image

from geoai_vlm.chat import GenerationOutput
from geoai_vlm.describer import (
    DESCRIPTION_COLUMNS,
    BaseBackend,
    ImageDescriber,
    TransformersBackend,
    VLLMBackend,
)
from geoai_vlm.provenance import (
    canonical_generation_params,
    compute_processing_id,
    resolve_model_revision,
)

SHA_A = "a" * 40
SHA_B = "b" * 40
RESPONSE = json.dumps({"description": "A lane.", "tags": ["lane"]})


class RecordingBackend(BaseBackend):
    """A custom backend that reports provenance and records calls."""

    name = "recording"

    def __init__(self, revision=SHA_A, temperature=0.0, max_new_tokens=128, responses=None):
        self.revision = revision
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.responses = list(responses or [])
        self.seen = []

    def load_model(self):
        pass

    def is_available(self):
        return True

    def generate(self, image_paths, system_prompt, user_prompt):
        self.seen.extend(image_paths)
        return [self.responses.pop(0) if self.responses else RESPONSE for _ in image_paths]

    def generate_outputs(self, images, system_prompt, user_prompt):
        texts = self.generate([str(i) for i in images], system_prompt, user_prompt)
        return [
            GenerationOutput(text=t, decoding_mode="unconstrained", system_prompt_mode="system")
            for t in texts
        ]

    def provenance(self):
        return {
            "backend": self.name,
            "backend_version": "recording 1.0",
            "model_revision": self.revision,
            "generation_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "device_map": "cuda:0",  # execution detail; must be ignored
            },
        }


def _images(tmp_path, ids):
    paths = []
    for i in ids:
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), "gray").save(p)
        paths.append(p)
    return paths


def _describer(backend, **kwargs):
    kwargs.setdefault("prompt_template", "simple")
    return ImageDescriber(model_name="org/model", backend=backend, **kwargs)


# ---------------------------------------------------------------------------
# compute_processing_id
# ---------------------------------------------------------------------------
class TestProcessingIdSensitivity:
    BASE = dict(
        model_name="org/model",
        prompt_version="p1",
        model_revision=SHA_A,
        generation_params={"temperature": 0.0, "max_new_tokens": 256, "dtype": "auto"},
    )

    def _pid(self, **changes):
        args = dict(self.BASE)
        params = dict(args.pop("generation_params"))
        params.update(changes.pop("params", {}))
        args.update(changes)
        return compute_processing_id(generation_params=params, **args)

    @pytest.mark.parametrize(
        "change",
        [
            {"model_name": "org/other"},
            {"model_revision": SHA_B},
            {"model_revision": None},
            {"prompt_version": "p2"},
            {"params": {"temperature": 0.7}},
            {"params": {"max_new_tokens": 512}},
            {"params": {"dtype": "float32"}},
            {"params": {"quantization": "4bit"}},
            {"params": {"structured_output": True}},
            {"params": {"system_prompt_mode": "prepend"}},
            {"params": {"json_schema_digest": "abc"}},
            {"params": {"image_max_side": 768}},
            {"params": {"repetition_penalty": 1.1}},
            {"backend": "vllm"},
            {"endpoint": "http://other-server:8000/v1"},
        ],
        ids=lambda c: next(iter(c.get("params", c))),
    )
    def test_output_affecting_changes_change_the_id(self, change):
        assert self._pid(**change) != self._pid()

    @pytest.mark.parametrize(
        "params",
        [
            {"batch_size": 32},
            {"device_map": "cpu"},
            {"max_concurrency": 16},
            {"timeout": 5},
            {"gpu_memory_utilization": 0.5},
        ],
        ids=lambda p: next(iter(p)),
    )
    def test_execution_details_do_not_change_the_id(self, params):
        assert self._pid(params=params) == self._pid()

    def test_sampling_only_settings_are_ignored_under_greedy_decoding(self):
        assert self._pid(params={"seed": 42, "top_p": 0.9}) == self._pid()

    def test_sampling_settings_count_when_sampling(self):
        sampled = {"temperature": 0.7}
        assert self._pid(params={**sampled, "seed": 1}) != self._pid(params={**sampled, "seed": 2})
        assert self._pid(params={**sampled, "top_p": 0.9}) != self._pid(params=sampled)

    def test_float_noise_does_not_change_the_id(self):
        assert self._pid(params={"temperature": 0.7}) == self._pid(params={"temperature": 0.7000000001})

    def test_id_is_short_and_stable(self):
        pid = self._pid()
        assert len(pid) == 12 and pid == self._pid()

    def test_canonical_params_drop_none_and_unknown_keys(self):
        assert canonical_generation_params(
            {"temperature": 0.5, "top_p": None, "device_map": "auto", "batch_size": 8}
        ) == {"temperature": 0.5}


# ---------------------------------------------------------------------------
# Through the describer
# ---------------------------------------------------------------------------
class TestDescriberProvenance:
    def test_records_carry_every_provenance_column(self, tmp_path):
        backend = RecordingBackend()
        df = _describer(backend).describe(image_paths=_images(tmp_path, ["a"]))
        for column in DESCRIPTION_COLUMNS:
            assert column in df.columns, column
        row = df.iloc[0]
        assert row["backend"] == "recording"
        assert row["backend_version"] == "recording 1.0"
        assert row["model_revision"] == SHA_A
        params = json.loads(row["generation_params"])
        assert params == {
            "max_new_tokens": 128,
            "system_prompt_mode": "auto",
            "temperature": 0.0,
        }
        assert row["system_prompt_mode_effective"] == "system"
        assert row["decoding_mode"] == "unconstrained"
        assert row["generation_error"] is None

    def test_processing_id_matches_the_recorded_fields(self, tmp_path):
        backend = RecordingBackend()
        d = _describer(backend)
        row = d.describe(image_paths=_images(tmp_path, ["a"])).iloc[0]
        expected = compute_processing_id(
            "org/model", d.prompt_version, row["model_revision"], json.loads(row["generation_params"]),
            backend=row["backend"], endpoint=row["backend_endpoint"],
        )
        assert row["processing_id"] == expected == d.processing_id

    def test_the_engine_is_part_of_the_configuration(self):
        class OtherEngine(RecordingBackend):
            name = "other-engine"

        # Same model, revision, prompt and settings, different engine.
        assert _describer(RecordingBackend()).processing_id != _describer(OtherEngine()).processing_id

    def test_http_servers_are_told_apart_by_endpoint_without_credentials(self):
        from geoai_vlm.openai_compat import OpenAICompatibleBackend

        def run(url):
            return _describer(OpenAICompatibleBackend("served-name", base_url=url)).provenance()

        a, b = run("http://server-a:8000/v1"), run("http://server-b:8000/v1")
        # A served model name is only a label: two servers may hold different weights.
        assert a["processing_id"] != b["processing_id"]
        with_credentials = run("http://user:s3cret-pass@server-a:8000/v1")
        assert with_credentials["backend_endpoint"] == "http://server-a:8000/v1"
        assert with_credentials["processing_id"] == a["processing_id"]
        assert "s3cret" not in json.dumps(with_credentials)

    def test_existing_columns_keep_their_order(self):
        assert DESCRIPTION_COLUMNS[:16] == (
            "image_path", "image_id", "raw_response", "parsed_json", "parse_error",
            "model_name", "prompt_version", "processing_id", "processed_at",
            "scene_narrative", "semantic_tags", "land_use_primary", "street_type",
            "place_character", "usable", "quality_status",
        )

    @pytest.mark.parametrize(
        "second",
        [
            {"backend": dict(revision=SHA_B)},
            {"backend": dict(temperature=0.7)},
            {"backend": dict(max_new_tokens=64)},
            {"describer": dict(structured_output=True)},
            {"describer": dict(system_prompt_mode="prepend")},
        ],
        ids=["revision", "temperature", "max_new_tokens", "structured", "system_mode"],
    )
    def test_changed_configuration_is_reprocessed_not_skipped(self, tmp_path, second):
        paths = _images(tmp_path, ["a", "b"])
        out = tmp_path / "desc.parquet"
        _describer(RecordingBackend()).describe(image_paths=paths, output_path=out)

        backend = RecordingBackend(**second.get("backend", {}))
        _describer(backend, **second.get("describer", {})).describe(image_paths=paths, output_path=out)

        assert len(backend.seen) == 2, "a changed configuration must not be resumed as done"
        stored = pd.read_parquet(out)
        assert stored["processing_id"].nunique() == 2
        assert len(stored) == 4

    def test_same_configuration_resumes(self, tmp_path):
        paths = _images(tmp_path, ["a", "b"])
        out = tmp_path / "desc.parquet"
        _describer(RecordingBackend()).describe(image_paths=paths, output_path=out)
        backend = RecordingBackend()
        df = _describer(backend).describe(image_paths=paths, output_path=out)
        assert backend.seen == []
        assert len(df) == 2

    def test_upsert_replaces_failed_records_of_the_same_configuration(self, tmp_path):
        paths = _images(tmp_path, ["a"])
        out = tmp_path / "desc.parquet"
        _describer(RecordingBackend(responses=["{not json"])).describe(image_paths=paths, output_path=out)
        _describer(RecordingBackend()).describe(image_paths=paths, output_path=out)
        stored = pd.read_parquet(out)
        assert len(stored) == 1
        assert bool(stored.iloc[0]["parse_error"]) is False

    def test_other_configurations_are_reported(self, tmp_path, capsys):
        paths = _images(tmp_path, ["a"])
        out = tmp_path / "desc.parquet"
        _describer(RecordingBackend(temperature=0.5)).describe(image_paths=paths, output_path=out)
        capsys.readouterr()
        _describer(RecordingBackend()).describe(image_paths=paths, output_path=out)
        assert "different revision or generation settings" in capsys.readouterr().out

    def test_custom_backend_without_provenance_still_works(self, tmp_path):
        class Minimal:
            def generate(self, image_paths, system_prompt, user_prompt):
                return [RESPONSE] * len(image_paths)

        d = ImageDescriber(model_name="org/model", prompt_template="simple")
        d._backend = Minimal()
        row = d.describe(image_paths=_images(tmp_path, ["a"])).iloc[0]
        assert row["backend"] == "Minimal"
        assert row["model_revision"] is None
        assert row["decoding_mode"] == "unconstrained"
        assert row["system_prompt_mode_effective"] is None


# ---------------------------------------------------------------------------
# Malformed output and generation errors are records, not exceptions
# ---------------------------------------------------------------------------
class TestFailureRecords:
    @pytest.mark.parametrize(
        "raw",
        ['{"description": "cut off', "```json\n{oops}\n```", "", "Sure! Here it is:"],
        ids=["truncated", "fenced-invalid", "empty", "prose"],
    )
    def test_malformed_json_is_a_failed_record(self, tmp_path, raw):
        d = _describer(RecordingBackend(responses=[raw]))
        row = d.describe(image_paths=_images(tmp_path, ["a"])).iloc[0]
        assert bool(row["parse_error"]) is True
        assert row["raw_response"] == raw
        assert row["quality_status"] == "error"
        assert pd.isna(row["usable"])
        assert row["validation_issues"] is None

    def test_generation_error_is_recorded_and_retried(self, tmp_path):
        class Failing(RecordingBackend):
            def generate_outputs(self, images, system_prompt, user_prompt):
                return [GenerationOutput(text="", error="HTTP 503 from server")] * len(images)

        paths = _images(tmp_path, ["a"])
        out = tmp_path / "d.parquet"
        row = _describer(Failing()).describe(image_paths=paths, output_path=out).iloc[0]
        assert bool(row["parse_error"]) is True
        assert row["generation_error"] == "HTTP 503 from server"
        assert json.loads(row["parsed_json"])["error"].startswith("generation failed")

        retry = RecordingBackend()
        _describer(retry).describe(image_paths=paths, output_path=out)
        assert retry.seen, "a generation failure must be retried on resume"

    def test_schema_violations_are_reported_not_hidden(self, tmp_path):
        bad = json.dumps({"description": "x", "tags": "not-a-list", "extra": 1})
        row = _describer(RecordingBackend(responses=[bad])).describe(
            image_paths=_images(tmp_path, ["a"])
        ).iloc[0]
        issues = json.loads(row["validation_issues"])
        assert any("tags" in i for i in issues)
        assert any("unexpected field 'extra'" in i for i in issues)
        assert bool(row["parse_error"]) is False, "validation does not discard the response"

    def test_backend_returning_wrong_count_raises(self, tmp_path):
        class Short(RecordingBackend):
            def generate_outputs(self, images, system_prompt, user_prompt):
                return []

        with pytest.raises(RuntimeError, match="0 responses for 1 images"):
            _describer(Short()).describe(image_paths=_images(tmp_path, ["a"]))


# ---------------------------------------------------------------------------
# Revision resolution
# ---------------------------------------------------------------------------
class TestModelRevision:
    def test_explicit_sha_is_returned_unchanged(self):
        assert resolve_model_revision("org/model", SHA_A) == SHA_A

    def test_local_directory_has_no_revision(self, tmp_path):
        assert resolve_model_revision(str(tmp_path)) is None

    def test_offline_without_cache_is_none(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert resolve_model_revision("nobody/no-such-model-in-cache") is None

    def test_offline_uses_the_cached_snapshot(self, monkeypatch, tmp_path):
        import huggingface_hub

        snapshot = tmp_path / "models--org--m" / "snapshots" / SHA_B / "config.json"
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr(
            huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(snapshot), raising=False
        )
        assert resolve_model_revision("org/m") == SHA_B

    def test_backends_resolve_lazily_and_cache(self, monkeypatch):
        calls = []

        def fake_resolve(model, revision=None, timeout=10.0):
            calls.append(model)
            return SHA_A

        import geoai_vlm.describer as describer_mod

        monkeypatch.setattr(describer_mod, "resolve_model_revision", fake_resolve)
        backend = TransformersBackend("org/model")
        assert calls == []
        assert backend.model_revision == SHA_A
        assert backend.model_revision == SHA_A
        assert calls == ["org/model"]

    def test_backend_versions_name_the_library(self):
        tf = TransformersBackend("m").backend_version()
        vl = VLLMBackend("m").backend_version()
        assert tf is None or tf.startswith("transformers ")
        assert vl is None or vl.startswith("vllm ")
