# -*- coding: utf-8 -*-
"""
check_model and the ``geoai-vlm check-model`` command (P0-A item 7).

Run against the fake HTTP server and the fake transformers objects used by the
backend tests, so the report logic is tested without downloading a model.
"""

from __future__ import annotations

import json

import pytest

from geoai_vlm import cli
from geoai_vlm.describer import TransformersBackend
from geoai_vlm.models import ModelCheckReport, check_model
from tests.test_openai_backend import FakeServer
from tests.test_transformers_backend import FakeModel, FakeProcessor


@pytest.fixture
def server():
    with FakeServer() as srv:
        yield srv


class JSONProcessor(FakeProcessor):
    """Decodes every generation as a valid answer to the simple template."""

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [json.dumps({"description": "drawn street", "tags": ["synthetic"]})] * len(tokens)


@pytest.fixture
def fake_transformers_load(monkeypatch):
    """Make TransformersBackend.load_model install fakes instead of downloading."""
    config = {"value": "system-ok", "fail": None, "processor_cls": JSONProcessor}

    def load(self):
        if config["fail"]:
            raise config["fail"]
        self.processor = config["processor_cls"](config["value"])
        self.model = FakeModel()

    monkeypatch.setattr(TransformersBackend, "load_model", load)
    return config


@pytest.fixture
def no_backoff(monkeypatch):
    from geoai_vlm.openai_compat import OpenAICompatibleBackend

    monkeypatch.setattr(OpenAICompatibleBackend, "_sleep_before_retry", lambda self, a, r: None)


class TestCheckModelTransformers:
    def test_ok_report(self, fake_transformers_load):
        report = check_model("fake/model", backend="transformers", revision="a" * 40)
        assert isinstance(report, ModelCheckReport)
        assert report.status == "ok"
        assert report.loaded and report.generated and report.json_parsed
        assert report.chat_template is True
        assert report.system_role == "supported"
        assert report.system_prompt_mode_effective == "system"
        assert report.decoding_mode == "unconstrained"
        assert report.model_revision == "a" * 40
        assert report.image == "synthetic"
        assert report.load_seconds is not None and report.generate_seconds is not None
        assert report.validation_issues == []

    def test_template_without_system_role_is_reported(self, fake_transformers_load):
        fake_transformers_load["value"] = "system-rejected"
        report = check_model("fake/model", backend="transformers")
        assert report.status == "ok"
        assert report.system_prompt_mode_effective == "prepend"
        assert report.system_role.startswith("not supported")
        assert "rejected" in report.system_prompt_note

    def test_missing_chat_template(self, fake_transformers_load):
        fake_transformers_load["value"] = None
        report = check_model("fake/model", backend="transformers")
        assert report.chat_template is False
        assert report.status == "failed"
        assert any("no chat template" in e for e in report.errors)

    def test_load_failure_is_captured(self, fake_transformers_load):
        fake_transformers_load["fail"] = RuntimeError("weights missing")
        report = check_model("fake/model", backend="transformers")
        assert report.status == "failed"
        assert not report.loaded
        assert any("weights missing" in e for e in report.errors)

    def test_non_json_answer_is_partial(self, fake_transformers_load):
        class ProseProcessor(FakeProcessor):
            def batch_decode(self, tokens, skip_special_tokens=True):
                return ["A street with a tree."] * len(tokens)

        fake_transformers_load["processor_cls"] = ProseProcessor
        report = check_model("fake/model", backend="transformers")
        assert report.status == "partial"
        assert report.json_parsed is False
        assert report.parse_error

    def test_summary_is_readable(self, fake_transformers_load):
        text = check_model("fake/model", backend="transformers").summary()
        assert "status" in text and "system role" in text and "JSON parsed" in text


class TestCheckModelOpenAI:
    def test_against_a_server(self, server):
        report = check_model("fake-vlm", backend="openai", base_url=server.base_url)
        assert report.status == "ok"
        assert report.chat_template is None, "a remote template is not inspectable"
        assert report.system_role == "not verified (template not inspectable)"
        assert report.errors == []

    def test_unlisted_model_is_flagged(self, server):
        report = check_model("other-model", backend="openai", base_url=server.base_url)
        assert any("does not list" in e for e in report.errors)

    def test_unknown_backend_fails_cleanly(self):
        report = check_model("m", backend="nope")
        assert report.status == "failed"
        assert any("Unknown backend" in e for e in report.errors)


class TestCli:
    def test_json_output_is_not_polluted_by_progress_messages(self, fake_transformers_load, capsys):
        code = cli.main(["check-model", "fake/model", "--json"])
        captured = capsys.readouterr()
        report = json.loads(captured.out)  # must parse as a whole
        assert report["status"] == "ok" and code == 0

    def test_check_model_command_json(self, server, capsys):
        code = cli.main(
            ["check-model", "fake-vlm", "--backend", "openai", "--base-url", server.base_url, "--json"]
        )
        out = json.loads(capsys.readouterr().out)
        assert code == 0
        assert out["status"] == "ok"
        assert out["model_id"] == "fake-vlm"

    def test_exit_code_reflects_failure(self, capsys, no_backoff):
        code = cli.main(
            [
                "check-model", "m", "--backend", "openai",
                "--base-url", "http://127.0.0.1:9/v1",
            ]
        )
        assert code == 2
        assert "failed" in capsys.readouterr().out

    def test_api_key_flag_takes_a_variable_name(self, server, monkeypatch, capsys):
        monkeypatch.setenv("CHECK_KEY", "sk-cli-secret-123456")
        cli.main(
            [
                "check-model", "fake-vlm", "--backend", "openai",
                "--base-url", server.base_url, "--api-key-env", "CHECK_KEY",
            ]
        )
        assert server.requests[-1]["headers"]["Authorization"] == "Bearer sk-cli-secret-123456"
        assert "sk-cli-secret-123456" not in capsys.readouterr().out

    def test_console_script_is_declared(self):
        from importlib.metadata import entry_points

        scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
        assert scripts.get("geoai-vlm") == "geoai_vlm.cli:main"
