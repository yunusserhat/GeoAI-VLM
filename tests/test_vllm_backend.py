# -*- coding: utf-8 -*-
"""
VLLMBackend behaviour against a fake ``vllm`` module (P0-A item 3 and 5).

The fake mirrors the API verified in vLLM 0.13 and 0.31: ``LLM(**engine_args)``,
``LLM.chat(messages, sampling_params=..., use_tqdm=...)``, ``SamplingParams``
and ``vllm.sampling_params.StructuredOutputsParams(json=...)``. No GPU, model
or network is used.
"""

from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from geoai_vlm.describer import ImageDescriber, RemoteCodeRequiredError, VLLMBackend


class _Completion:
    def __init__(self, text):
        self.text = text


class _RequestOutput:
    def __init__(self, text):
        self.outputs = [_Completion(text)]


def _install_fake_vllm(monkeypatch, *, structured=True, reject_system=False, init_error=None):
    state = {"llm_args": None, "chat_calls": [], "sampling": []}

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state["sampling"].append(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            if init_error is not None:
                raise init_error
            state["llm_args"] = kwargs

        def chat(self, messages, sampling_params=None, use_tqdm=True, **kwargs):
            state["chat_calls"].append({"messages": messages, "sampling": sampling_params})
            outputs = []
            for conv in messages:
                if reject_system and any(m["role"] == "system" for m in conv):
                    raise ValueError("System role not supported")
                image = next(
                    p["image_pil"] for p in conv[-1]["content"] if p.get("type") == "image_pil"
                )
                outputs.append(_RequestOutput(f"img-{image.getpixel((0, 0))[0]}"))
            return outputs

    vllm = types.ModuleType("vllm")
    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    sp_module = types.ModuleType("vllm.sampling_params")
    if structured:
        class StructuredOutputsParams:
            def __init__(self, json=None, **kwargs):
                self.json = json

        sp_module.StructuredOutputsParams = StructuredOutputsParams
    vllm.sampling_params = sp_module
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp_module)
    return state


def _img(identity):
    return Image.new("RGB", (16, 16), color=(identity, identity, identity))


@pytest.fixture
def no_probe(monkeypatch):
    """The HF processor used to probe templates is not loadable offline."""
    monkeypatch.setattr(VLLMBackend, "_probe_renderer", lambda self: None)


class TestEngine:
    def test_engine_arguments(self, monkeypatch, no_probe):
        state = _install_fake_vllm(monkeypatch)
        backend = VLLMBackend(
            "org/model",
            tensor_parallel_size=2,
            max_model_len=8192,
            max_num_seqs=16,
            dtype="bfloat16",
            quantization="awq",
            gpu_memory_utilization=0.5,
            revision="main",
        )
        backend.load_model()
        args = state["llm_args"]
        assert args["model"] == "org/model"
        assert args["trust_remote_code"] is False
        assert args["tensor_parallel_size"] == 2
        assert args["max_model_len"] == 8192
        assert args["max_num_seqs"] == 16
        assert args["dtype"] == "bfloat16"
        assert args["quantization"] == "awq"
        assert args["gpu_memory_utilization"] == 0.5
        assert args["limit_mm_per_prompt"] == {"image": 1}
        assert args["revision"] == "main"
        assert args["seed"] == 42

    def test_optional_arguments_are_omitted_when_unset(self, monkeypatch, no_probe):
        state = _install_fake_vllm(monkeypatch)
        VLLMBackend("org/model", tensor_parallel_size=1).load_model()
        for key in ("max_model_len", "max_num_seqs", "quantization", "revision"):
            assert key not in state["llm_args"]

    def test_custom_mm_limit_and_llm_kwargs(self, monkeypatch, no_probe):
        state = _install_fake_vllm(monkeypatch)
        VLLMBackend(
            "org/model",
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 2},
            llm_kwargs={"enable_prefix_caching": True},
        ).load_model()
        assert state["llm_args"]["limit_mm_per_prompt"] == {"image": 2}
        assert state["llm_args"]["enable_prefix_caching"] is True

    def test_no_qwen_helper_is_imported(self, monkeypatch, no_probe):
        _install_fake_vllm(monkeypatch)
        monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)
        backend = VLLMBackend("org/model", tensor_parallel_size=1)
        assert [o.text for o in backend.generate_outputs([_img(1)], "s", "u")] == ["img-1"]

    def test_remote_code_error_is_explained(self, monkeypatch, no_probe):
        _install_fake_vllm(
            monkeypatch,
            init_error=ValueError("... requires you to execute code. Set trust_remote_code=True"),
        )
        with pytest.raises(RemoteCodeRequiredError, match="org/model"):
            VLLMBackend("org/model", tensor_parallel_size=1).load_model()


class TestChat:
    def test_batch_goes_through_llm_chat_in_order(self, monkeypatch, no_probe):
        state = _install_fake_vllm(monkeypatch)
        backend = VLLMBackend("org/model", tensor_parallel_size=1)
        outputs = backend.generate_outputs([_img(3), _img(1), _img(2)], "sys", "describe")
        assert [o.text for o in outputs] == ["img-3", "img-1", "img-2"]
        assert len(state["chat_calls"]) == 1
        conv = state["chat_calls"][0]["messages"][0]
        assert conv[0] == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
        assert conv[1]["content"][0]["type"] == "image_pil"
        assert conv[1]["content"][1] == {"type": "text", "text": "describe"}

    def test_sampling_parameters(self, monkeypatch, no_probe):
        state = _install_fake_vllm(monkeypatch)
        backend = VLLMBackend(
            "org/model", tensor_parallel_size=1, max_new_tokens=99, temperature=0.3, top_p=0.8
        )
        backend.generate_outputs([_img(1)], "s", "u")
        kwargs = state["sampling"][-1]
        assert kwargs["max_tokens"] == 99
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.8
        assert "structured_outputs" not in kwargs

    def test_unreadable_image_is_a_per_item_error(self, monkeypatch, no_probe, tmp_path):
        _install_fake_vllm(monkeypatch)
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"nope")
        backend = VLLMBackend("org/model", tensor_parallel_size=1)
        out = backend.generate_outputs([_img(1), bad], "s", "u")
        assert out[0].text == "img-1"
        assert out[1].error and "could not be read" in out[1].error

    def test_system_role_error_falls_back_to_prepend(self, monkeypatch, no_probe):
        state = _install_fake_vllm(monkeypatch, reject_system=True)
        backend = VLLMBackend("org/model", tensor_parallel_size=1)
        out = backend.generate_outputs([_img(7)], "Be precise.", "Describe.")
        assert out[0].text == "img-7"
        assert out[0].system_prompt_mode == "prepend"
        retried = state["chat_calls"][-1]["messages"][0]
        assert [m["role"] for m in retried] == ["user"]
        assert retried[0]["content"][1]["text"] == "Be precise.\n\nDescribe."
        # The fallback sticks: the next batch does not try the system role again.
        backend.generate_outputs([_img(8)], "Be precise.", "Describe.")
        assert [m["role"] for m in state["chat_calls"][-1]["messages"][0]] == ["user"]

    def test_explicit_system_mode_does_not_fall_back(self, monkeypatch, no_probe):
        _install_fake_vllm(monkeypatch, reject_system=True)
        backend = VLLMBackend("org/model", tensor_parallel_size=1, system_prompt_mode="system")
        with pytest.raises(ValueError, match="System role not supported"):
            backend.generate_outputs([_img(1)], "s", "u")

    def test_probe_result_is_used_when_available(self, monkeypatch):
        _install_fake_vllm(monkeypatch)

        def rejecting_render(messages):
            if any(m["role"] == "system" for m in messages):
                raise ValueError("System role not supported")
            return "ok"

        monkeypatch.setattr(VLLMBackend, "_probe_renderer", lambda self: rejecting_render)
        backend = VLLMBackend("org/model", tensor_parallel_size=1)
        out = backend.generate_outputs([_img(1)], "s", "u")
        assert out[0].system_prompt_mode == "prepend"


class TestStructuredOutput:
    SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}

    def test_json_schema_constrains_decoding(self, monkeypatch, no_probe):
        state = _install_fake_vllm(monkeypatch, structured=True)
        backend = VLLMBackend(
            "org/model", tensor_parallel_size=1, structured_output=True, json_schema=self.SCHEMA
        )
        out = backend.generate_outputs([_img(1)], "s", "u")
        assert out[0].decoding_mode == "json_schema"
        assert state["sampling"][-1]["structured_outputs"].json == self.SCHEMA

    def test_missing_structured_api_falls_back_and_says_so(self, monkeypatch, no_probe):
        _install_fake_vllm(monkeypatch, structured=False)
        backend = VLLMBackend(
            "org/model", tensor_parallel_size=1, structured_output=True, json_schema=self.SCHEMA
        )
        with pytest.warns(UserWarning, match="cannot constrain decoding"):
            out = backend.generate_outputs([_img(1)], "s", "u")
        assert out[0].decoding_mode == "unconstrained"

    def test_unrequested_structured_output_is_unconstrained(self, monkeypatch, no_probe):
        _install_fake_vllm(monkeypatch)
        out = VLLMBackend("org/model", tensor_parallel_size=1).generate_outputs([_img(1)], "s", "u")
        assert out[0].decoding_mode == "unconstrained"

    def test_decoding_mode_reaches_the_record(self, monkeypatch, no_probe, tmp_path):
        _install_fake_vllm(monkeypatch)
        path = tmp_path / "x.png"
        _img(4).save(path)

        # The template's own schema is used: the fake returns non-JSON text,
        # which must still be recorded as a parse failure, not hidden.
        d = ImageDescriber(
            model_name="org/model",
            backend="vllm",
            prompt_template="simple",
            structured_output=True,
            tensor_parallel_size=1,
        )
        df = d.describe(image_paths=[path])
        row = df.iloc[0]
        assert row["decoding_mode"] == "json_schema"
        assert row["backend"] == "vllm"
        assert bool(row["parse_error"]) is True
