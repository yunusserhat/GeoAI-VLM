# -*- coding: utf-8 -*-
"""
TransformersBackend behaviour with mock processor and model objects (P0-A).

Neither torch nor transformers is needed: a fake ``transformers`` module is
injected where loading is exercised, and tensors are numpy arrays. The fake
processor gives each image a different prompt length and the fake model
"generates" garbage when a row ends in padding, so batch order and left
padding are tested for real rather than assumed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from geoai_vlm import describer as describer_mod
from geoai_vlm.describer import (
    ImageDescriber,
    RemoteCodeRequiredError,
    TransformersBackend,
)

PAD = 0


def _image(identity: int, width: int) -> Image.Image:
    """An image whose grey level is its identity and whose width sets its length."""
    return Image.new("RGB", (width, 8), color=(identity, identity, identity))


def _messages_text(conv):
    out = []
    for m in conv:
        content = m["content"]
        if isinstance(content, str):
            out.append((m["role"], content))
        else:
            out.append((m["role"], "".join(p.get("text", "") for p in content if p.get("type") == "text")))
    return out


class FakeBatch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moved_to = None

    def to(self, device, dtype=None):
        self.moved_to = (device, dtype)
        return self


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "</s>"


class FakeProcessor:
    """Processor with a transformers-5 style ``apply_chat_template``."""

    def __init__(self, template="system-ok"):
        self.tokenizer = FakeTokenizer()
        self.chat_template = None if template is None else template
        self.template = template
        self.calls = []

    def _render(self, conversation):
        roles = [m["role"] for m in conversation]
        if self.template == "system-rejected" and "system" in roles:
            raise ValueError("System role not supported")
        msgs = [
            (r, t) for r, t in _messages_text(conversation)
            if not (self.template == "system-dropped" and r == "system")
        ]
        return "".join(f"<{r}>{t}" for r, t in msgs) + "<assistant>"

    def _encode(self, conversation):
        identity, width = None, None
        for m in conversation:
            if isinstance(m["content"], list):
                for part in m["content"]:
                    if part.get("type") == "image":
                        img = part["image"]
                        identity = img.getpixel((0, 0))[0]
                        width = img.size[0]
        length = 3 + width // 8
        return [7] * length + [identity]

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=False,
        return_tensors=None,
        return_dict=False,
        processor_kwargs=None,
        **kwargs,
    ):
        batched = isinstance(conversation[0], list)
        convs = conversation if batched else [conversation]
        if not tokenize:
            rendered = [self._render(c) for c in convs]
            return rendered if batched else rendered[0]
        options = dict(processor_kwargs or {})
        self.calls.append({"conversations": convs, "options": options, "kwargs": kwargs})
        for c in convs:
            self._render(c)  # a real template would fail here too
        side = options.get("padding_side", self.tokenizer.padding_side)
        seqs = [self._encode(c) for c in convs]
        width = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = [PAD] * (width - len(s))
            ids.append(pad + s if side == "left" else s + pad)
            mask.append([0] * len(pad) + [1] * len(s) if side == "left" else [1] * len(s) + [0] * len(pad))
        batch = FakeBatch(input_ids=np.array(ids), attention_mask=np.array(mask))
        self.calls[-1]["batch"] = batch
        return batch

    def batch_decode(self, tokens, skip_special_tokens=True):
        out = []
        for row in tokens:
            row = [int(t) for t in row]
            if row and row[0] == 999:
                out.append("CORRUPT: continued from padding")
            else:
                out.append(" " + " ".join(f"desc-{t - 100}" for t in row) + " ")
        return out


class FakeV4Processor(FakeProcessor):
    """Processor with the transformers-4 signature: no processor_kwargs."""

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        options = {k: kwargs.pop(k) for k in list(kwargs) if k in ("padding", "padding_side")}
        return super().apply_chat_template(
            conversation, chat_template=chat_template, processor_kwargs=options, **kwargs
        )


class FakeModel:
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.generate_calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.generate_calls.append(kwargs)
        new = []
        for row in input_ids:
            # A right-padded row ends in PAD: the model would continue from padding.
            new.append([999] if row[-1] == PAD else [100 + int(row[-1])])
        return np.concatenate([input_ids, np.array(new)], axis=1)


def _ready_backend(processor=None, **kwargs) -> TransformersBackend:
    backend = TransformersBackend("fake/model", **kwargs)
    backend.processor = processor or FakeProcessor()
    backend.processor.tokenizer.padding_side = "left"  # what load_model sets
    backend.model = FakeModel()
    return backend


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
class TestBatchedGeneration:
    def test_output_order_matches_input_order(self):
        backend = _ready_backend()
        images = [_image(3, 64), _image(1, 8), _image(2, 32)]
        outputs = backend.generate_outputs(images, "sys", "describe")
        assert [o.text for o in outputs] == ["desc-3", "desc-1", "desc-2"]
        assert len(backend.model.generate_calls) == 1, "batch must be one generate call"

    def test_left_padding_is_requested_from_the_processor(self):
        backend = _ready_backend()
        backend.generate_outputs([_image(1, 8), _image(2, 64)], "sys", "describe")
        options = backend.processor.calls[-1]["options"]
        assert options.get("padding") is True
        assert options.get("padding_side") == "left"

    def test_right_padding_would_have_corrupted_the_batch(self):
        """Guards the fake itself: it must be able to detect right padding."""
        backend = _ready_backend()
        processor = backend.processor
        enc = processor.apply_chat_template(
            [
                describer_mod.build_chat_messages("u", _image(1, 8), "s"),
                describer_mod.build_chat_messages("u", _image(2, 64), "s"),
            ],
            tokenize=True,
            processor_kwargs={"padding": True, "padding_side": "right"},
        )
        out = backend.model.generate(**enc)
        texts = processor.batch_decode(out[:, enc["input_ids"].shape[1]:])
        assert any("CORRUPT" in t for t in texts)

    def test_transformers4_signature_gets_keyword_arguments(self):
        backend = _ready_backend(processor=FakeV4Processor())
        outputs = backend.generate_outputs([_image(5, 8), _image(6, 40)], "s", "u")
        assert [o.text for o in outputs] == ["desc-5", "desc-6"]
        assert backend.processor.calls[-1]["options"] == {"padding": True, "padding_side": "left"}

    def test_inputs_are_moved_to_the_model_device_and_dtype(self):
        backend = _ready_backend()
        backend.generate_outputs([_image(1, 8)], "s", "u")
        assert backend.processor.calls[-1]["batch"].moved_to == ("cpu", "float32")

    def test_unreadable_image_is_a_per_item_error(self, tmp_path):
        bad = tmp_path / "broken.jpg"
        bad.write_bytes(b"not an image")
        backend = _ready_backend()
        outputs = backend.generate_outputs([_image(1, 8), bad, _image(2, 16)], "s", "u")
        assert outputs[0].text == "desc-1" and outputs[0].error is None
        assert outputs[1].text == "" and "could not be read" in outputs[1].error
        assert outputs[2].text == "desc-2"

    def test_generate_returns_plain_strings(self):
        backend = _ready_backend()
        assert backend.generate([_image(4, 8)], "s", "u") == ["desc-4"]

    def test_decoding_is_recorded_as_unconstrained(self):
        backend = _ready_backend(structured_output=True, json_schema={"type": "object"})
        with pytest.warns(UserWarning, match="cannot constrain decoding"):
            out = backend.generate_outputs([_image(1, 8)], "s", "u")
        assert out[0].decoding_mode == "unconstrained"


class TestGenerationSettings:
    def test_greedy_by_default(self):
        backend = _ready_backend(max_new_tokens=12)
        backend.generate_outputs([_image(1, 8)], "s", "u")
        kwargs = backend.model.generate_calls[-1]
        assert kwargs == {"max_new_tokens": 12, "do_sample": False}

    def test_sampling_settings_are_passed(self):
        backend = _ready_backend(temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.1)
        backend.generate_outputs([_image(1, 8)], "s", "u")
        kwargs = backend.model.generate_calls[-1]
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9 and kwargs["top_k"] == 20
        assert kwargs["repetition_penalty"] == 1.1

    def test_old_parameter_names_still_work(self):
        backend = TransformersBackend("m", "cuda", "bfloat16", 512, 0.2)
        assert backend.device_map == "cuda" and backend.device == "cuda"
        assert backend.dtype == "bfloat16" and backend.torch_dtype == "bfloat16"
        assert backend.max_new_tokens == 512 and backend.max_tokens == 512
        assert backend.temperature == 0.2

    def test_new_parameter_names(self):
        backend = TransformersBackend(
            "m", dtype="float16", device_map="cpu", max_new_tokens=64, attn_implementation="sdpa"
        )
        assert (backend.dtype, backend.device_map, backend.max_new_tokens) == ("float16", "cpu", 64)
        assert backend.attn_implementation == "sdpa"

    def test_conflicting_old_and_new_names_raise(self):
        with pytest.raises(ValueError, match="conflicts"):
            TransformersBackend("m", torch_dtype="float16", dtype="bfloat16")

    def test_quantization_values_are_validated(self):
        with pytest.raises(ValueError):
            TransformersBackend("m", quantization="3bit")


# ---------------------------------------------------------------------------
# System prompt handling
# ---------------------------------------------------------------------------
class TestSystemPromptModes:
    def test_template_with_system_role_uses_it(self):
        backend = _ready_backend(processor=FakeProcessor("system-ok"))
        out = backend.generate_outputs([_image(1, 8)], "Be precise.", "Describe.")
        conv = backend.processor.calls[-1]["conversations"][0]
        assert conv[0]["role"] == "system"
        assert out[0].system_prompt_mode == "system"

    @pytest.mark.parametrize("template", ["system-rejected", "system-dropped"])
    def test_auto_falls_back_to_prepend(self, template):
        backend = _ready_backend(processor=FakeProcessor(template))
        out = backend.generate_outputs([_image(1, 8)], "Be precise.", "Describe.")
        conv = backend.processor.calls[-1]["conversations"][0]
        assert [m["role"] for m in conv] == ["user"]
        assert _messages_text(conv)[0][1] == "Be precise.\n\nDescribe."
        assert out[0].system_prompt_mode == "prepend"
        assert out[0].text == "desc-1"

    def test_forced_prepend_never_sends_system(self):
        backend = _ready_backend(processor=FakeProcessor("system-ok"), system_prompt_mode="prepend")
        out = backend.generate_outputs([_image(1, 8)], "s", "u")
        assert backend.processor.calls[-1]["conversations"][0][0]["role"] == "user"
        assert out[0].system_prompt_mode == "prepend"

    def test_forced_system_mode_fails_loudly_on_a_rejecting_template(self):
        backend = _ready_backend(processor=FakeProcessor("system-rejected"), system_prompt_mode="system")
        with pytest.raises(ValueError, match="System role not supported"):
            backend.generate_outputs([_image(1, 8)], "s", "u")

    def test_missing_chat_template_is_a_clear_error(self):
        backend = _ready_backend(processor=FakeProcessor(None))
        with pytest.raises(ValueError, match="no chat template"):
            backend.generate_outputs([_image(1, 8)], "s", "u")

    def test_describer_records_the_effective_mode(self, tmp_path):
        path = tmp_path / "img1.png"
        _image(1, 8).save(path)
        backend = _ready_backend(processor=FakeProcessor("system-rejected"))
        d = ImageDescriber(model_name="fake/model", backend=backend, prompt_template="simple")
        df = d.describe(image_paths=[path])
        row = df.iloc[0]
        assert row["system_prompt_mode_effective"] == "prepend"
        assert row["backend"] == "transformers"
        assert row["decoding_mode"] == "unconstrained"


# ---------------------------------------------------------------------------
# Loading: trust_remote_code, model class, padding side
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_transformers(monkeypatch):
    """Install a fake ``transformers`` module and record from_pretrained calls."""
    calls = {"processor": [], "model": []}
    behaviour = {"raise": None, "processor": FakeProcessor}

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["processor"].append((name, kwargs))
            if behaviour["raise"] is not None:
                raise behaviour["raise"]
            return behaviour["processor"]()

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["model"].append((name, kwargs))
            return FakeModel()

    module = types.ModuleType("transformers")
    module.__version__ = "5.19.0"
    module.AutoProcessor = AutoProcessor
    module.AutoModelForImageTextToText = AutoModelForImageTextToText
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(describer_mod, "_dtype_load_kwarg", lambda dtype: {"dtype": dtype})
    return module, calls, behaviour


class TestLoading:
    def test_trust_remote_code_is_off_by_default(self, fake_transformers):
        _, calls, _ = fake_transformers
        backend = TransformersBackend("fake/model")
        assert backend.trust_remote_code is False
        backend.load_model()
        assert calls["processor"][0][1]["trust_remote_code"] is False
        assert calls["model"][0][1]["trust_remote_code"] is False

    def test_describer_default_does_not_trust_remote_code(self):
        d = ImageDescriber(model_name="fake/model", backend="transformers")
        assert d.trust_remote_code is False
        assert d.backend.trust_remote_code is False

    def test_trust_remote_code_must_be_explicit(self, fake_transformers):
        _, calls, _ = fake_transformers
        TransformersBackend("fake/model", trust_remote_code=True).load_model()
        assert calls["model"][0][1]["trust_remote_code"] is True

    def test_model_needing_remote_code_gets_a_clear_error(self, fake_transformers):
        _, _, behaviour = fake_transformers
        behaviour["raise"] = ValueError(
            "The repository fake/model contains custom code which must be executed to "
            "correctly load the model. Please pass the argument `trust_remote_code=True`"
        )
        with pytest.raises(RemoteCodeRequiredError) as exc:
            TransformersBackend("fake/model").load_model()
        message = str(exc.value)
        assert "fake/model" in message
        assert "trust_remote_code=True" in message
        assert "does not run downloaded code by default" in message

    def test_load_sets_left_padding_and_pad_token(self, fake_transformers):
        backend = TransformersBackend("fake/model")
        backend.load_model()
        assert backend.processor.tokenizer.padding_side == "left"
        assert backend.processor.tokenizer.pad_token == "</s>"
        assert backend.model.eval_called

    def test_load_passes_model_options(self, fake_transformers):
        _, calls, _ = fake_transformers
        TransformersBackend(
            "fake/model",
            device_map="cpu",
            dtype="float32",
            attn_implementation="sdpa",
            revision="abc",
        ).load_model()
        kwargs = calls["model"][0][1]
        assert kwargs["device_map"] == "cpu"
        assert kwargs["dtype"] == "float32"
        assert kwargs["attn_implementation"] == "sdpa"
        assert kwargs["revision"] == "abc"
        assert calls["processor"][0][1]["revision"] == "abc"

    def test_falls_back_to_vision2seq_on_old_transformers(self, fake_transformers, monkeypatch):
        module, calls, _ = fake_transformers
        legacy = module.AutoModelForImageTextToText
        del module.AutoModelForImageTextToText
        module.AutoModelForVision2Seq = legacy
        with pytest.warns(UserWarning, match="AutoModelForVision2Seq"):
            TransformersBackend("fake/model").load_model()
        assert calls["model"]

    def test_quantization_without_bitsandbytes_names_the_extra(self, fake_transformers, monkeypatch):
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)
        with pytest.raises(ImportError, match=r"geoai-vlm\[quant\]"):
            TransformersBackend("fake/model", quantization="4bit").load_model()

    def test_missing_transformers_names_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError, match=r"geoai-vlm\[transformers\]"):
            TransformersBackend("fake/model").load_model()
