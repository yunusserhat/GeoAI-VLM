# -*- coding: utf-8 -*-
"""
Image Describer Module for GeoAI-VLM
=====================================
VLM-based image description with model-agnostic backends:

* :class:`VLLMBackend` -- vLLM's generic chat path (``LLM.chat``), so every
  vision-language model vLLM supports runs with its own chat template and
  processor;
* :class:`TransformersBackend` -- Hugging Face ``AutoProcessor`` +
  ``AutoModelForImageTextToText`` with batched, left-padded generation;
* :class:`~geoai_vlm.openai_compat.OpenAICompatibleBackend` -- any
  OpenAI-compatible HTTP endpoint (vLLM server, TGI / Inference Endpoints,
  SGLang, Ollama, LM Studio, ...).

``backend="auto"`` keeps its historical meaning: vLLM when it is installed and
a GPU is present, Transformers otherwise.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
from tqdm import tqdm

from .chat import (
    SYSTEM_PROMPT_MODES,
    GenerationOutput,
    build_chat_messages,
    resolve_system_prompt_mode,
)
from .prompts import GEOAI_SYSTEM_PROMPT, GEOAI_USER_PROMPT, get_prompt_template
from .provenance import (
    canonical_generation_params,
    compute_processing_id,
    library_version,
    resolve_model_revision,
    schema_digest,
)
from .schemas import validate_json


__all__ = [
    "ImageDescriber",
    "BaseBackend",
    "VLLMBackend",
    "TransformersBackend",
    "OpenAICompatibleBackend",  # noqa: F822 - provided lazily by __getattr__ below
    "RemoteCodeRequiredError",
    "parse_json_response",
    "extract_summary_fields",
    "DESCRIPTION_COLUMNS",
    "DEFAULT_MODEL",
]

logger = logging.getLogger(__name__)

#: Default description model.
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

#: Columns of a description record, in the order describe() produces them.
#: Columns added in 0.4 are appended so positional readers keep working.
DESCRIPTION_COLUMNS = (
    "image_path",
    "image_id",
    "raw_response",
    "parsed_json",
    "parse_error",
    "model_name",
    "prompt_version",
    "processing_id",
    "processed_at",
    "scene_narrative",
    "semantic_tags",
    "land_use_primary",
    "street_type",
    "place_character",
    "usable",
    "quality_status",
    # -- provenance added in 0.4 --
    "backend",
    "backend_version",
    "model_revision",
    "generation_params",
    "system_prompt_mode_effective",
    "decoding_mode",
    "generation_error",
    "validation_issues",
)


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from model response.

    Args:
        text: Raw text response from VLM

    Returns:
        Parsed JSON dictionary, or error dict if parsing fails
    """
    try:
        text = text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Find start and end of code block
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("```") and i == 0:
                    start_idx = 1
                elif line.startswith("```") and i > 0:
                    end_idx = i
                    break
            text = "\n".join(lines[start_idx:end_idx])

            # Remove json language identifier if present
            if text.startswith("json"):
                text = text[4:].strip()

        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}", "raw_response": text}

    # Syntactically valid JSON is not necessarily a usable record. A bare list,
    # null, number or string parses cleanly but has none of the expected fields,
    # and silently returning it pushed an AttributeError into the caller mid-batch.
    if not isinstance(parsed, dict):
        return {
            "error": (
                "Expected a JSON object, got "
                f"{type(parsed).__name__}"
            ),
            "raw_response": text,
        }

    return parsed



# Summary columns are populated from whichever schema the prompt template
# produces. The "geoai" template nests its fields; the "simple" template uses
# flat description/tags keys. Reading only the geoai names silently left every
# simple-template run with empty summary columns.
_NARRATIVE_KEYS = ("scene_narrative", "description", "alt_detailed")
_TAG_KEYS = ("semantic_tags", "tags", "keywords")


def _first_present(parsed: Dict[str, Any], keys) -> Optional[Any]:
    for key in keys:
        if key in parsed and parsed[key] not in (None, ""):
            return parsed[key]
    return None


def _nested(parsed: Dict[str, Any], outer: str, inner: str) -> str:
    block = parsed.get(outer)
    if isinstance(block, dict):
        value = block.get(inner)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def extract_summary_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the flat summary columns from a parsed model response.

    Quality is reported as three distinguishable states rather than collapsed
    into a permissive default:

    ``reported``  the model stated whether the image is usable (``usable`` is
                  True/False)
    ``unknown``   no quality block was returned (``usable`` is None -- it is
                  *not* assumed usable)
    ``error``     the response could not be parsed (``usable`` is None)
    """
    if "error" in parsed:
        return {
            "scene_narrative": "",
            "semantic_tags": "",
            "land_use_primary": "error",
            "street_type": "error",
            "place_character": "error",
            "usable": None,
            "quality_status": "error",
        }

    narrative = _first_present(parsed, _NARRATIVE_KEYS)
    tags = _first_present(parsed, _TAG_KEYS)

    if isinstance(tags, (list, tuple)):
        tags_str = ",".join(str(t) for t in tags)
    elif tags is None:
        tags_str = ""
    else:
        tags_str = str(tags)

    quality = parsed.get("image_quality")
    raw_usable = quality.get("usable_for_analysis") if isinstance(quality, dict) else None
    # Only a real boolean counts as a verdict. Coercing with bool() would turn
    # null into "reported unusable" and the string "false" into "reported
    # usable" -- both of which invent a judgement the model never made.
    if isinstance(raw_usable, bool):
        usable = raw_usable
        quality_status = "reported"
    else:
        usable = None
        quality_status = "unknown"

    return {
        "scene_narrative": "" if narrative is None else str(narrative),
        "semantic_tags": tags_str,
        "land_use_primary": _nested(parsed, "land_use_character", "primary"),
        "street_type": _nested(parsed, "urban_morphology", "street_type"),
        "place_character": _nested(parsed, "place_character", "dominant_activity"),
        "usable": usable,
        "quality_status": quality_status,
    }


def _upsert_records(
    existing: Optional[pd.DataFrame], batch: pd.DataFrame
) -> pd.DataFrame:
    """Append *batch*, replacing any prior rows for the same derived output.

    Keyed on (image_id, processing_id) so that re-describing an image with the
    same model and prompt overwrites its record instead of adding a duplicate,
    while a different model or prompt is kept as a separate record.
    """
    if existing is None or len(existing) == 0:
        return batch.reset_index(drop=True)

    if {"image_id", "processing_id"}.issubset(existing.columns):
        keys = set(
            zip(batch["image_id"].astype(str), batch["processing_id"].astype(str))
        )
        mask = [
            (str(i), str(pid)) not in keys
            for i, pid in zip(existing["image_id"], existing["processing_id"])
        ]
        existing = existing[mask]

    return pd.concat([existing, batch], ignore_index=True)


# =============================================================================
# Errors and small helpers
# =============================================================================
class RemoteCodeRequiredError(ValueError):
    """Raised when a model needs ``trust_remote_code=True`` and it was not given.

    geoai-vlm never executes code downloaded with a model unless asked to.
    """

    def __init__(self, model_name: str, backend: str):
        super().__init__(
            f"Model {model_name!r} ships custom Python code that must be executed "
            "to load it, and geoai-vlm does not run downloaded code by default. "
            "Review the model repository, then opt in explicitly, for example "
            f"ImageDescriber(model_name={model_name!r}, backend={backend!r}, "
            "trust_remote_code=True). Prefer a natively supported checkpoint "
            "(often published with an '-hf' suffix) when one exists."
        )
        self.model_name = model_name


def _needs_remote_code(exc: BaseException) -> bool:
    return "trust_remote_code" in str(exc)


def _merge_alias(new_name: str, new_value: Any, old_name: str, old_value: Any, old_default: Any) -> Any:
    """Combine a new parameter with the older name it replaces."""
    if new_value is None:
        return old_value
    if old_value != old_default and old_value != new_value:
        raise ValueError(
            f"{new_name}={new_value!r} conflicts with {old_name}={old_value!r}; "
            f"pass only {new_name} ({old_name} is its older name)"
        )
    return new_value


def _cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


@contextlib.contextmanager
def _no_grad():
    try:
        import torch
    except ImportError:
        yield
        return
    with torch.no_grad():
        yield


def _manual_seed(seed: int) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def _looks_like_system_role_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if type(exc).__name__ == "TemplateError" or "jinja" in type(exc).__module__.lower():
        return True
    return "system" in text and ("role" in text or "template" in text)


# =============================================================================
# Backends
# =============================================================================
class BaseBackend(ABC):
    """Abstract base class for VLM backends.

    A backend must implement :meth:`load_model`, :meth:`generate` and
    :meth:`is_available`. Backends that can say more about each response --
    a per-item error, whether decoding was constrained, how the system prompt
    was delivered -- override :meth:`generate_outputs`; the default wraps
    :meth:`generate`, so custom backends written for 0.3 keep working.
    """

    #: Short name recorded as ``backend`` on every description.
    name: str = "custom"

    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def generate(
        self,
        image_paths: List[str],
        system_prompt: str,
        user_prompt: str,
    ) -> List[str]:
        """Generate descriptions for a batch of images."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass

    def generate_outputs(
        self,
        images: Sequence[Any],
        system_prompt: str,
        user_prompt: str,
    ) -> List[GenerationOutput]:
        """Generate one :class:`~geoai_vlm.chat.GenerationOutput` per image."""
        texts = self.generate([str(i) for i in images], system_prompt, user_prompt)
        return [GenerationOutput(text=t) for t in texts]

    def provenance(self) -> Dict[str, Any]:
        """What this backend can say about how it produces output."""
        return {
            "backend": self.name,
            "backend_version": None,
            "model_revision": None,
            "generation_params": {},
        }


class _ChatBackend(BaseBackend):
    """Shared configuration and provenance for the built-in chat backends."""

    name = "chat"
    _image_style = "hf"
    _system_content = "parts"

    def _init_common(
        self,
        model_name: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        seed: Optional[int],
        system_prompt_mode: str,
        structured_output: bool,
        json_schema: Optional[Dict[str, Any]],
        trust_remote_code: bool,
        revision: Optional[str],
    ) -> None:
        if system_prompt_mode not in SYSTEM_PROMPT_MODES:
            raise ValueError(
                f"system_prompt_mode must be one of {SYSTEM_PROMPT_MODES}, "
                f"got {system_prompt_mode!r}"
            )
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.system_prompt_mode = system_prompt_mode
        self.structured_output = bool(structured_output)
        self.json_schema = json_schema
        self.trust_remote_code = bool(trust_remote_code)
        self.revision = revision
        # system prompt text -> (effective mode, reason)
        self._system_modes: Dict[str, Any] = {}
        self._resolved_revision: Any = _UNSET
        self._warned_structured = False

    # Backwards-compatible name for max_new_tokens.
    @property
    def max_tokens(self) -> int:
        return self.max_new_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self.max_new_tokens = int(value)

    @property
    def wants_structured(self) -> bool:
        """True when constrained decoding was requested and a schema is set."""
        return bool(self.structured_output and self.json_schema)

    # -- system prompt handling -----------------------------------------
    def _probe_renderer(self) -> Optional[Callable[[List[Dict[str, Any]]], Any]]:
        """A callable rendering messages with the chat template, if inspectable."""
        return None

    def _system_mode_for(self, system_prompt: Optional[str]) -> str:
        key = system_prompt or ""
        if key not in self._system_modes:
            render = self._probe_renderer() if self.system_prompt_mode == "auto" else None
            self._system_modes[key] = resolve_system_prompt_mode(
                self.system_prompt_mode, system_prompt, render
            )
        return self._system_modes[key][0]

    def _fall_back_to_prepend(self, system_prompt: Optional[str], reason: str) -> None:
        logger.warning("system prompt will be prepended to the user turn: %s", reason)
        self._system_modes[system_prompt or ""] = ("prepend", reason)

    def system_prompt_mode_reason(self, system_prompt: Optional[str]) -> Optional[str]:
        """Why the effective mode differs from a plain system turn, if it does."""
        entry = self._system_modes.get(system_prompt or "")
        return entry[1] if entry else None

    def _text_chat(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalise a text-only chat and apply the system-prompt fallback.

        Content becomes a list of text parts (what processor templates
        expect). When the chat template cannot take a system turn, the system
        text is moved to the start of the first user turn, exactly as for
        image descriptions.
        """
        chat: List[Dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            parts = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            chat.append({"role": m["role"], "content": parts})
        if chat and chat[0]["role"] == "system":
            system_text = "".join(p.get("text", "") for p in chat[0]["content"])
            if self._system_mode_for(system_text) == "prepend":
                rest = chat[1:]
                if rest and rest[0]["role"] == "user":
                    rest[0] = {
                        "role": "user",
                        "content": [{"type": "text", "text": system_text + "\n\n"}] + rest[0]["content"],
                    }
                else:
                    rest.insert(0, {"role": "user", "content": [{"type": "text", "text": system_text}]})
                chat = rest
        return chat

    def _warn_unstructured(self) -> None:
        if self.wants_structured and not self._warned_structured:
            warnings.warn(
                f"{type(self).__name__} cannot constrain decoding to a JSON schema "
                "here; responses are parsed as free text (decoding_mode='unconstrained').",
                stacklevel=3,
            )
            self._warned_structured = True

    # -- provenance -------------------------------------------------------
    def generation_params(self) -> Dict[str, Any]:
        """Output-affecting settings under canonical names."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "repetition_penalty": self.repetition_penalty,
            "system_prompt_mode": self.system_prompt_mode,
            "structured_output": self.structured_output or None,
            "json_schema_digest": schema_digest(self.json_schema) if self.wants_structured else None,
        }

    def backend_version(self) -> Optional[str]:
        return None

    @property
    def model_revision(self) -> Optional[str]:
        """Resolved Hugging Face commit sha (``None`` when it cannot be known)."""
        if self._resolved_revision is _UNSET:
            self._resolved_revision = resolve_model_revision(self.model_name, self.revision)
        return self._resolved_revision

    def provenance(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "backend_version": self.backend_version(),
            "model_revision": self.model_revision,
            "generation_params": self.generation_params(),
        }


class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


def _auto_vlm_class():
    """``AutoModelForImageTextToText``, or the older Vision2Seq class.

    ``AutoModelForImageTextToText`` exists from transformers 4.46 and is the
    only one left in transformers 5 (``AutoModelForVision2Seq`` was removed).
    The fallback only matters for installs older than the ``vlm`` extra pins.
    """
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        warnings.warn(
            "transformers has no AutoModelForImageTextToText; falling back to "
            "AutoModelForVision2Seq. Upgrade with: pip install 'geoai-vlm[vlm]'",
            stacklevel=3,
        )
        return AutoModelForVision2Seq


def _dtype_load_kwarg(dtype: Optional[str]) -> Dict[str, Any]:
    """``{"dtype": ...}`` on transformers >= 4.56, ``{"torch_dtype": ...}`` before."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The Transformers backend needs torch. Install it with: "
            "pip install 'geoai-vlm[transformers]'"
        ) from exc
    import transformers
    from packaging.version import Version

    if dtype in (None, "auto"):
        # Historical default of this package: bf16 on GPU, fp32 on CPU.
        value = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        value = getattr(torch, str(dtype), None)
        if not isinstance(value, torch.dtype):
            raise ValueError(f"unknown dtype {dtype!r}; use e.g. 'bfloat16', 'float16', 'float32'")
    key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    return {key: value}


def _bnb_config(quantization: str, dtype: Optional[str]):
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            f"quantization={quantization!r} needs bitsandbytes. "
            "Install it with: pip install 'geoai-vlm[quant]'"
        ) from exc
    import torch

    compute = getattr(torch, str(dtype), None) if dtype not in (None, "auto") else None
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute or torch.bfloat16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


class TransformersBackend(_ChatBackend):
    """Hugging Face Transformers backend for image-text-to-text models.

    Loads ``AutoProcessor`` + ``AutoModelForImageTextToText`` (any model
    transformers registers for image-text-to-text), builds the chat with the
    image inside the message, and generates a whole batch at once with left
    padding. Output order always equals input order.

    Args:
        model_name: Hugging Face model id or local path.
        device: Older name for ``device_map``.
        torch_dtype: Older name for ``dtype``.
        max_tokens: Older name for ``max_new_tokens``.
        temperature: ``0`` (default) decodes greedily; ``> 0`` samples.
        dtype: ``"auto"`` (bf16 on GPU, fp32 on CPU), or a torch dtype name.
        device_map: Passed to ``from_pretrained`` (default ``"auto"``).
        attn_implementation: e.g. ``"sdpa"`` or ``"flash_attention_2"``.
        max_new_tokens: Maximum generated tokens per image.
        top_p, top_k, repetition_penalty: Sampling settings (used when
            ``temperature > 0``; repetition_penalty always).
        seed: Seeds torch before every generate call when sampling.
        quantization: ``None``, ``"4bit"`` or ``"8bit"`` (bitsandbytes; install
            ``geoai-vlm[quant]``).
        trust_remote_code: Run code shipped with the model. Off by default.
        revision: Model revision (branch, tag or commit sha) to load.
        system_prompt_mode: ``"auto"`` (default), ``"system"`` or ``"prepend"``.
        structured_output: Accepted for a uniform interface; this backend has
            no constrained decoding, so responses are parsed as free text and
            recorded as ``decoding_mode="unconstrained"``.
        json_schema: The response schema (used for validation upstream).
        processor_kwargs: Extra keyword arguments for ``AutoProcessor``.
        model_kwargs: Extra keyword arguments for the model ``from_pretrained``.
    """

    name = "transformers"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        *,
        dtype: Optional[str] = None,
        device_map: Optional[Any] = None,
        attn_implementation: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        quantization: Optional[str] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        system_prompt_mode: str = "auto",
        structured_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if quantization not in (None, "4bit", "8bit"):
            raise ValueError("quantization must be None, '4bit' or '8bit'")
        self._init_common(
            model_name,
            max_new_tokens=_merge_alias("max_new_tokens", max_new_tokens, "max_tokens", max_tokens, 2048),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            system_prompt_mode=system_prompt_mode,
            structured_output=structured_output,
            json_schema=json_schema,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        self.dtype = _merge_alias("dtype", dtype, "torch_dtype", torch_dtype, "auto")
        self.device_map = _merge_alias("device_map", device_map, "device", device, "auto")
        self.attn_implementation = attn_implementation
        self.quantization = quantization
        self.processor_kwargs = dict(processor_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})

        self.model = None
        self.processor = None

    # Older attribute names, kept readable.
    @property
    def device(self) -> Any:
        return self.device_map

    @property
    def torch_dtype(self) -> Any:
        return self.dtype

    def is_available(self) -> bool:
        """Check if Transformers is available."""
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self) -> None:
        """Load the processor and model (once)."""
        if self.model is not None:
            return

        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "The Transformers backend needs transformers and torch. Install them with: "
                "pip install 'geoai-vlm[transformers]' (or 'geoai-vlm[vlm]' for vLLM as well)"
            ) from exc

        print(f"Loading Transformers model: {self.model_name}")
        model_cls = _auto_vlm_class()
        common: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.revision:
            common["revision"] = self.revision

        load_kwargs: Dict[str, Any] = dict(common, device_map=self.device_map)
        load_kwargs.update(_dtype_load_kwarg(self.dtype))
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        if self.quantization:
            load_kwargs["quantization_config"] = _bnb_config(self.quantization, self.dtype)
        load_kwargs.update(self.model_kwargs)

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name, **common, **self.processor_kwargs
            )
            model = model_cls.from_pretrained(self.model_name, **load_kwargs)
        except ValueError as exc:
            if _needs_remote_code(exc) and not self.trust_remote_code:
                raise RemoteCodeRequiredError(self.model_name, "transformers") from exc
            if "Unrecognized configuration class" in str(exc):
                raise ValueError(
                    f"{self.model_name!r} is not registered as an image-text-to-text "
                    f"model in transformers {library_version('transformers')}. If its "
                    "model card says it ships custom code, review it and pass "
                    "trust_remote_code=True; otherwise try backend='vllm' or an "
                    "OpenAI-compatible server."
                ) from exc
            raise

        model.eval()
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            # Decoder-only generation must be left-padded, or every shorter
            # prompt in a batch would continue from padding tokens.
            tokenizer.padding_side = "left"
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
                tokenizer.pad_token = tokenizer.eos_token

        self.processor = processor
        self.model = model
        print(f"Model loaded on {getattr(model, 'device', 'unknown device')}")

    # -- chat template --------------------------------------------------
    def _probe_renderer(self):
        processor = self.processor
        if processor is None or getattr(processor, "chat_template", None) is None:
            return None

        def render(messages):
            return processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return render

    def _processor_takes_kwargs_dict(self) -> bool:
        import inspect

        try:
            params = inspect.signature(self.processor.apply_chat_template).parameters
        except (TypeError, ValueError):
            return False
        return "processor_kwargs" in params

    def _apply_chat_template(self, conversations: List[List[Dict[str, Any]]]):
        """Tokenise a batch of chats, images included, left-padded."""
        if getattr(self.processor, "chat_template", None) is None:
            raise ValueError(
                f"{self.model_name!r} has no chat template; geoai-vlm needs an "
                "instruction-tuned chat model."
            )
        kwargs: Dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        text_kwargs = {"padding": True, "padding_side": "left"}
        # transformers 5 takes processor arguments as one dict; 4.x took them
        # as keyword arguments.
        if self._processor_takes_kwargs_dict():
            kwargs["processor_kwargs"] = text_kwargs
        else:
            kwargs.update(text_kwargs)
        return self.processor.apply_chat_template(conversations, **kwargs)

    def _to_model(self, inputs):
        model = self.model
        if not hasattr(inputs, "to"):
            return inputs
        device = getattr(model, "device", None)
        dtype = getattr(model, "dtype", None)
        try:
            # BatchFeature casts only floating tensors (pixel values), never ids.
            return inputs.to(device, dtype=dtype) if dtype is not None else inputs.to(device)
        except TypeError:
            return inputs.to(device)

    def _generation_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature)
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
        else:
            kwargs["do_sample"] = False
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs

    def _generate_batch(self, conversations: List[List[Dict[str, Any]]]) -> List[str]:
        inputs = self._to_model(self._apply_chat_template(conversations))
        if self.seed is not None and self.temperature > 0:
            _manual_seed(self.seed)
        with _no_grad():
            output_ids = self.model.generate(**inputs, **self._generation_kwargs())
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_length:]
        texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def generate_outputs(
        self,
        images: Sequence[Any],
        system_prompt: str,
        user_prompt: str,
    ) -> List[GenerationOutput]:
        """Describe a batch of images in one generate call.

        An image that cannot be read becomes a per-item error; a failure of
        the model itself raises, since it would fail every item alike.
        """
        self.load_model()
        self._warn_unstructured()
        mode = self._system_mode_for(system_prompt)

        outputs: List[Optional[GenerationOutput]] = [None] * len(images)
        conversations, positions = [], []
        for i, image in enumerate(images):
            try:
                conversations.append(
                    build_chat_messages(user_prompt, image, system_prompt, mode=mode, image_style="hf")
                )
                positions.append(i)
            except Exception as exc:
                outputs[i] = GenerationOutput(
                    text="",
                    error=f"image could not be read: {type(exc).__name__}: {exc}",
                    decoding_mode="unconstrained",
                    system_prompt_mode=mode,
                )

        if conversations:
            texts = self._generate_batch(conversations)
            for i, text in zip(positions, texts):
                outputs[i] = GenerationOutput(
                    text=text, decoding_mode="unconstrained", system_prompt_mode=mode
                )
        return [o for o in outputs if o is not None]

    def generate(
        self,
        image_paths: List[str],
        system_prompt: str,
        user_prompt: str,
    ) -> List[str]:
        """Generate descriptions for a batch of images, in input order."""
        return [o.text for o in self.generate_outputs(image_paths, system_prompt, user_prompt)]

    def complete(self, messages: Sequence[Dict[str, Any]]) -> str:
        """Reply to a text-only chat with the loaded model.

        Same interface as :meth:`OpenAICompatibleBackend.complete`, so an
        application can use either for text questions.
        """
        self.load_model()
        return self._generate_batch([self._text_chat(messages)])[0]

    def generation_params(self) -> Dict[str, Any]:
        params = super().generation_params()
        params.update(dtype=self.dtype, quantization=self.quantization)
        return params

    def backend_version(self) -> Optional[str]:
        version = library_version("transformers")
        torch_version = library_version("torch")
        if version is None:
            return None
        return f"transformers {version}" + (f"; torch {torch_version}" if torch_version else "")


def _vllm_structured_kwargs(schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """SamplingParams keyword for JSON-schema decoding on the installed vLLM.

    ``structured_outputs=StructuredOutputsParams(json=...)`` is the API from
    vLLM 0.10.2 (verified on 0.13 and 0.31); older releases used
    ``guided_decoding=GuidedDecodingParams(json=...)``.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    except ImportError:
        return None


class VLLMBackend(_ChatBackend):
    """vLLM backend using the generic chat path.

    Messages go through ``LLM.chat()``, so each model is driven by its own
    chat template and multimodal processor -- no model-family helper package
    is involved. Images are passed in memory (``image_pil`` content parts).

    Args:
        model_name: Hugging Face model id or local path.
        gpu_memory_utilization: Fraction of GPU memory vLLM may reserve.
        tensor_parallel_size: GPUs for tensor parallelism (default: all).
        max_tokens: Older name for ``max_new_tokens``.
        temperature: ``0`` (default) decodes greedily.
        max_new_tokens: Maximum generated tokens per image.
        top_p, top_k, repetition_penalty: Sampling settings.
        seed: Engine seed (default 42, as before).
        dtype: Model dtype (``"auto"`` by default).
        quantization: vLLM quantization method (e.g. ``"awq"``, ``"fp8"``).
        max_model_len: Context length cap.
        max_num_seqs: Maximum concurrent sequences.
        limit_mm_per_prompt: Per-prompt multimodal limits; default one image.
        trust_remote_code: Run code shipped with the model. Off by default.
        revision: Model revision to load.
        enforce_eager: Disable CUDA graphs.
        system_prompt_mode: ``"auto"`` (default), ``"system"`` or ``"prepend"``.
        structured_output: Constrain decoding to ``json_schema`` when set.
        json_schema: The response JSON schema.
        llm_kwargs: Extra keyword arguments for ``vllm.LLM``.
    """

    name = "vllm"
    _image_style = "vllm_pil"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: Optional[int] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        *,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = 42,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        enforce_eager: bool = False,
        system_prompt_mode: str = "auto",
        structured_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._init_common(
            model_name,
            max_new_tokens=_merge_alias("max_new_tokens", max_new_tokens, "max_tokens", max_tokens, 2048),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            system_prompt_mode=system_prompt_mode,
            structured_output=structured_output,
            json_schema=json_schema,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.limit_mm_per_prompt = dict(limit_mm_per_prompt) if limit_mm_per_prompt else {"image": 1}
        self.enforce_eager = enforce_eager
        self.llm_kwargs = dict(llm_kwargs or {})

        self.llm = None
        self.processor = None  # Hugging Face processor, loaded only to probe the template
        self.sampling_params = None
        self._structured_kwargs: Any = _UNSET

    def is_available(self) -> bool:
        """Check if VLLM is available."""
        try:
            import vllm  # noqa: F401
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _engine_args(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "model": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size or max(1, _cuda_device_count()),
            "enforce_eager": self.enforce_eager,
            "dtype": self.dtype,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
        }
        if self.seed is not None:
            args["seed"] = self.seed
        for key in ("revision", "quantization", "max_model_len", "max_num_seqs"):
            value = getattr(self, key)
            if value is not None:
                args[key] = value
        args.update(self.llm_kwargs)
        return args

    def load_model(self) -> None:
        """Start the vLLM engine (once)."""
        if self.llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "The vLLM backend needs vLLM (Linux with an NVIDIA GPU). Install it with: "
                "pip install 'geoai-vlm[vlm]', or use backend='transformers' or 'openai'"
            ) from exc

        # vLLM must spawn its workers when CUDA is already initialised; keep a
        # caller's explicit choice.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        args = self._engine_args()
        print(f"Loading VLLM model: {self.model_name}")
        try:
            self.llm = LLM(**args)
        except Exception as exc:
            if _needs_remote_code(exc) and not self.trust_remote_code:
                raise RemoteCodeRequiredError(self.model_name, "vllm") from exc
            raise
        self.sampling_params, _ = self._sampling_params()
        print(f"Model loaded on {args['tensor_parallel_size']} GPU(s)")

    def _probe_renderer(self):
        """Render with the model's Hugging Face processor template, if loadable."""
        try:
            from transformers import AutoProcessor

            if self.processor is None:
                kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
                if self.revision:
                    kwargs["revision"] = self.revision
                self.processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        except Exception as exc:
            logger.info("chat template not inspectable for %s: %s", self.model_name, type(exc).__name__)
            return None
        processor = self.processor
        if getattr(processor, "chat_template", None) is None:
            return None
        return lambda messages: processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sampling_params(self):
        from vllm import SamplingParams

        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty

        decoding_mode = "unconstrained"
        if self.wants_structured:
            if self._structured_kwargs is _UNSET:
                self._structured_kwargs = _vllm_structured_kwargs(self.json_schema)
            if self._structured_kwargs:
                kwargs.update(self._structured_kwargs)
                decoding_mode = "json_schema"
            else:
                self._warn_unstructured()
        return SamplingParams(**kwargs), decoding_mode

    def _conversations(self, images, system_prompt, user_prompt, mode):
        outputs: List[Optional[GenerationOutput]] = [None] * len(images)
        conversations, positions = [], []
        for i, image in enumerate(images):
            try:
                conversations.append(
                    build_chat_messages(
                        user_prompt, image, system_prompt, mode=mode, image_style=self._image_style
                    )
                )
                positions.append(i)
            except Exception as exc:
                outputs[i] = GenerationOutput(
                    text="",
                    error=f"image could not be read: {type(exc).__name__}: {exc}",
                    system_prompt_mode=mode,
                )
        return outputs, conversations, positions

    def generate_outputs(
        self,
        images: Sequence[Any],
        system_prompt: str,
        user_prompt: str,
    ) -> List[GenerationOutput]:
        """Describe a batch of images with one ``LLM.chat`` call."""
        self.load_model()
        mode = self._system_mode_for(system_prompt)
        sampling, decoding_mode = self._sampling_params()
        outputs, conversations, positions = self._conversations(images, system_prompt, user_prompt, mode)

        if conversations:
            try:
                results = self.llm.chat(conversations, sampling_params=sampling, use_tqdm=False)
            except Exception as exc:
                retry = (
                    self.system_prompt_mode == "auto"
                    and mode == "system"
                    and _looks_like_system_role_error(exc)
                )
                if not retry:
                    raise
                self._fall_back_to_prepend(
                    system_prompt, f"chat template rejected the system role ({type(exc).__name__}: {exc})"
                )
                mode = "prepend"
                outputs, conversations, positions = self._conversations(
                    images, system_prompt, user_prompt, mode
                )
                results = self.llm.chat(conversations, sampling_params=sampling, use_tqdm=False)

            for i, result in zip(positions, results):
                outputs[i] = GenerationOutput(
                    text=result.outputs[0].text,
                    decoding_mode=decoding_mode,
                    system_prompt_mode=mode,
                )

        for out in outputs:
            if out is not None and out.decoding_mode is None:
                out.decoding_mode = decoding_mode
        return [o for o in outputs if o is not None]

    def generate(
        self,
        image_paths: List[str],
        system_prompt: str,
        user_prompt: str,
    ) -> List[str]:
        """Generate descriptions for a batch of images, in input order."""
        return [o.text for o in self.generate_outputs(image_paths, system_prompt, user_prompt)]

    def complete(self, messages: Sequence[Dict[str, Any]]) -> str:
        """Reply to a text-only chat (same interface as the HTTP backend)."""
        self.load_model()
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
        result = self.llm.chat([self._text_chat(messages)], sampling_params=sampling, use_tqdm=False)
        return result[0].outputs[0].text

    def generation_params(self) -> Dict[str, Any]:
        params = super().generation_params()
        params.update(dtype=self.dtype, quantization=self.quantization)
        return params

    def backend_version(self) -> Optional[str]:
        version = library_version("vllm")
        return f"vllm {version}" if version else None


# =============================================================================
# Describer
# =============================================================================
_BACKEND_ALIASES = {
    "openai": "openai",
    "openai_compatible": "openai",
    "openai-compatible": "openai",
}


def _validation_issues(parsed: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> Optional[str]:
    if not schema or "error" in parsed:
        return None
    return json.dumps(validate_json(parsed, schema), ensure_ascii=False)


class ImageDescriber:
    """
    VLM-based image describer with model-agnostic backends.

    Args:
        model_name: Hugging Face model id (default: Qwen/Qwen3-VL-2B-Instruct),
            or the model name an OpenAI-compatible server knows it by.
        backend: ``"auto"`` (vLLM with a GPU, else Transformers), ``"vllm"``,
            ``"transformers"``, ``"openai"``, or a backend instance (anything
            with a ``generate()`` method). An instance is used as configured,
            so it can be shared, e.g. by two describers with different
            templates.
        prompt_template: Prompt template name (see
            :func:`~geoai_vlm.prompts.get_prompt_template`) or None for custom.
        system_prompt: Custom system prompt (overrides template)
        user_prompt: Custom user prompt (overrides template)
        system_prompt_mode: ``"auto"`` (default) sends a system turn when the
            model's chat template accepts it and otherwise prepends the
            system text to the user text; ``"system"`` or ``"prepend"``
            force one form. The form used is recorded per description.
        structured_output: Constrain decoding to the template's JSON schema
            where the backend supports it (vLLM, OpenAI-compatible servers);
            otherwise fall back to parsing. Recorded as ``decoding_mode``.
        json_schema: Schema for custom prompts (templates bring their own).
        trust_remote_code: Allow code shipped with a model to run. Off by
            default; a model that needs it raises
            :class:`RemoteCodeRequiredError`.
        revision: Model revision (branch, tag or commit) to load.
        **backend_kwargs: Additional kwargs passed to the backend, e.g.
            ``dtype``, ``device_map``, ``quantization``, ``max_new_tokens``,
            ``temperature``; or ``base_url`` / ``api_key_env`` for
            ``backend="openai"``.

    Example:
        >>> describer = ImageDescriber(model_name="Qwen/Qwen3-VL-2B-Instruct")
        >>> results = describer.describe("./images", batch_size=8)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        backend: Union[str, BaseBackend] = "auto",
        prompt_template: Optional[str] = "geoai",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        *,
        system_prompt_mode: str = "auto",
        structured_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        **backend_kwargs,
    ):
        if system_prompt_mode not in SYSTEM_PROMPT_MODES:
            raise ValueError(
                f"system_prompt_mode must be one of {SYSTEM_PROMPT_MODES}, "
                f"got {system_prompt_mode!r}"
            )
        self.model_name = model_name
        self.backend_kwargs = backend_kwargs
        self.system_prompt_mode = system_prompt_mode
        self.structured_output = bool(structured_output)
        self.trust_remote_code = bool(trust_remote_code)
        self.revision = revision

        # Initialize backend
        self._backend: Optional[BaseBackend] = None
        if isinstance(backend, BaseBackend) or (
            not isinstance(backend, str) and callable(getattr(backend, "generate", None))
        ):
            # A backend instance (built-in, subclass, or any object with a
            # generate() method), e.g. one shared by several describers.
            self._backend = backend
            self.backend_name = getattr(backend, "name", type(backend).__name__)
        else:
            self.backend_name = backend

        # Set up prompts
        template = None
        if prompt_template and system_prompt is None and user_prompt is None:
            template = get_prompt_template(prompt_template)
            self.system_prompt = template["system"]
            self.user_prompt = template["user"]
            self.prompt_template = prompt_template
        else:
            self.system_prompt = system_prompt or GEOAI_SYSTEM_PROMPT
            self.user_prompt = user_prompt or GEOAI_USER_PROMPT
            self.prompt_template = None

        self.json_schema = json_schema
        if self.json_schema is None and template is not None:
            self.json_schema = template.get("json_schema")
        self._flatten: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = (
            template.get("flatten") if template is not None else None
        )
        if self.structured_output and not self.json_schema:
            warnings.warn(
                "structured_output=True but no JSON schema is known for these "
                "prompts; pass json_schema=... (responses will be parsed as free text).",
                stacklevel=2,
            )

    def _backend_options(self) -> Dict[str, Any]:
        options = {
            "system_prompt_mode": self.system_prompt_mode,
            "structured_output": self.structured_output,
            "json_schema": self.json_schema,
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
        }
        options.update(self.backend_kwargs)
        return options

    @property
    def backend(self) -> BaseBackend:
        """Get or initialize the backend."""
        if self._backend is not None:
            return self._backend

        name = _BACKEND_ALIASES.get(self.backend_name, self.backend_name)
        options = self._backend_options()
        if name == "auto":
            # Try VLLM first, fall back to Transformers
            vllm_backend = VLLMBackend(self.model_name, **options)
            if vllm_backend.is_available():
                print("Using VLLM backend")
                self._backend = vllm_backend
            else:
                print("VLLM not available, falling back to Transformers")
                self._backend = TransformersBackend(self.model_name, **options)
        elif name == "vllm":
            self._backend = VLLMBackend(self.model_name, **options)
        elif name == "transformers":
            self._backend = TransformersBackend(self.model_name, **options)
        elif name == "openai":
            from .openai_compat import OpenAICompatibleBackend

            self._backend = OpenAICompatibleBackend(self.model_name, **options)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")

        return self._backend

    # -- provenance -------------------------------------------------------
    @property
    def prompt_version(self) -> str:
        """Stable short hash of the prompt pair currently configured."""
        payload = json.dumps(
            {"system": self.system_prompt, "user": self.user_prompt},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def provenance(self) -> Dict[str, Any]:
        """Provenance shared by every record of a run, ``processing_id`` included.

        Generation settings that belong to the describer (system-prompt mode,
        structured output and its schema) are merged over the backend's own,
        so they are covered even by a custom backend that reports nothing.
        """
        backend = self.backend
        info_fn = getattr(backend, "provenance", None)
        info = dict(info_fn()) if callable(info_fn) else {}
        params = dict(info.get("generation_params") or {})
        params.update(
            system_prompt_mode=self.system_prompt_mode,
            structured_output=self.structured_output or None,
            json_schema_digest=schema_digest(self.json_schema) if self.structured_output else None,
        )
        canonical = canonical_generation_params(params)
        revision = info.get("model_revision")
        return {
            "backend": info.get("backend") or getattr(backend, "name", type(backend).__name__),
            "backend_version": info.get("backend_version"),
            "model_revision": revision,
            "generation_params": canonical,
            "processing_id": compute_processing_id(
                self.model_name, self.prompt_version, revision, canonical
            ),
        }

    @property
    def processing_id(self) -> str:
        """Identity of *this* derived output.

        Covers the model, its resolved revision, the prompt version and every
        output-affecting generation setting. Resume decisions key on this
        rather than the image id alone, so re-running the same images under a
        different model, revision, prompt or decoding configuration produces
        a new derived record instead of being skipped as already finished.
        """
        return self.provenance()["processing_id"]

    def describe(
        self,
        image_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 8,
        resume: bool = True,
        image_extensions: List[str] = None,
        recursive: bool = True,
        image_paths: Optional[List[Union[str, Path]]] = None,
    ) -> pd.DataFrame:
        """
        Describe a set of images.

        Args:
            image_dir: Directory to scan for images. Ignored when *image_paths*
                is given.
            output_path: Path to save results (Parquet). If None, returns
                without saving.
            batch_size: Number of images to process per batch
            resume: If True, skip images already processed *by this same
                derived-output configuration* (see :attr:`processing_id`).
                Failed records are always retried.
            image_extensions: Extensions to scan for (default: .jpg/.jpeg/.png)
            recursive: If True, search subdirectories recursively
            image_paths: Explicit list of images to describe. Use this to
                guarantee that only a selected subset is processed, rather than
                everything that happens to sit in *image_dir*.

        Returns:
            DataFrame with image paths, IDs, raw responses, parsed JSON fields
            and provenance columns (see :data:`DESCRIPTION_COLUMNS`).
        """
        if image_paths is not None:
            paths = [Path(p) for p in image_paths]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} requested image(s) do not exist, "
                    f"first: {missing[0]}"
                )
            image_paths_list = sorted(set(paths))
            print(f"Describing {len(image_paths_list)} selected images")
        elif image_dir is not None:
            image_dir = Path(image_dir)
            extensions = image_extensions or [".jpg", ".jpeg", ".png"]

            collected = []
            pattern_prefix = "**/" if recursive else ""
            for ext in extensions:
                collected.extend(image_dir.glob(f"{pattern_prefix}*{ext}"))
                collected.extend(image_dir.glob(f"{pattern_prefix}*{ext.upper()}"))

            image_paths_list = sorted(set(collected))
            print(f"Found {len(image_paths_list)} images in {image_dir}")
        else:
            raise ValueError("Provide either image_dir or image_paths")

        if len(image_paths_list) == 0:
            return pd.DataFrame()

        run = self.provenance()
        processing_id = run["processing_id"]

        # -- resume ---------------------------------------------------------
        # Only records produced by this same derived-output configuration, and
        # which actually succeeded, count as finished work.
        existing_df = None
        processed_ids = set()
        if output_path and Path(output_path).exists():
            existing_df = pd.read_parquet(output_path)
            if resume:
                done = existing_df
                if "processing_id" in done.columns:
                    done = done[done["processing_id"] == processing_id]
                    self._report_other_configurations(existing_df, processing_id)
                else:
                    # Output written before provenance existed cannot be
                    # attributed to any model or prompt, so it cannot be
                    # claimed as this run's finished work.
                    print(
                        f"Existing output {output_path} predates processing "
                        "provenance (no processing_id column); re-describing "
                        "all images rather than assuming they match this run."
                    )
                    done = done.iloc[0:0]
                if "parse_error" in done.columns:
                    done = done[~done["parse_error"].astype(bool)]
                processed_ids = set(done["image_id"].astype(str).tolist())
                print(f"Resuming: {len(processed_ids)} images already processed")

        pending = [p for p in image_paths_list if p.stem not in processed_ids]
        print(f"Images to process: {len(pending)}")

        if len(pending) == 0:
            print("All images already processed!")
            if existing_df is not None:
                return existing_df
            return pd.DataFrame()

        # -- process --------------------------------------------------------
        all_results: List[Dict[str, Any]] = []

        for batch_idx in tqdm(range(0, len(pending), batch_size), desc="Processing"):
            batch_paths = pending[batch_idx: batch_idx + batch_size]
            outputs = self._generate(batch_paths)

            for img_path, output in zip(batch_paths, outputs):
                all_results.append(self._build_record(img_path, output, run))

            if output_path:
                batch_df = pd.DataFrame(all_results[-len(batch_paths):])
                existing_df = _upsert_records(existing_df, batch_df)
                existing_df.to_parquet(output_path, index=False)

        if output_path and existing_df is not None:
            return existing_df

        return pd.DataFrame(all_results)

    def _report_other_configurations(self, existing: pd.DataFrame, processing_id: str) -> None:
        if "model_name" not in existing.columns or "prompt_version" not in existing.columns:
            return
        same_model_prompt = existing[
            (existing["model_name"] == self.model_name)
            & (existing["prompt_version"] == self.prompt_version)
            & (existing["processing_id"] != processing_id)
        ]
        if len(same_model_prompt):
            print(
                f"Note: {len(same_model_prompt)} existing record(s) used this model "
                "and prompt under a different revision or generation settings. "
                "They are kept as separate records and not counted as finished."
            )

    def _generate(self, images: Sequence[Any]) -> List[GenerationOutput]:
        backend = self.backend
        if hasattr(backend, "generate_outputs"):
            outputs = backend.generate_outputs(list(images), self.system_prompt, self.user_prompt)
        else:  # a duck-typed backend that only implements generate()
            texts = backend.generate([str(p) for p in images], self.system_prompt, self.user_prompt)
            outputs = [GenerationOutput(text=t) for t in texts]
        if len(outputs) != len(images):
            raise RuntimeError(
                f"backend returned {len(outputs)} responses for {len(images)} images"
            )
        return outputs

    def _build_record(
        self,
        img_path: Path,
        response: Union[str, GenerationOutput],
        run: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Turn one raw model response into a fully-provenanced record."""
        output = response if isinstance(response, GenerationOutput) else GenerationOutput(text=response)
        run = run if run is not None else self.provenance()

        if output.error is not None:
            parsed: Dict[str, Any] = {"error": f"generation failed: {output.error}"}
        else:
            parsed = parse_json_response(output.text)
        failed = "error" in parsed

        record: Dict[str, Any] = {
            "image_path": str(img_path),
            "image_id": img_path.stem,
            "raw_response": output.text,
            "parsed_json": json.dumps(parsed),
            "parse_error": failed,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "processing_id": run["processing_id"],
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        record.update(extract_summary_fields(parsed))
        record.update(
            {
                "backend": run.get("backend"),
                "backend_version": run.get("backend_version"),
                "model_revision": run.get("model_revision"),
                "generation_params": json.dumps(run.get("generation_params") or {}, sort_keys=True),
                "system_prompt_mode_effective": output.system_prompt_mode,
                "decoding_mode": output.decoding_mode or "unconstrained",
                "generation_error": output.error,
                "validation_issues": _validation_issues(parsed, self.json_schema),
            }
        )
        if self._flatten is not None:
            record.update(self._flatten(parsed))
        return record

    def describe_single(self, image_path: Union[str, Path, Any]) -> Dict[str, Any]:
        """
        Describe a single image.

        Args:
            image_path: Path to the image (built-in backends also accept a PIL
                image or raw bytes)

        Returns:
            Parsed JSON dictionary with description
        """
        output = self._generate([image_path])[0]
        if output.error is not None:
            return {"error": f"generation failed: {output.error}"}
        return parse_json_response(output.text)


def __getattr__(name: str):  # PEP 562: re-export without an import cycle
    if name == "OpenAICompatibleBackend":
        from .openai_compat import OpenAICompatibleBackend

        return OpenAICompatibleBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
