# -*- coding: utf-8 -*-
"""
Provenance helpers for GeoAI-VLM
================================
What a derived record needs to say about how it was produced.

``processing_id`` identifies one *derived output*: the same images described
by the same model revision, prompt and generation settings. Resume and upsert
key on ``(image_id, processing_id)``, so anything that can change the model's
text must be part of it, and nothing that cannot (batch size, device,
timeouts, concurrency) may be.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


__all__ = [
    "OUTPUT_AFFECTING_PARAMS",
    "canonical_generation_params",
    "compute_processing_id",
    "library_version",
    "resolve_model_revision",
    "schema_digest",
]

logger = logging.getLogger(__name__)

#: Generation settings that can change a model's output, under one canonical
#: name each regardless of backend. Everything else a backend accepts
#: (device placement, batch size, memory limits, timeouts, concurrency) is
#: execution detail and is deliberately excluded from ``processing_id``.
OUTPUT_AFFECTING_PARAMS = (
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "repetition_penalty",
    "dtype",
    "quantization",
    "system_prompt_mode",
    "structured_output",
    "json_schema_digest",
    "image_max_side",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _normalise(value: Any) -> Any:
    if isinstance(value, float):
        # 0.7 and 0.70000001 must not produce different ids.
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items())}
    return value


#: Settings that only matter when sampling. Under greedy decoding
#: (``temperature == 0``) they cannot change the output, so they are dropped:
#: otherwise two backends with different default seeds would never share a
#: ``processing_id`` for identical greedy runs.
SAMPLING_ONLY_PARAMS = ("seed", "top_p", "top_k")


def canonical_generation_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only output-affecting settings, drop ``None``, normalise floats.

    Unknown keys are ignored rather than rejected, so a backend can pass its
    full configuration and only the parts that matter are hashed. Sampling-only
    settings are dropped when ``temperature`` is exactly zero.
    """
    greedy = params.get("temperature") is not None and float(params["temperature"]) == 0.0
    out: Dict[str, Any] = {}
    for key in OUTPUT_AFFECTING_PARAMS:
        value = params.get(key)
        if value is None or (greedy and key in SAMPLING_ONLY_PARAMS):
            continue
        out[key] = _normalise(value)
    return out


def schema_digest(schema: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Short stable hash of a JSON schema (``None`` when there is no schema)."""
    if not schema:
        return None
    payload = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def compute_processing_id(
    model_name: str,
    prompt_version: str,
    model_revision: Optional[str] = None,
    generation_params: Optional[Mapping[str, Any]] = None,
    backend: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    """Identity of one derived output.

    Covers the model id, its resolved revision, the prompt version, the
    output-affecting generation settings and the backend that ran the model.
    For an HTTP server the endpoint counts too, because a served model name is
    only a label: two servers can answer to the same name with different
    weights. Two runs share an id only when all of these match.
    """
    payload = json.dumps(
        {
            "model": model_name,
            "model_revision": model_revision,
            "prompt": prompt_version,
            "generation": canonical_generation_params(generation_params or {}),
            "backend": backend,
            "endpoint": endpoint,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def library_version(distribution: str) -> Optional[str]:
    """Installed version of *distribution*, or ``None`` if it is absent."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _revision_from_cache(model_id: str, revision: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        cached = try_to_load_from_cache(model_id, "config.json", revision=revision or "main")
    except Exception:  # pragma: no cover - depends on cache layout
        return None
    if not isinstance(cached, str):
        return None
    # .../models--org--name/snapshots/<commit_sha>/config.json
    parts = Path(cached).parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts) and _SHA_RE.match(parts[idx + 1]):
            return parts[idx + 1]
    return None


def resolve_model_revision(
    model_id: str,
    revision: Optional[str] = None,
    timeout: float = 10.0,
) -> Optional[str]:
    """Resolve the Hugging Face commit sha that *model_id* refers to.

    Order: an explicit 40-character sha is returned as is; otherwise the Hub
    is asked (unless ``HF_HUB_OFFLINE`` is set); if that fails, the local
    cache snapshot is used. A local directory, a missing ``huggingface_hub``
    or no network and no cache all give ``None`` -- never a guess.
    """
    if revision and _SHA_RE.match(revision):
        return revision
    if not model_id or Path(model_id).expanduser().is_dir():
        return None

    if not _offline():
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(model_id, revision=revision, timeout=timeout)
            sha = getattr(info, "sha", None)
            if sha:
                return sha
        except Exception as exc:  # network, auth, unknown repo, missing package
            logger.debug("could not resolve revision of %s from the Hub: %s", model_id, type(exc).__name__)

    return _revision_from_cache(model_id, revision)
