# -*- coding: utf-8 -*-
"""
Model compatibility checks for GeoAI-VLM
========================================
:func:`check_model` runs one model through a single-image dry run and reports
what actually happened: whether it loaded, whether it has a chat template,
whether its template accepts a system turn, whether its answer parsed as
JSON, and how long and how much memory that took.

The image is a synthetic drawing (:func:`~geoai_vlm.chat.make_synthetic_street_image`)
unless you pass your own, so a check needs no third-party imagery.

A passing check means the *software path* works for that model. It says
nothing about the quality of the descriptions.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .chat import make_synthetic_street_image


__all__ = ["ModelCheckReport", "check_model"]


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def _gpu_peak_mb() -> Optional[float]:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)


def _hardware() -> str:
    parts = [platform.machine() or "unknown-arch", platform.system()]
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(torch.cuda.get_device_name(0))
        else:
            parts.append("CPU only")
    except ImportError:
        parts.append("torch not installed")
    return ", ".join(parts)


@dataclass
class ModelCheckReport:
    """Outcome of :func:`check_model`.

    ``status`` is ``"ok"`` (loaded, generated, parsed as JSON), ``"partial"``
    (generated but the answer did not parse) or ``"failed"``.
    """

    model_id: str
    backend: str
    status: str = "failed"
    backend_version: Optional[str] = None
    model_revision: Optional[str] = None
    hardware: Optional[str] = None
    loaded: bool = False
    load_seconds: Optional[float] = None
    chat_template: Optional[bool] = None
    system_role: Optional[str] = None
    system_prompt_mode_effective: Optional[str] = None
    system_prompt_note: Optional[str] = None
    decoding_mode: Optional[str] = None
    generated: bool = False
    generate_seconds: Optional[float] = None
    json_parsed: Optional[bool] = None
    parse_error: Optional[str] = None
    validation_issues: List[str] = field(default_factory=list)
    response_excerpt: Optional[str] = None
    peak_rss_mb: Optional[float] = None
    gpu_peak_memory_mb: Optional[float] = None
    image: str = "synthetic"
    prompt_template: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable report."""
        rows = [
            ("model", self.model_id),
            ("backend", f"{self.backend} ({self.backend_version or 'version unknown'})"),
            ("revision", self.model_revision or "unknown"),
            ("hardware", self.hardware or "unknown"),
            ("status", self.status),
            ("loaded", f"{self.loaded} ({self.load_seconds}s)" if self.load_seconds is not None else str(self.loaded)),
            ("chat template", "not inspectable" if self.chat_template is None else str(self.chat_template)),
            ("system role", self.system_role or "unknown"),
            ("system prompt sent as", self.system_prompt_mode_effective or "unknown"),
            ("decoding", self.decoding_mode or "unknown"),
            (
                "generated",
                f"{self.generated} ({self.generate_seconds}s)" if self.generate_seconds is not None else str(self.generated),
            ),
            ("JSON parsed", "unknown" if self.json_parsed is None else str(self.json_parsed)),
            ("schema issues", str(len(self.validation_issues))),
            ("peak RSS (MB)", str(self.peak_rss_mb)),
            ("peak GPU (MB)", str(self.gpu_peak_memory_mb)),
            ("image", self.image),
        ]
        width = max(len(k) for k, _ in rows)
        lines = [f"{k.ljust(width)} : {v}" for k, v in rows]
        if self.system_prompt_note:
            lines.append(f"{'note'.ljust(width)} : {self.system_prompt_note}")
        if self.parse_error:
            lines.append(f"{'parse error'.ljust(width)} : {self.parse_error}")
        for err in self.errors:
            lines.append(f"{'error'.ljust(width)} : {err}")
        if self.response_excerpt:
            lines.append(f"{'response'.ljust(width)} : {self.response_excerpt}")
        return "\n".join(lines)


def check_model(
    model_id: str,
    backend: str = "transformers",
    *,
    prompt_template: str = "simple",
    image: Any = None,
    max_new_tokens: int = 256,
    system_prompt_mode: str = "auto",
    structured_output: bool = False,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **backend_kwargs: Any,
) -> ModelCheckReport:
    """Dry-run one model on one small image and report what happened.

    Args:
        model_id: Hugging Face model id or local path (for ``backend="openai"``,
            the model name the server knows).
        backend: ``"transformers"`` (default), ``"vllm"`` or ``"openai"``.
        prompt_template: Template to test with; ``"simple"`` keeps the answer
            short. Use ``"geoai"`` or ``"active_mobility_audit_v1"`` to test
            the long schemas.
        image: Path or PIL image; a synthetic drawing by default.
        max_new_tokens: Generation cap for the dry run.
        system_prompt_mode: ``"auto"``, ``"system"`` or ``"prepend"``.
        structured_output: Request JSON-schema constrained decoding.
        trust_remote_code: Allow code shipped with the model to run.
        revision: Model revision to load.
        **backend_kwargs: Passed to the backend (``dtype``, ``device_map``,
            ``base_url``, ``api_key_env``, ...).

    Returns:
        A :class:`ModelCheckReport`. Exceptions are captured in ``errors``
        rather than raised, so a batch of checks can run to completion.
    """
    from .describer import ImageDescriber, parse_json_response
    from .schemas import validate_json

    report = ModelCheckReport(
        model_id=model_id,
        backend=backend,
        hardware=_hardware(),
        prompt_template=prompt_template,
        image="synthetic" if image is None else str(getattr(image, "filename", None) or image),
    )

    try:
        describer = ImageDescriber(
            model_name=model_id,
            backend=backend,
            prompt_template=prompt_template,
            system_prompt_mode=system_prompt_mode,
            structured_output=structured_output,
            trust_remote_code=trust_remote_code,
            revision=revision,
            max_new_tokens=max_new_tokens,
            **backend_kwargs,
        )
        engine = describer.backend
        report.backend = getattr(engine, "name", backend)
    except Exception as exc:
        report.errors.append(f"setup: {type(exc).__name__}: {exc}")
        report.peak_rss_mb = _peak_rss_mb()
        return report

    # -- load -------------------------------------------------------------
    start = time.perf_counter()
    try:
        engine.load_model()
        report.loaded = True
    except Exception as exc:
        report.errors.append(f"load: {type(exc).__name__}: {exc}")
    report.load_seconds = round(time.perf_counter() - start, 2)

    version_fn = getattr(engine, "backend_version", None)
    report.backend_version = version_fn() if callable(version_fn) else None
    try:
        report.model_revision = getattr(engine, "model_revision", None)
    except Exception:  # pragma: no cover - network edge cases
        report.model_revision = None

    if not report.loaded:
        report.peak_rss_mb = _peak_rss_mb()
        report.gpu_peak_memory_mb = _gpu_peak_mb()
        return report

    # -- chat template and system role -----------------------------------
    renderer = getattr(engine, "_probe_renderer", lambda: None)()
    if report.backend == "openai":
        report.chat_template = None
        try:
            served = engine.list_models()
            if model_id not in served:
                report.errors.append(f"server does not list {model_id!r}; it lists {served}")
        except Exception as exc:
            report.errors.append(f"list models: {type(exc).__name__}: {exc}")
    else:
        report.chat_template = renderer is not None

    try:
        mode = engine._system_mode_for(describer.system_prompt)
        report.system_prompt_mode_effective = mode
        report.system_prompt_note = engine.system_prompt_mode_reason(describer.system_prompt)
    except Exception as exc:
        report.errors.append(f"system role probe: {type(exc).__name__}: {exc}")
        mode = None

    # -- one generation ---------------------------------------------------
    image = image if image is not None else make_synthetic_street_image()
    start = time.perf_counter()
    try:
        output = describer._generate([image])[0]
        report.generate_seconds = round(time.perf_counter() - start, 2)
        report.decoding_mode = output.decoding_mode
        report.system_prompt_mode_effective = output.system_prompt_mode or mode
        report.system_prompt_note = engine.system_prompt_mode_reason(describer.system_prompt)
        if output.error:
            report.errors.append(f"generate: {output.error}")
        else:
            report.generated = True
            text = output.text or ""
            report.response_excerpt = text[:300] + ("..." if len(text) > 300 else "")
            parsed = parse_json_response(text)
            report.json_parsed = "error" not in parsed
            if not report.json_parsed:
                report.parse_error = parsed.get("error")
            elif describer.json_schema:
                report.validation_issues = validate_json(parsed, describer.json_schema)
    except Exception as exc:
        report.generate_seconds = round(time.perf_counter() - start, 2)
        report.errors.append(f"generate: {type(exc).__name__}: {exc}")

    eff = report.system_prompt_mode_effective
    note = report.system_prompt_note or ""
    if eff == "system" and not note:
        report.system_role = "supported"
    elif eff == "system":
        report.system_role = "not verified (template not inspectable)"
    elif eff == "prepend" and system_prompt_mode == "auto":
        report.system_role = "not supported (prepended to user turn)"
    elif eff == "prepend":
        report.system_role = "not tested (prepend forced)"

    report.peak_rss_mb = _peak_rss_mb()
    report.gpu_peak_memory_mb = _gpu_peak_mb()
    if report.generated and report.json_parsed:
        report.status = "ok"
    elif report.generated:
        report.status = "partial"
    return report


def _report_json(report: ModelCheckReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
