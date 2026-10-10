# -*- coding: utf-8 -*-
"""
Command-line interface for GeoAI-VLM
====================================

    geoai-vlm check-model HuggingFaceTB/SmolVLM-256M-Instruct
    geoai-vlm check-model Qwen/Qwen3-VL-2B-Instruct --backend vllm --template geoai
    geoai-vlm check-model my-model --backend openai --base-url http://localhost:8000/v1

Exit status: 0 when the check is ``ok``, 1 when ``partial``, 2 when ``failed``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geoai-vlm", description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check-model",
        help="dry-run one model on one small image and report compatibility",
    )
    check.add_argument("model_id", help="Hugging Face model id, local path, or served model name")
    check.add_argument(
        "--backend", default="transformers", choices=["transformers", "vllm", "openai"]
    )
    check.add_argument("--template", default="simple", help="prompt template (default: simple)")
    check.add_argument("--image", default=None, help="image path (default: a synthetic drawing)")
    check.add_argument("--max-new-tokens", type=int, default=256)
    check.add_argument(
        "--system-prompt-mode", default="auto", choices=["auto", "system", "prepend"]
    )
    check.add_argument("--structured-output", action="store_true")
    check.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow code shipped with the model to run (off by default)",
    )
    check.add_argument("--revision", default=None)
    check.add_argument("--dtype", default=None, help="e.g. bfloat16, float32 (local backends)")
    check.add_argument("--device-map", default=None, help="transformers device_map")
    check.add_argument("--base-url", default=None, help="OpenAI-compatible server URL")
    check.add_argument(
        "--api-key-env",
        default=None,
        help="NAME of the environment variable holding the API key (never the key itself)",
    )
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the ``geoai-vlm`` console script."""
    args = _build_parser().parse_args(argv)

    if args.command == "check-model":
        from .models import _report_json, check_model

        backend_kwargs = {}
        if args.dtype:
            backend_kwargs["dtype"] = args.dtype
        if args.device_map:
            backend_kwargs["device_map"] = args.device_map
        if args.backend == "openai":
            if args.base_url:
                backend_kwargs["base_url"] = args.base_url
            if args.api_key_env:
                backend_kwargs["api_key_env"] = args.api_key_env

        # Backends print progress; keep stdout clean so --json is parseable.
        redirect = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
        with redirect:
            report = check_model(
                args.model_id,
                backend=args.backend,
                prompt_template=args.template,
                image=args.image,
                max_new_tokens=args.max_new_tokens,
                system_prompt_mode=args.system_prompt_mode,
                structured_output=args.structured_output,
                trust_remote_code=args.trust_remote_code,
                revision=args.revision,
                **backend_kwargs,
            )
        print(_report_json(report) if args.json else report.summary())
        return {"ok": 0, "partial": 1}.get(report.status, 2)

    return 2  # pragma: no cover - argparse enforces a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
