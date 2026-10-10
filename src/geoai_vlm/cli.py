# -*- coding: utf-8 -*-
"""
Command-line interface for GeoAI-VLM
====================================

    geoai-vlm check-model HuggingFaceTB/SmolVLM-256M-Instruct
    geoai-vlm check-model Qwen/Qwen3-VL-2B-Instruct --backend vllm --template geoai
    geoai-vlm check-model my-model --backend openai --base-url http://localhost:8000/v1
    geoai-vlm build-index descriptions.parquet --out ./demo_index
    geoai-vlm app --index ./demo_index --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct

check-model exits with 0 when the check is ``ok``, 1 when ``partial``, 2 when
``failed``.
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

    index = sub.add_parser(
        "build-index",
        help="embed a table of described scenes into a small local SceneIndex",
    )
    index.add_argument("table", help="(Geo)Parquet with image_id, lat/lon or geometry, and a text column")
    index.add_argument("--out", required=True, help="output directory")
    index.add_argument("--text-column", default="scene_narrative")
    index.add_argument("--embedding-backend", default="clip", choices=["clip", "transformers", "vllm", "auto"])
    index.add_argument("--embedding-model", default=None)
    index.add_argument("--limit", type=int, default=None, help="use only the first N rows")

    app = sub.add_parser("app", help="launch the local research demo (needs geoai-vlm[app])")
    app.add_argument("--index", default=None, help="SceneIndex directory (from build-index)")
    app.add_argument("--model", default=None, help="description model (none: no image description)")
    app.add_argument("--backend", default="transformers", choices=["transformers", "vllm", "openai"])
    app.add_argument("--base-url", default=None, help="OpenAI-compatible server URL")
    app.add_argument("--api-key-env", default=None, help="NAME of the env var holding the API key")
    app.add_argument("--embedding-backend", default=None, help="defaults to the index's own")
    app.add_argument("--embedding-model", default=None, help="defaults to the index's own")
    app.add_argument("--max-new-tokens", type=int, default=1024)
    app.add_argument("--min-similarity", type=float, default=0.2)
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=7860)
    return parser


def _build_index(args) -> int:
    import geopandas as gpd
    import pandas as pd

    from .embedding import ImageEmbedder
    from .service import SceneIndex

    try:
        table = gpd.read_parquet(args.table)
    except Exception:
        table = pd.read_parquet(args.table)
    if args.limit:
        table = table.head(args.limit)
    embedder = ImageEmbedder(model_name=args.embedding_model, backend=args.embedding_backend)
    index = SceneIndex.build(
        table,
        embedder,
        text_column=args.text_column,
        metadata={"source_table": str(args.table), "embedding_backend": args.embedding_backend},
    )
    index.save(args.out)
    print(f"indexed {len(index)} scenes -> {args.out}")
    return 0


def _launch_app(args) -> int:
    from .app import launch
    from .embedding import ImageEmbedder
    from .service import DemoService, SceneIndex

    index = None
    if args.index:
        import json
        from pathlib import Path

        meta = json.loads((Path(args.index) / "index.json").read_text(encoding="utf-8")).get("metadata", {})
        embedder = ImageEmbedder(
            model_name=args.embedding_model or meta.get("embedding_model"),
            backend=args.embedding_backend or meta.get("embedding_backend") or "clip",
        )
        index = SceneIndex.load(args.index, embedder=embedder)

    backend = None
    if args.model:
        from .describer import TransformersBackend, VLLMBackend
        from .openai_compat import OpenAICompatibleBackend

        if args.backend == "openai":
            backend = OpenAICompatibleBackend(
                args.model,
                base_url=args.base_url or "http://localhost:8000/v1",
                api_key_env=args.api_key_env,
                max_new_tokens=args.max_new_tokens,
            )
        elif args.backend == "vllm":
            backend = VLLMBackend(args.model, max_new_tokens=args.max_new_tokens)
        else:
            backend = TransformersBackend(args.model, max_new_tokens=args.max_new_tokens)

    service = DemoService(backend=backend, index=index, chat=backend, min_similarity=args.min_similarity)
    launch(service, host=args.host, port=args.port)
    return 0


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
    if args.command == "build-index":
        return _build_index(args)
    if args.command == "app":
        return _launch_app(args)

    return 2  # pragma: no cover - argparse enforces a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
