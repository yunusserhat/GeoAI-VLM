# Changelog

All notable changes to GeoAI-VLM. Dates are release dates on PyPI.

## [0.4.0] - 2026-10-10

Model independence and street-level indicators for active mobility research.
Real-model results and what was only tested with mocks are listed in
[docs/models.md](docs/models.md).

### Added

- **Model-agnostic description backends.**
  - `TransformersBackend` now loads any image-text-to-text model through
    `AutoProcessor` + `AutoModelForImageTextToText`, puts the image inside
    the chat message, and generates a whole batch in one call with left
    padding (output order equals input order). New options: `dtype`,
    `device_map`, `attn_implementation`, `max_new_tokens`, `top_p`, `top_k`,
    `repetition_penalty`, `seed`, `quantization` (`"4bit"`/`"8bit"`),
    `revision`. Works with transformers 4.57 and 5.x.
  - `VLLMBackend` uses vLLM's generic `LLM.chat()` with in-memory images,
    so models run with their own chat template and processor. New options:
    `max_model_len`, `max_num_seqs`, `limit_mm_per_prompt`, `dtype`,
    `quantization`, `revision`, `llm_kwargs`.
  - `OpenAICompatibleBackend` (new): vLLM server, TGI / Inference Endpoints,
    SGLang, Ollama, LM Studio or a hosted API, using only `requests`, with
    timeouts, bounded retries and backoff, bounded concurrency and base64
    images. `ImageDescriber(backend="openai", base_url=..., api_key_env=...)`.
  - `build_chat_messages()` and `system_prompt_mode` (`"auto"`, `"system"`,
    `"prepend"`): templates that reject or silently drop a system turn get
    the system text prepended to the user turn.
  - Optional structured output (`structured_output=True`): JSON-schema
    constrained decoding where the backend supports it, parsing otherwise.
    `decoding_mode` records `json_schema` (enforced locally by vLLM),
    `json_schema_requested` (sent to an OpenAI-compatible server that
    accepted it; some servers ignore it) or `unconstrained`.
  - Every template now has a JSON schema; responses are validated and the
    problems recorded (`validation_issues`).
- **Provenance columns** on every description: `backend`, `backend_version`,
  `model_revision` (Hugging Face commit sha, `None` when unknown),
  `generation_params`, `system_prompt_mode_effective`, `decoding_mode`,
  `generation_error`, `validation_issues`.
- **`geoai-vlm` command line**: `check-model` (single-image dry run reporting
  load, chat template, system role, JSON parsing, time and memory),
  `build-index` and `app`. Also `geoai_vlm.check_model()`.
- **CLIP-family embeddings**: `ImageEmbedder(backend="clip", model_name=...)`
  for CLIP-family dual encoders (run with SigLIP 2 and CLIP ViT-B/32; MetaCLIP,
  StreetCLIP and others use the same path but were not run);
  L2-normalised vectors; image+text inputs fused by a weighted mean.
- **`active_mobility_audit_v1` prompt template**: 19 features observable in a
  photograph, each with a state (`present` / `absent` / `not_visible` /
  `uncertain`), a confidence and visual evidence; flat `audit_*` columns;
  missing -> `not_assessed`, out-of-vocabulary -> `invalid`, unparseable ->
  `failed`. Module `geoai_vlm.audit`.
- **Street segments** (`geoai_vlm.segments`): match images to the nearest
  segment (20 m, deterministic ties, 2 m ambiguity flag), summarise per
  segment (image and sequence counts, capture dates, covered length share
  with a 25 m support, measurement summaries), write GeoParquet with the
  analysis unit and rules in its metadata. OSMnx networks via the new
  `network` extra.
- **Coverage and recency** (`geoai_vlm.coverage`): by road class, by
  user-supplied areas and by capture year, with network coverage and image
  counts kept apart.
- **Evaluation tools** (`geoai_vlm.evaluation`): sequence- and
  spatial-block splits with leakage checks, percent agreement, Cohen's kappa
  (weighted or not), ICC(2,1) and ICC(3,1), R-squared, Bland-Altman,
  cluster-bootstrap CIs, repeated-generation stability, a solar-elevation
  daylight proxy and seeded stratified reference sampling.
- **Segmentation measurements** (`geoai_vlm.segmentation`, `segment`
  extra): class pixel fractions from a Cityscapes-trained model, with named
  and versioned label mappings checked against the model.
- **Research demo**: `geoai_vlm.service` (scene index, image description,
  nearby and similar scenes, answers grounded in retrieved descriptions with
  image-id citations, declining without evidence) and a local Gradio
  interface `geoai_vlm.app` (`app` extra). Example:
  `examples/build_demo_index.py` builds an index from a small subset of a
  published dataset, keeping its attribution.
- `complete(messages)` on all built-in backends for text-only chats.
- Extras: `transformers`, `qwen`, `quant`, `network`, `segment`, `app`.

### Changed

- `processing_id` now covers the model revision and the output-affecting
  generation settings as well as the model and prompt; sampling-only
  settings are ignored under greedy decoding. **Migration:** records written
  by unreleased development builds with the older two-part id are not
  counted as finished on resume (they cannot be shown to match) and are
  kept as separate records; a note says how many. Releases up to 0.3 wrote
  no `processing_id`, which was already handled this way.
- `trust_remote_code` is **off by default** in every backend (description,
  embedding, segmentation). A model that needs it raises
  `RemoteCodeRequiredError` with the opt-in to use. The default models load
  without it.
- `qwen-vl-utils` moved from the `vlm` extra to the new `qwen` extra; only
  the Qwen3-VL-Embedding Transformers backend uses it, and its absence now
  names the extra.
- `ImageEmbedder(model_name=None)` picks the default per backend (Qwen3-VL-
  Embedding, or SigLIP 2 for `backend="clip"`).
- `ImageDescriber(backend=...)` also accepts a backend instance, which can
  be shared by several describers.
- New description columns are appended to `DESCRIPTION_COLUMNS`; the
  existing columns keep their order.
- `requires-python` is now `>=3.10`. The core dependencies (geopandas 1.1,
  shapely 2.1, libpysal 4.9) already required it, so 3.9 could not install.

### Fixed

- The Transformers backend could not load any model with transformers 5
  (`AutoModelForVision2Seq` was removed upstream).
- The vLLM backend only worked with Qwen models (`qwen_vl_utils`).
- `uv pip install` / `uv sync` inside the repository paired transformers 5
  with an incompatible huggingface-hub because of a uv-only
  `override-dependencies` block (`huggingface-hub<1.0.0`, `attrs>=22.2.0`).
  The block is removed; the `download` and `all` extras resolve to the same
  versions without it. pip was never affected.
- README examples that raised or misbehaved: `describe_place(query=...)`,
  `embed_multimodal(texts=..., image_paths=...)`, `find_optimal_k(k_range=range(...))`
  (silently evaluated k=2 only), `moran_global(...).I` (it returns a dict),
  a `scene_narrative` column that `VectorDB.search` does not return, and a
  link to a text-only embedding model. The API contract test now also
  checks class constructors, method calls and documented `**kwargs`
  forwarding.

### Also in this release: the Phase 0 review fixes

These were merged after 0.3 and ship for the first time in 0.4.0. Details,
tests and migration paths are in [docs/review_findings.md](docs/review_findings.md).

- **Install layout (breaking):** `pip install geoai-vlm` no longer pulls
  torch, vLLM, OpenCV or ZenSVI. Install `geoai-vlm[vlm]`, `[download]`,
  `[slope]`, `[search]` (or `[all]`) for those parts; Vision2Slope names
  stay importable and raise a message naming the extra.
- **`embed_place` (behaviour change):** builds a joint image + text embedding
  by default, as documented; `embedding_modality="text"` restores the old
  text-only vectors. A missing image raises unless `on_missing_image` says
  otherwise.
- **`merge_metadata_and_descriptions` (behaviour change):** raises on
  duplicate join keys instead of inflating rows; `on_duplicate="keep"`
  restores the old join.
- Descriptions: the query's selected images only, a custom `output_path`
  honoured, non-object JSON recorded as a failure, quality reported as
  `reported` / `unknown` / `error` rather than assumed usable, resume keyed on
  model and prompt with failed records retried.
- Vector search: every result states its metric and direction with a
  uniform `similarity`; FAISS inner-product IVF, upsert, persistence and
  metadata filtering fixed.
- Vision2Slope: the parallel path runs, the model loads once per worker,
  every image is accounted for in a records table, unknown headings stay
  unknown.
- CI: CPU tests on Python 3.10-3.12 gate publishing.

### Security

- API keys for OpenAI-compatible servers are read from a named environment
  variable only, none is sent by default, and keys are redacted from logs,
  errors and `repr`; they never enter model inputs or records.

### Known issues

- The vLLM backend and 4-bit/8-bit loading were not run on a GPU for this
  release (API checked against the vLLM 0.13.0 and 0.31.0 sources and with a
  fake module).
- CLIP-family text encoders separate similar descriptions poorly, so the
  demo's question answering retrieves better with Qwen3-VL-Embedding.

## [0.3] - 2026-06-25

- Vision2Slope road slope estimation, vendored under `geoai_vlm.vision2slope`,
  with the `vision2slope` command and slope helpers.

## [0.2.2] - 2026-06-24

- Licence changed from MIT to GPL-3.0-or-later.

## [0.2.1] - 2026-02-28

- Colab-compatible example notebooks.

## [0.2.0] - 2026-02-28

- Multimodal embeddings (Qwen3-VL-Embedding), ChromaDB/FAISS vector search,
  semantic clustering, Moran's I, visualisation and data preparation.

## [0.1.4], [0.1.2], [0.1.1], [0.1.0] - 2026-01-06 to 2026-01-07

- First releases: Mapillary download through ZenSVI, VLM descriptions with
  vLLM and Transformers, GeoParquet output, citation metadata.
