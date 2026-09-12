# Review findings and Phase 0 status

Baseline reviewed: commit `7f880f70` (`release: add Vision2Slope slope estimation v0.3`),
50 tracked files. This document records what was verified, what was changed, and
what is still open.

## What "verified" means here

Each finding below was confirmed by reading the code in this checkout **and** by
a test that fails on the old behaviour and passes on the new one. Findings
marked *open* were confirmed by reading only; no claim is made that they are
fixed.

The test suite runs on CPU. It downloads no models, makes no network calls and
needs no API key. It therefore verifies **software behaviour, not model quality
or scientific validity**.

## Phase 0 batch 1 — data integrity

| # | Finding | Status | Test |
|---|---------|--------|------|
| R1 | `describe_query` accepted `output_path` but the normal export branch always wrote `output_dir/results.*` | fixed | `TestDescribeQueryOutputContract::test_custom_output_path_is_actually_written` |
| R2 | `describe_query` described every image found in `output_dir`, not the images the query selected | fixed | `TestDescribeQueryOutputContract::test_only_selected_images_are_described` |
| R3 | `parse_json_response` returned any valid JSON, so a list or `null` raised `AttributeError`/`TypeError` mid-batch | fixed | `TestInvalidModelResponses` (5 payload types) |
| R4 | The `simple` prompt template emits `description`/`tags`, which never reached the summary columns | fixed | `TestPromptSchemaAndQuality::test_simple_schema_populates_summary_columns` |
| R5 | A missing `image_quality` block was recorded as `usable=True` | fixed | `TestPromptSchemaAndQuality::test_missing_quality_block_is_unknown_not_usable` |
| R6 | `build_embedding_text` raised `ValueError` on list-valued cells (`semantic_tags` is a list) | fixed | `TestBuildEmbeddingText` (5 cases) |
| R7 | `merge_metadata_and_descriptions` silently inflated the row count on duplicate ids | fixed | `TestDuplicateIdMerge` |
| R8 | Resume was keyed on image id alone, so a changed model or prompt was skipped as "done", and failures were skipped permanently | fixed | `TestResumeBehaviour` (4 cases) |
| R9 | `embed_place` documented a joint image+text representation but embedded text only | fixed | `TestEmbeddingModality` (5 cases) |
| R10 | `import geoai_vlm` required `cv2`/`torch`/`vllm`, so no data work was possible without the full GPU stack | fixed | `TestOptionalHeavyDependencies` |
| R11 | The test embedder seeded on batch length: content-independent and batch-size-dependent | fixed | `TestMockEmbedderContract` (4 cases) |
| R12 | README and notebook calls did not match real signatures; the elbow test swapped its axes and only asserted the figure type | fixed | `tests/test_api_contract.py`, `TestPlotElbowCurve::test_plots_k_on_x_axis_and_inertia_on_y` |
| R13 | The only workflow published to PyPI and ran no tests | fixed | `.github/workflows/tests.yml`, gating `publish.yml` |

### Three quality states, not one default

`usable` is no longer a permissive boolean. Each description record now carries
`quality_status`:

| `quality_status` | `usable` | Meaning |
|---|---|---|
| `reported` | `True`/`False` | The model stated whether the image is usable |
| `unknown` | `None` | No quality block was returned — **not** assumed usable |
| `error` | `None` | The response could not be parsed |

A record that failed to parse is never counted as a successful one, and it is
retried on the next resume rather than being permanently skipped.

### Provenance on every derived record

`ImageDescriber.describe` now writes `model_name`, `prompt_version`,
`processing_id` and `processed_at` alongside each description.
`processing_id` is a hash of model **and** prompt; resume and de-duplication key
on `(image_id, processing_id)`, so re-running the same images under a different
model or prompt produces a new record instead of being skipped, and re-running
under the *same* model and prompt overwrites rather than duplicates.

## API changes and migration

All three changes below are deliberate and change behaviour. Each has a
one-argument path back to the old behaviour.

| Change | Old behaviour | Restore it with |
|---|---|---|
| `embed_place` now builds a joint image+text embedding | text-only, despite the docstring | `embedding_modality="text"` |
| `merge_metadata_and_descriptions` raises on duplicate join keys | one-to-many join, inflating row counts | `on_duplicate="keep"` |
| Heavy dependencies moved to extras | `pip install geoai-vlm` pulled torch, vllm, cv2 | `pip install 'geoai-vlm[all]'` |

Additive, backward-compatible changes:

- `ImageDescriber.describe(image_paths=[...])` — describe an explicit set.
  `image_dir=` still works exactly as before.
- `save_path=` on the four plotting helpers.
- `generate_report(gdf)` — `keywords` is now optional.
- `embed_place(on_missing_image=...)` — `"error"` (default), `"skip"`, `"text"`.
  A missing image is never silently downgraded to a text embedding; the chosen
  behaviour is recorded per row in `embedding_modality`.

### Install layout

```
pip install geoai-vlm            # core: queries, download, I/O, clustering, spatial
pip install 'geoai-vlm[vlm]'     # + local VLM inference (torch, vllm, transformers)
pip install 'geoai-vlm[search]'  # + chromadb, faiss
pip install 'geoai-vlm[slope]'   # + Vision2Slope (cv2, torch, zensvi, streetlevel)
pip install 'geoai-vlm[all]'     # everything
```

Vision2Slope names remain importable from the top-level package; they resolve on
first attribute access (PEP 562) and raise a message naming the extra if the
dependencies are absent.

## Still open — confirmed by reading, not yet fixed

These were reproduced by source inspection in this checkout. None has a
regression test yet, and none should be treated as closed.

**Vector search**
- `vectorstore.py` reports both FAISS and Chroma results in a `distance` field
  without a stated metric or ordering direction.
- Index reload, duplicate-id update, delete, empty results and metadata filters
  are exercised only with random vectors; no small hand-checkable reference case.
- No test covers IVF configuration or differing index types.

**Vision2Slope / scale**
- `Vision2SlopePipeline.process_batch_parallel` defines its worker inside the
  method, so it does not survive standard multiprocessing pickling; the worker
  body also re-creates the model for every image.
- `ModelConfig.device` and `cache_dir` are not forwarded on the `_create_processor`
  path, though the simple `ImageSlopeEstimator` honours them.
- `pano2perspective.generate_left_right` does not branch the transformation; the
  returned file list collects only top-level files while the pipeline searches
  nested folders and takes the first match.
- `_bi_slope_estimate` returns only the successful, in-threshold subset, so the
  summary denominator loses failures and filtered steep angles.
- Unknown GSV heading is coerced to 0°, and later metadata handling keeps the
  first record per panorama, discarding other road-direction matches.
- `visualizers.py` does not create `masks_dir` when only the road-mask output is
  enabled.

**Scale behaviour**
- The batch write path in `ImageDescriber.describe` still rewrites the whole
  table each batch. It is now correct (no duplicates, provenance preserved) but
  its cost grows with the table; a partitioned layout is Phase 1 work.
