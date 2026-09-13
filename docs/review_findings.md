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
pip install geoai-vlm              # core: queries, GeoParquet I/O, parsing, clustering, spatial
pip install 'geoai-vlm[download]'  # + Mapillary download (zensvi)
pip install 'geoai-vlm[vlm]'       # + local VLM inference (torch, vllm, transformers)
pip install 'geoai-vlm[search]'    # + chromadb, faiss
pip install 'geoai-vlm[slope]'     # + Vision2Slope (cv2, torch, zensvi, streetlevel)
pip install 'geoai-vlm[all]'       # everything
```

The core install deliberately excludes the Mapillary download path: `zensvi`
resolves to roughly 109 packages including `torch`, `torchvision`,
`transformers` and CUDA runtime libraries, which would defeat the split. The
end-to-end pipeline (`describe_place`, `describe_query`) therefore needs
`[download]`; calling it without that extra raises an error naming it.

Vision2Slope names remain importable from the top-level package; they resolve on
first attribute access (PEP 562) and raise a message naming the extra if the
dependencies are absent.

## Phase 0 batch 2a — vector search contract

Tested against a hand-computable fixture of six unit vectors whose cosine
similarity to the query can be read off by eye, so the expected ranking is
written out by hand rather than taken from whatever the library returns
(`tests/test_vectorstore_contract.py`, 31 tests).

What the backends actually returned, all under one field named `distance`:

| Backend | Raw `distances` for the fixture | Meaning |
|---|---|---|
| ChromaDB `cosine` | `0.0, 0.2, 0.4, 1.0, 1.6, 2.0` | distance, lower is closer |
| FAISS `ip` (**the default**) | `1.0, 0.8, 0.6, 0.0, -0.6, -1.0` | **similarity, higher is closer** |
| FAISS `l2` | `0.0, 0.4, 0.8, 2.0, 3.2, 4.0` | squared distance, lower is closer |

Sorting results ascending by `distance` therefore returned the *least* similar
items first on the FAISS default.

| # | Finding | Status |
|---|---------|--------|
| V1 | One `distance` field carried three different meanings, one of them inverted | fixed |
| V2 | `metric="ip"` with `index_type="ivf"` built an **L2** index: `IndexIVFFlat` defaults to `METRIC_L2` regardless of the quantizer, so inner-product searches were silently ranked by L2 | fixed |
| V3 | Re-adding an existing id appended a second entry in FAISS (count and ids diverged; the stale vector stayed searchable) while ChromaDB upserted | fixed |
| V4 | `FAISSVectorStore.load` did not restore `metric`, so a persisted `l2` store came back declaring `ip` while its numbers were still L2 | fixed |
| V5 | IVF with fewer vectors than `nlist` died inside faiss clustering with a bare `RuntimeError` | fixed |
| V6 | No metadata filtering existed on either backend | added |
| V7 | IVF searched a single cell (`nprobe` left at 1) | fixed |

### The result contract

Every query result now states what it is:

| Key | Meaning |
|---|---|
| `distances` | the backend's raw score, unchanged |
| `metric` | the backend's own metric name |
| `direction` | `lower_is_closer` or `higher_is_closer`, describing `distances` |
| `similarity` | uniform score, **higher is always closer** |
| `rank` | 0-based, already ordered best-first |

`similarity` is the true cosine similarity where the metric allows it (ChromaDB
cosine/ip, FAISS ip); for an L2 metric it is the negated distance —
order-preserving within one store, but not a calibrated cosine and not
comparable across metrics. `VectorDB.search` returns `id`, `rank`, `similarity`
and `distance` columns ordered best-first, and reports the metric and direction
in `df.attrs`.

`query(..., where={"key": "value"})` filters on stored metadata: natively on
ChromaDB, and by over-fetching then filtering on FAISS so a selective filter
still returns up to `n_results` rows.

## Still open — confirmed by reading, not yet fixed

These were reproduced by source inspection in this checkout. None has a
regression test yet, and none should be treated as closed.

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

## Review round 1 (PR #1)

Five further defects were raised by the PR review bot and fixed in the same PR.
Four of them were introduced by the batch-1 changes themselves.

| # | Finding | Test |
|---|---------|------|
| R12 | `zensvi` moved out of core, but `MapillaryDownloader` needs it, leaving the default install unable to run the documented pipeline | `TestDownloadDependencyContract` |
| R13 | An empty selected set fell through to scanning the work directory — R2 again, exactly when no image was downloadable | `TestEmptySelection` |
| R14 | `bool()` coercion made `null` a reported *unusable* and the string `"false"` a reported *usable* | `TestNonBooleanQuality` (10 cases) |
| R15 | A legacy output without `processing_id` was treated as this model and prompt's finished work | `TestLegacyResume` |
| R16 | Making `keywords` optional left `generate_report` listing no clusters at all | `TestGenerateReportClusters` |

**Scale behaviour**
- The batch write path in `ImageDescriber.describe` still rewrites the whole
  table each batch. It is now correct (no duplicates, provenance preserved) but
  its cost grows with the table; a partitioned layout is Phase 1 work.
