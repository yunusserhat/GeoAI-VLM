# Roadmap

Phase 0 batch 1 is implemented (see `review_findings.md`). Everything below is
proposed, not built. Each phase lists what it depends on and what would count as
done, so a phase can be accepted or rejected on evidence rather than on the
existence of code.

Nothing in this roadmap should be read as a claim about health outcomes. The
package produces street-level measurements and their provenance; it does not
establish that any design change affects health or behaviour.

---

## Phase 0 batch 2 — remaining correctness work

**Depends on:** batch 1 (done).

| Item | Acceptance |
|---|---|
| Vector search contract | A stated metric and ordering direction per backend; a small hand-computable fixture (≈6 vectors) where the expected ranking is written out by hand and both backends must reproduce it; tests for reload, duplicate-id update, delete, empty result and metadata filters |
| Vision2Slope parallelism | Worker moved to module level and shown to pickle; model created once per process, not per image; a test asserting the model is constructed once for an N-image batch |
| `ModelConfig` plumbing | `device` and `cache_dir` shown to reach the model on both the `_create_processor` and `ImageSlopeEstimator` paths |
| Panorama manifest | Every generated perspective recorded with its source panorama, angle and status; a test that nested folders are fully covered and that the returned file list equals what was produced |
| Full-denominator slope tables | A record-level table keeping every input with its status, plus a separate subset of measurements that passed thresholds; success rate reported over all processed images |
| `visualizers.py` directories | Each output option creates its own directory; tested as independent combinations |

---

## Phase 1 — traceable data and measurement layer

**Depends on:** Phase 0 batch 2.

Per-image record: source id, capture time, position and CRS, heading, panorama
and sequence id where available, local file mapping, quality status, usage
terms. Per-derived-output record: model id and revision, prompt version, config
digest, processing time, code version, status. `processing_id` from batch 1 is
the seed of this; it needs a config digest and code version added.

Separate data structures for observed physical features, model perception
estimates, human annotations, and any later design suggestion — these must not
collapse into one table.

**Acceptance**
- Not-visible, not-present, not-assessed and failed-to-process are four
  distinguishable values; no unknown is written as `0` or as a safe default.
- Segmentation label mappings are versioned and named; Cityscapes and Mapillary
  ids are never assumed equal.
- Green-pixel fraction is named as an image measurement, not as human green-space
  exposure; a daytime image yields no night-lighting claim.
- Interruption and resume: a killed run resumes without duplicating successful
  records and without permanently skipping failed ones (batch 1 covers the
  describer; this extends it to every step).
- Cache key covers input **and** processing configuration.
- Writes are chunked; a batch does not rewrite the whole table.
- A data card states coverage, gaps, quality filters and redistribution limits.
- A publication gate blocks any content lacking source and permission metadata.

**Open question for you:** redistribution terms differ for source imagery and for
derived products. I have not assumed any licence for either.

---

## Phase 2 — research comparisons and local validation

**Depends on:** Phase 1.

Versioned experiment profiles recording model, prompt, image selection,
embedding dimension, normalisation, clustering settings and every randomness
source. **Running a profile reproduces a configuration, not a published result.**

**Acceptance**
- Image-only, text-only and joint-representation runs use the same sample ids;
  samples that shift because a modality is missing are reported separately.
- A fitted clustering model can be reused; transfer experiments never re-fit on
  held-out data.
- Sequence-disjoint and spatial-block evaluation splits; no panorama or near
  duplicate spans train and test.
- Task-appropriate metrics: per-class overlap for segmentation, error and
  agreement for indicator estimates, ranking quality against relevance judgements
  for retrieval. No single overall accuracy score.
- Uncertainty accounts for sequence and spatial dependence.
- Inter-rater disagreement is retained, not averaged away.
- Cluster labels are never treated as ordered magnitudes.
- Slope outputs distinguish raw image angle, camera-geometry-corrected estimate,
  and independently measured physical gradient.
- A panorama may map to several road segments; unknown camera heading is
  represented as unknown, never as 0°.

**Note:** `k=3` in the district study is a finding under specific internal
indices (peak silhouette ≈0.11), not a validated universal street typology. The
software must not fix three clusters as correct for any city. OSM overlap is a
convergence check, not ground truth, and its distance and matching rule must be
recorded.

---

## Phase 3 — explanation audit

**Depends on:** Phase 2.

Traceable links between semantic masks, image variants, model scores and
human-coded rationales. The six semantic groups and their class membership live
in configuration, not in code. Masking options become experiment profiles;
coalition results are cached; model sampling settings are recorded and repeat
runs measure generation variability.

Human coding keeps the PP / NP / PA / NA / U categories. Model-assisted
pre-labelling is allowed, but nothing becomes a reference label without human
confirmation. Visual-presence errors, direction mismatches and failures to
mention an influential feature are reported separately.

**Acceptance**
- An additive toy scorer with known contributions verifies sign, total
  contribution, null feature, absent feature and coalition repeats.
- Defined behaviour for tied scores, empty rationale and zero variance.
- Masking sensitivity and segmentation error stay visible in the output.
- Image masking is never presented as simulating a real intervention or a health
  effect.

**Two questions I need answered before implementing — I will not guess:**
1. The coalition notation describes *removed* classes. Does the sign of the
   marginal difference use the same direction as the "positive contribution"
   reading? This needs checking against the original experiment code.
2. In the top-k measure, the numerator appears to be per-image any-match while
   the denominator counts mentioned features. Per-image hit rate and per-feature
   rate should be defined as two separate metrics.

These are notation and metric-definition questions. They are **not** evidence
that the published results are wrong, and no research text should be edited
automatically.

---

## Phase 4 — local model adaptation

**Depends on:** Phase 2 and 3, plus confirmed hardware, data rights and
annotation capacity.

Measure baseline performance of existing models on local reference data first.
Adaptation is proposed only to address a specific measured failure.

**Acceptance**
- A simple model over frozen embeddings is compared against limited-parameter
  adaptation where appropriate.
- Train / validation / final test separation is preserved.
- Gain, cost and cross-city performance loss are reported together.

No large training job starts before hardware, data rights and annotation
sufficiency are established.

---

## Phase 5 — data and evidence contracts

**Depends on:** Phase 1 and 2.

Auditable street indicators; open data contracts for mobility and qualitative
data that *may* be supplied later; explicit spatial and temporal matching,
coverage differences and unit of analysis.

**Acceptance**
- Raw individual GPS traces never reach a public interface; prototyping uses
  aggregated or synthetic samples, and synthetic data is labelled as such.
- Participant quotes are never invented; what people said and what a model
  summarised are separate fields.
- Visual similarity search is kept distinct from research-evidence search.
- Recommendation output has separate fields for observation, design option
  considered, supporting research source, local applicability rationale,
  uncertainty and missing information — and can decline to produce a
  recommendation when evidence is insufficient.
- The evidence collection starts small and verified, recording each source's
  setting, study design and relevant finding. Association is not reported as
  causation. No unsupported composite health or walkability score.
- Research texts, retrieval results and user feedback are treated as data;
  instructions inside them never change system behaviour. API keys, private
  paths and personal data never enter model inputs or logs.

**Open question:** study area must stay configurable. The pilot area and data
access need your confirmation before anything is hard-coded.

---

## Phase 6 — small, evaluable interface

**Depends on:** Phases 1–5 working.

One interface technology, chosen to fit the existing architecture — not a Gradio
app and a Streamlit app. Data processing and recommendation logic stay
independent of the interface.

**Acceptance**
- The map shows image coverage, capture time and missing data.
- A user can inspect the original image, its masks, the model output and the
  supporting source.
- Scientific evaluation results and the short user-facing explanation are
  separate.
- Feedback flow lets users flag an incorrect observation or an unsuitable
  recommendation; feedback does not become training data without review.
- Setup instructions, a small sample dataset, a data card, experiment configs and
  a known-limitations list ship with it.
- It runs locally first; data permissions and privacy controls are verified
  before any public deployment.

No claim of a co-designed or validated system is made without actual municipal
or participant evaluation.
