# GeoAI-VLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18169685.svg)](https://doi.org/10.5281/zenodo.18169685)
[![PyPI version](https://img.shields.io/pypi/v/geoai-vlm.svg)](https://pypi.org/project/geoai-vlm/)
[![PyPI downloads](https://static.pepy.tech/badge/geoai-vlm)](https://pepy.tech/project/geoai-vlm)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub stars](https://img.shields.io/github/stars/yunusserhat/GeoAI-VLM?style=social)](https://github.com/yunusserhat/GeoAI-VLM)

**Geospatial Vision-Language Model analysis for street-level imagery.**

GeoAI-VLM downloads street-level imagery from Mapillary (through [ZenSVI](https://github.com/koito19960406/ZenSVI)) and describes it with open vision-language models (VLMs) -- any Hugging Face image-text-to-text model through [Transformers](https://github.com/huggingface/transformers) or [vLLM](https://github.com/vllm-project/vllm), or any model behind an OpenAI-compatible endpoint. It embeds images and descriptions (Qwen3-VL-Embedding or CLIP-family encoders) for **semantic clustering**, **spatial autocorrelation analysis** and **vector search**, estimates **road slope** (inspired by [Vision2Slope](https://github.com/CubicsYang/Vision2Slope)), and turns image-level observations into **street-segment indicators** with explicit coverage, recency and evaluation tools for active mobility research. It is designed for GeoAI research.

## Features

### Core

- 🗺️ **Geospatial Queries**: Point, line, polygon, and bounding box queries with automatic buffering
- 📸 **Mapillary Integration**: Download street-level imagery via ZenSVI
- 🤖 **Model-agnostic VLM analysis**: Qwen-VL, SmolVLM, LLaVA-OneVision, InternVL and other image-text-to-text models, locally or through an OpenAI-compatible server (vLLM server, TGI, SGLang, Ollama, LM Studio)
- 🧾 **Provenance on every record**: backend and its version, model revision, generation settings, how the system prompt was delivered and whether decoding was schema-constrained
- 📊 **GeoParquet Output**: Native geometry columns for seamless GIS integration
- 📏 **Distance Calculations**: Automatic distance-to-query computation using haversine
- 🛣️ **Road Slope Estimation**: Estimate road edge angle from Mapillary semantic segmentation maps or directly from images
- 🔄 **Resume Support**: Skip already-processed images for incremental workflows

### Embedding & Analysis

- 🧬 **Embeddings**: [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) (multimodal) or a CLIP-family dual encoder such as CLIP or SigLIP 2 ([models actually run](docs/models.md))
- 🔍 **Vector Search**: ChromaDB or FAISS indices, searchable by text or image
- 📈 **Semantic Clustering**: K-Means over embeddings with keyword extraction per cluster
- 🌐 **Spatial Autocorrelation**: Global and local Moran's I
- 📉 **Visualization**: Elbow curves, cluster maps, LISA maps, category distributions and Markdown reports

### Street-level indicators (v0.4)

- 🚶 **Active-mobility observation template**: 19 features visible in a photograph (sidewalks, crossings, cycle infrastructure, trees, seating, light poles, ramps, barriers...), each with a state, a confidence and the visual evidence
- 🧭 **Street segments**: match images to street segments and summarise them per segment with coverage and capture dates
- 🗓️ **Coverage and recency**: how much of the network is actually observed, and how recently -- kept separate from image counts
- ✅ **Evaluation tools**: leakage-free sequence and spatial-block splits, kappa, ICC, Bland-Altman, cluster bootstrap, generation stability and stratified reference sampling
- 🧩 **Segmentation measurements**: class pixel fractions from a Cityscapes model with versioned label mappings
- 🖥️ **Research demo**: a small local web interface with grounded, cited answers

## Requirements & Platform Support

- Python 3.10-3.12 (the core dependencies need 3.10 or later)
- The **vLLM** backend needs Linux and an NVIDIA GPU. The **Transformers** and **OpenAI-compatible** backends do not use vLLM; this release was tested on Linux only.
- [Mapillary API key](https://www.mapillary.com/developer) for downloading street-level imagery

## Set up using Python

### Create a new Python environment

It's recommended to use [uv](https://github.com/astral-sh/uv), a very fast Python environment manager, to create and manage Python environments. Please follow the [documentation](https://docs.astral.sh/uv/#getting-started) to install uv. After installing uv, you can create a new Python environment using the following commands:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

## Installation

### Option 1: Install from PyPI

```bash
uv pip install geoai-vlm
```

The core install covers queries, GeoParquet I/O, parsing, clustering, spatial statistics, street segments, coverage and evaluation. Heavier parts are extras:

| Extra | Adds |
|---|---|
| `geoai-vlm[vlm]` | vLLM + Transformers + torch (local VLM inference) |
| `geoai-vlm[transformers]` | Transformers + torch only (no vLLM; also enough for CLIP-family embeddings) |
| `geoai-vlm[qwen]` | `qwen-vl-utils`, needed only by the Qwen3-VL-Embedding Transformers backend |
| `geoai-vlm[quant]` | bitsandbytes for 4-bit / 8-bit loading |
| `geoai-vlm[download]` | ZenSVI (Mapillary download) |
| `geoai-vlm[search]` | ChromaDB and FAISS |
| `geoai-vlm[network]` | OSMnx (download street networks) |
| `geoai-vlm[segment]` | semantic segmentation measurements |
| `geoai-vlm[app]` | Gradio research demo |
| `geoai-vlm[slope]` / `[panorama]` | Vision2Slope |
| `geoai-vlm[all]` | everything |

### Option 2: Install from GitHub

```bash
# Clone the repository
git clone https://github.com/yunusserhat/geoai-vlm.git
cd geoai-vlm

# Install in the current environment
uv pip install .

# For development (editable mode)
uv pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import geoai_vlm; print('GeoAI-VLM installed successfully!')"
```

## Quick Start

### Basic Usage

```python
from geoai_vlm import describe_place

# Describe images from a place name
results = describe_place(
    place_name="Sultanahmet, Istanbul",
    mly_api_key="YOUR_MAPILLARY_API_KEY",
    buffer_m=100,
    output_path="sultanahmet_descriptions.parquet"
)

print(results.head())
```

### Point Query with Distance

```python
from geoai_vlm import describe_point

# Query images near a specific coordinate
results = describe_point(
    lat=41.0082,
    lon=28.9784,
    buffer_m=50,
    mly_api_key="YOUR_API_KEY",
    output_path="hagia_sophia.parquet"
)

# Results include distance_to_query_m column
print(results[['image_id', 'distance_to_query_m', 'scene_narrative']].head())
```

### Line Query (Street/Route Analysis)

```python
from geoai_vlm import describe_line
from shapely.geometry import LineString

# Analyze images along a street
street_line = LineString([
    (28.9700, 41.0100),  # Start point (lon, lat)
    (28.9750, 41.0120),  # Midpoint
    (28.9800, 41.0080),  # End point
])

results = describe_line(
    geometry=street_line,
    buffer_m=25,
    mly_api_key="YOUR_API_KEY"
)

# Results include distance_to_line_m and distance_along_line_m
```

### Bounding Box Query

```python
from geoai_vlm import describe_bbox

results = describe_bbox(
    minx=28.970, miny=41.005,
    maxx=28.985, maxy=41.015,
    mly_api_key="YOUR_API_KEY",
    model_name="Qwen/Qwen3-VL-2B-Instruct"
)
```

### Custom Prompts

```python
from geoai_vlm import describe_place

# Use custom system/user prompts
custom_system = """You are an urban safety analyst. Describe safety-relevant features."""
custom_user = """Analyze this street image for: lighting, visibility, foot traffic, escape routes."""

results = describe_place(
    place_name="Fatih, Istanbul",
    mly_api_key="YOUR_API_KEY",
    system_prompt=custom_system,
    user_prompt=custom_user,
    output_path="safety_analysis.parquet"
)
```

## Choosing a Model and Backend

Any image-text-to-text model works the same way; only the model id and the backend change.

```python
from geoai_vlm import ImageDescriber

# vLLM: fastest on a GPU; any model vLLM supports, via its own chat template
describer = ImageDescriber(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    backend="vllm",
    gpu_memory_utilization=0.8,
    max_model_len=8192,
)

# Transformers: CPU or GPU, no vLLM needed; batched with left padding
describer = ImageDescriber(
    model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    backend="transformers",
    device_map="auto",
    dtype="bfloat16",
    max_new_tokens=1024,
)

# Describe images
results = describer.describe(
    image_dir="./my_images",
    output_path="descriptions.parquet",
    batch_size=8
)
```

- `backend="auto"` (default) uses vLLM when it is installed and a GPU is present, Transformers otherwise.
- `system_prompt_mode="auto"` (default) checks the model's chat template once: if it rejects or silently drops a system turn, the system text is put at the start of the user turn instead. `"system"` and `"prepend"` force one form. Each record says which form was used (`system_prompt_mode_effective`).
- `structured_output=True` constrains decoding to the prompt template's JSON schema where the backend supports it and otherwise parses free text as before. Each record states `decoding_mode`: `json_schema` (enforced locally by vLLM), `json_schema_requested` (sent to an OpenAI-compatible server that accepted it -- some servers enforce it, others ignore it, which the client cannot see) or `unconstrained`. Responses are validated against the schema either way (`validation_issues`).
- `trust_remote_code` is **off by default**. A model that ships its own code raises `RemoteCodeRequiredError` until you review it and opt in with `trust_remote_code=True`.
- The older backend arguments (`device`, `torch_dtype`, `max_tokens`) still work.

Every description record carries its provenance: `backend`, `backend_version`, `backend_endpoint` (an HTTP server's URL without credentials), `model_revision` (the Hugging Face commit, or `None` when it cannot be known), `generation_params`, `system_prompt_mode_effective`, `decoding_mode` and `processing_id`. `processing_id` covers the model, its revision, the prompt version, the output-affecting generation settings and the backend (with the endpoint of an HTTP server, whose model names are only labels), so resuming a run never mixes outputs from different configurations.

Check whether a model works before a long run (one synthetic image; reports load, chat template, system-role support, JSON parsing, time and memory):

```bash
geoai-vlm check-model HuggingFaceTB/SmolVLM2-256M-Video-Instruct
```

See [docs/models.md](docs/models.md) for the models actually run for this release and those only expected to work.

## OpenAI-compatible Endpoints

Describe images with a model served by a vLLM server, Hugging Face TGI or Inference Endpoints, SGLang, Ollama, LM Studio or a hosted API:

```python
from geoai_vlm import ImageDescriber

describer = ImageDescriber(
    model_name="Qwen/Qwen3-VL-2B-Instruct",   # the name the server knows the model by
    backend="openai",
    base_url="http://localhost:8000/v1",       # e.g. `vllm serve Qwen/Qwen3-VL-2B-Instruct`
    api_key_env="MY_ENDPOINT_KEY",             # NAME of the env var holding the key
    max_concurrency=4,
    timeout=120,
)
results = describer.describe(image_dir="./my_images", output_path="descriptions.parquet")
```

Requests use only `requests`, with timeouts, bounded retries and backoff (honouring `Retry-After`) and a bounded thread pool; images are sent as base64 data URLs. The key is read from the environment variable you name; **no key is sent by default**, so a key meant for one service is never forwarded to another server. It never appears in logs, records, error messages or model inputs.

## Output Schema

The default GeoAI schema extracts structured urban features:

```python
{
    "scene_narrative": "80-120 word description of the urban scene",
    "land_use_character": {"primary": "commercial", "intensity": "high"},
    "urban_morphology": {"street_type": "pedestrian", "enclosure_ratio": "high"},
    "streetscape_elements": {"sidewalk_quality": "good", "street_trees": "moderate"},
    "mobility_infrastructure": {"modes_visible": ["pedestrian", "bicycle"]},
    "place_character": {"dominant_activity": "shopping", "human_presence": "crowded"},
    "environmental_quality": {"greenery_coverage": "moderate", "cleanliness": "good"},
    "semantic_tags": ["historic", "tourist", "commercial", "pedestrian", "busy"]
}
```

## Active-mobility Observation Template

`active_mobility_audit_v1` asks only for what a single photograph can show. For each of 19 items -- sidewalk (with a visual width class), sidewalk gap, sidewalk obstruction, pedestrian crossing (and its type), intersection, traffic calming, cycle infrastructure, street trees, other greenery, shade on the walkway, seating, street light pole (presence only), parking on the sidewalk, curb ramp, accessibility barrier, surface damage, litter, active frontage and view obstruction -- the model returns a state, a confidence and the visual evidence.

```python
from geoai_vlm import ImageDescriber, audit_table

describer = ImageDescriber(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    prompt_template="active_mobility_audit_v1",
    structured_output=True,
)
df = describer.describe(image_dir="./my_images", output_path="audit.parquet")

print(df[["image_id", "audit_sidewalk_state", "audit_sidewalk_width_class",
          "audit_pedestrian_crossing_state", "audit_cycle_infrastructure_state"]])
long_table = audit_table(df)   # one row per image and item
```

| State | Meaning |
|---|---|
| `present` | the feature is visible |
| `absent` | where it would be is clearly visible, and it is not there |
| `not_visible` | that part of the street is out of frame, occluded or unclear |
| `uncertain` | something is visible but cannot be told apart |
| `not_assessed` | the model returned nothing for the item |
| `invalid` | the model returned a value outside the vocabulary (rejected) |
| `failed` | the response could not be parsed |

An unknown is never written as `absent` or `0`. The template asks for no lighting-adequacy, safety, health or walkability judgement and produces no score.

## Street Segments and Coverage

Move image observations to street segments, keeping track of how well each segment is actually observed:

```python
import geopandas as gpd
from geoai_vlm import (
    describe_place, prepare_segments, snap_images_to_segments, aggregate_segments,
    write_segment_parquet, network_coverage, coverage_by_area, recency_report,
)

# Image points with Mapillary metadata (capture time, sequence) and audit columns
images = describe_place(
    place_name="Your district, Your city",
    mly_api_key="YOUR_API_KEY",
    prompt_template="active_mobility_audit_v1",
)
segments = prepare_segments(gpd.read_file("streets.gpkg"), id_column="street_id")

snapped = snap_images_to_segments(images, segments, max_distance_m=20, ambiguity_margin_m=2)
summary = aggregate_segments(
    snapped,
    segments,
    state_columns=["audit_sidewalk_state", "audit_street_trees_state"],
    support_length_m=25,
)
write_segment_parquet(summary, "segments.parquet")  # analysis unit and rules in the metadata

print(network_coverage(summary))                                  # share of network length observed
print(recency_report(summary, year_edges=(2018, 2021, 2024)))      # by year of the newest image
```

- Images go to the nearest segment within 20 m; equal distances go to the smaller segment id; a runner-up within 2 m marks the match `ambiguous` (kept, and reported).
- Each image is taken to observe 25 m of street centred on it; `covered_length_share` is the union of those stretches over the segment length. Image counts are reported separately -- many images of one corner still cover one corner.
- For audit states, `present_share` is present / (present + absent); `not_visible`, `uncertain`, `not_assessed`, `invalid`, `failed` and missing values are counted, never folded in.
- `coverage_by_area(summary, areas)` reports the same within your own polygons (neighbourhoods, districts). Street networks can come from any line layer, or from OpenStreetMap with `load_osm_segments(place=...)` (`geoai-vlm[network]`).

## Evaluation Tools

```python
from geoai_vlm import (
    spatial_block_split, check_split_leakage, agreement_report,
    light_condition, stratified_reference_sample,
)

# Train/test split where neither a 500 m block nor a capture sequence spans both sides
split = spatial_block_split(images, block_size_m=500, test_size=0.2, seed=0, sequence_column="sequence_id")
check_split_leakage(split, ["block_id", "sequence_id"])   # raises on any leak

# Agreement of model observations with reference labels, CIs by sequence
# (labels: one row per image with both states and its sequence id)
report = agreement_report(labels, "reference_state", "model_state", kind="categorical",
                          cluster_column="sequence_id", n_boot=1000, seed=0)

# Seeded, quota-based reference sample: area x road class x daylight proxy
images["light"] = light_condition(images["captured_at"], images["lat"], images["lon"])
sample = stratified_reference_sample(images, ["area", "road_class", "light"], n_per_stratum=20, seed=2026)
```

Also available: `cohen_kappa` (unweighted or weighted), `icc` (ICC(2,1), ICC(3,1)), `r_squared` and `bland_altman` (e.g. between two segmentation models), `cluster_bootstrap_ci`, and `repeated_generation` + `generation_stability` to measure how often repeated sampled generations agree per field. `light_condition` is a proxy computed from capture time and solar elevation, not an observation of the image.

## Segmentation Measurements

```python
from geoai_vlm import SemanticSegmenter

segmenter = SemanticSegmenter()   # SegFormer-B0 trained on Cityscapes
table = segmenter.measure(["a.jpg", "b.jpg"])
print(table[["image_id", "vegetation_pixel_fraction", "sky_pixel_fraction", "label_mapping"]])
```

Columns name the measurement (`vegetation_pixel_fraction` is the share of labelled pixels assigned to vegetation in that image -- not green-space exposure). Label mappings are versioned (`cityscapes-19@1`); a model whose labels do not match its mapping is rejected, so Cityscapes and Mapillary Vistas ids are never mixed.

## Research Demo

A small local interface: describe an uploaded image (structured description plus the observation template), find similar indexed scenes on a map, list scenes near a coordinate, and ask questions answered **only** from retrieved descriptions -- every statement cites an image id, and the tool declines when no indexed record supports an answer.

```bash
pip install 'geoai-vlm[app,transformers]'
python examples/build_demo_index.py --n 300 --out ./demo_index   # published dataset, attribution kept
geoai-vlm app --index ./demo_index --model Qwen/Qwen3-VL-2B-Instruct
```

The demo binds to 127.0.0.1 and never creates a public link. It is a research demonstration: it shows model outputs and retrieval, gives no design recommendations and makes no claim about health, safety or walkability. The logic lives in `geoai_vlm.service`, independent of the interface.

## Road Slope Estimation (v0.3)

Estimate road slope from a Mapillary Vistas semantic segmentation map:

```python
from geoai_vlm import SlopeConfig, estimate_slope_from_semantic_map

result = estimate_slope_from_semantic_map(
    semantic_map,
    config=SlopeConfig(morphology_kernel_size=15, min_edge_points=10)
)

print(result.road_edge_line_angle)  # road edge angle in degrees
```

Or estimate directly from an image using the default Mask2Former Mapillary Vistas model:

```python
from geoai_vlm import ImageSlopeEstimator

estimator = ImageSlopeEstimator()
result = estimator.estimate("street_view.jpg")

print(result.to_dict())
```

The complete Vision2Slope pipeline is also available inside GeoAI-VLM:

```python
from geoai_vlm import (
    PipelineConfig,
    ProcessingConfig,
    VisualizationConfig,
    Vision2SlopePipeline,
)

config = PipelineConfig(
    input_dir="street_view_images",
    output_dir="slope_output",
    processing_config=ProcessingConfig(
        is_panorama=True,
        panorama_fov=90,
        panorama_phi=0.0,
        panorama_aspects=(10, 10),
    ),
    viz_config=VisualizationConfig(
        save_visualizations=True,
        save_corrected_images=True,
        save_intermediate_results=True,
    ),
)

pipeline = Vision2SlopePipeline(config)
slope_results = pipeline.process_batch()
```

All original Vision2Slope modules are vendored under `geoai_vlm.vision2slope`, including panorama transformation, semantic segmentation, skew correction, road-edge fitting, visualization, CLI, and optional Google Street View downloading.

For panorama left/right perspective outputs, aggregate image-level measurements into a panorama-level slope:

```python
from geoai_vlm import aggregate_pano_slopes

slope_df = aggregate_pano_slopes(results_df, angle_threshold=10)
print(slope_df[["pano_id", "road_estimated_slope"]].drop_duplicates())
```

## Multimodal Embeddings

Generate dense vectors from descriptions and images with [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) (default) or a CLIP-family encoder:

```python
from geoai_vlm import ImageEmbedder

# Qwen3-VL-Embedding (auto-selects vLLM or Transformers)
embedder = ImageEmbedder(model_name="Qwen/Qwen3-VL-Embedding-2B", backend="auto")

# Or a CLIP-family dual encoder. Run for this release: SigLIP 2 and CLIP ViT-B/32;
# MetaCLIP, StreetCLIP and other checkpoints use the same path but were not run.
embedder = ImageEmbedder(backend="clip", model_name="google/siglip2-base-patch16-224")

# Embed text descriptions
vectors = embedder.embed_texts(["A busy commercial street with shops"])
print(vectors.shape)  # (1, embedding_dim)

# Embed images directly
img_vectors = embedder.embed_images(["path/to/image.jpg"])

# Multimodal: one vector for an image together with its description
mm_vectors = embedder.embed_multimodal(
    [{"image": "path/to/image.jpg", "text": "A quiet residential area"}]
)
```

All vectors are L2-normalised, so inner product equals cosine similarity. With the CLIP backend an image-plus-text input is the normalised mean of its two vectors (a simple fusion, kept in the same space as text-only and image-only vectors).

## Semantic Clustering

Cluster geotagged descriptions by semantic similarity and extract per-cluster keywords:

```python
from geoai_vlm import SemanticClusterer, ClusterConfig

config = ClusterConfig(
    n_clusters=8,
    embedding_columns=["scene_narrative", "semantic_tags"],
    n_keywords=10
)
clusterer = SemanticClusterer(embedder=embedder, config=config)

# Cluster a GeoDataFrame of VLM descriptions
gdf = clusterer.cluster(gdf)
print(gdf["cluster"].value_counts())

# Find the optimal number of clusters (k from 2 to 19)
k_values, inertias = clusterer.find_optimal_k(gdf, k_range=(2, 20))

# Extract TF-IDF keywords per cluster
keywords = clusterer.extract_keywords(gdf)
for cluster_id, words in keywords.items():
    print(f"Cluster {cluster_id}: {words}")
```

## Spatial Autocorrelation

Detect whether semantic clusters are spatially random or form significant patterns:

```python
from geoai_vlm import SpatialAnalyzer

analyzer = SpatialAnalyzer(k_neighbors=8)

# Global Moran's I per cluster -- {cluster_id: MoranResult}
global_results = analyzer.moran_global(gdf, column="cluster")
for cluster_id, result in global_results.items():
    print(f"cluster {cluster_id}: Moran's I = {result.I:.3f}, p = {result.p_value:.4f}")

# Local Moran's I (LISA) -- where are the hot/cold spots?
gdf = analyzer.moran_local(gdf, column="cluster")
# Adds 'lisa_cluster', 'lisa_p_value' and 'lisa_significant' columns
```

## Vector Similarity Search

Build a searchable index over your geotagged descriptions and find semantically similar places:

```python
from geoai_vlm import VectorDB

# Build an index from a GeoDataFrame
vdb = VectorDB(embedder=embedder, store_backend="chromadb")
vdb.build(
    gdf,
    text_column="scene_narrative",
    image_dir="./images",
    metadata_columns=["land_use_primary", "cluster"]
)

# Search by natural language (best match first)
results = vdb.search(query_text="tree-lined residential street", n_results=5)
print(results[["id", "similarity", "land_use_primary"]])

# Search by image
results = vdb.search(query_image="query_photo.jpg", n_results=5)
```

Rank by `similarity` (higher is always closer); `distance` is the backend's raw score, whose direction depends on the metric.

## Visualization

```python
from geoai_vlm import (
    plot_elbow_curve,
    plot_cluster_map,
    plot_lisa_map,
    plot_category_distribution,
    generate_report
)

# Elbow curve for choosing k
plot_elbow_curve(k_values, inertias, save_path="elbow.png")

# Map of clusters
plot_cluster_map(gdf, cluster_column="cluster", save_path="clusters.png")

# LISA significance map
plot_lisa_map(gdf, save_path="lisa.png")

# Category breakdown
plot_category_distribution(gdf, category_columns=["land_use_primary"])

# Full report (Markdown)
generate_report(gdf, output_path="./report/summary.md")
```

## One-Line Pipeline

Run the entire workflow — download, describe, embed, cluster, analyze — in a single call:

```python
from geoai_vlm import embed_place, cluster_descriptions, analyze_spatial

# 1. Download + embed
gdf = embed_place(
    place_name="Sultanahmet, Istanbul",
    mly_api_key="YOUR_API_KEY",
    model_name="Qwen/Qwen3-VL-Embedding-2B",
    # "multimodal" (default) encodes image + description jointly;
    # use "text" for description-only or "image" for image-only.
    embedding_modality="multimodal",
)

# 2. Cluster
gdf = cluster_descriptions(gdf, n_clusters=8)

# 3. Spatial analysis
# Returns {"global": {cluster_id: MoranResult}, "gdf": <GeoDataFrame with LISA columns>}
spatial = analyze_spatial(gdf, column="cluster", k_neighbors=8)
gdf = spatial["gdf"]
```

## GeoParquet Output

Results are saved as GeoParquet with native geometry:

```python
import geopandas as gpd

# Load results
gdf = gpd.read_parquet("results.parquet")

# Native geometry column preserved
print(gdf.geometry)  # POINT geometries
print(gdf.crs)       # EPSG:4326

# Easy GIS operations
gdf.to_file("results.geojson", driver="GeoJSON")
gdf.explore()  # Interactive map in Jupyter
```

## Limitations: Measurements, Not Claims

- Every value is an **image observation**: what one photograph shows, from one viewpoint, at one moment, as read by a model. `vegetation_pixel_fraction` is a share of pixels, not green-space exposure; a street light pole being visible says nothing about lighting at night.
- Model outputs are not ground truth. Check them against reference labels (see the evaluation tools) before using them as measurements, and report agreement with its uncertainty.
- Street-segment summaries depend on coverage: report `covered_length_share` and capture dates with them. Few or old images describe a segment poorly, however many indicators they produce.
- Nothing in this package establishes an effect of the built environment on walking, cycling, physical activity or health, and it computes no composite walkability or health score.
- Imagery and derived data keep their original licences (Mapillary imagery is CC BY-SA 4.0); keep attribution when you redistribute.

## Dependencies

- **Core**: geopandas, pandas, shapely, pyarrow, haversine, requests, scikit-learn, libpysal, esda
- **Downloading**: zensvi (Mapillary integration)
- **VLM**: vLLM and/or Transformers + torch; any OpenAI-compatible server needs nothing extra
- **Embedding & Analysis**: chromadb, faiss-cpu
- **Networks / segmentation / demo**: osmnx, transformers + torch, gradio
- **Slope Estimation / Vision2Slope**: transformers, torch, Pillow, scikit-learn, scikit-image, opencv-python, zensvi, streetlevel

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## Citation

If you use GeoAI-VLM in your research, please cite:

```bibtex
@software{geoai_vlm,
  author  = {B{\i}{\c{c}}ak{\c{c}}{\i}, Yunus Serhat},
  title   = {GeoAI-VLM: Geospatial Vision-Language Model Analysis},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.18169685},
  url     = {https://github.com/yunusserhat/GeoAI-VLM}
}
```

Related publications and data:

- Bıçakçı, Y. S., Shingleton, J., Wang, Y., & Basiri, A. (2026). *Mapping the Semantics of the Street: A VLM-Driven Geospatial Analysis in Fatih, Istanbul.* 1st International Conference on Geospatial Artificial Intelligence (GeoAI 2026). [doi:10.5281/zenodo.19390648](https://doi.org/10.5281/zenodo.19390648)
- Fatih Mapillary street-level images with derived layers (dataset, CC BY-SA 4.0). [doi:10.57967/hf/10144](https://doi.org/10.57967/hf/10144)
- Bıçakçı, Y. S. (2026). *yunusserhat/alphaearth_asprs: v1.0.0* [Software], code accompanying an article accepted in *Photogrammetric Engineering & Remote Sensing*. [doi:10.5281/zenodo.22818970](https://doi.org/10.5281/zenodo.22818970)

## Acknowledgments

- [ZenSVI](https://github.com/koito19960406/ZenSVI) for Mapillary integration
- [Vision2Slope](https://github.com/CubicsYang/Vision2Slope) for the road slope estimation workflow
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for vision-language models
- [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) for multimodal embeddings
- [vLLM](https://github.com/vllm-project/vllm) and [Transformers](https://github.com/huggingface/transformers) for inference
- [ChromaDB](https://github.com/chroma-core/chroma) and [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [PySAL](https://pysal.org/) for spatial statistics
- [OSMnx](https://github.com/gboeing/osmnx) and OpenStreetMap contributors for street networks
