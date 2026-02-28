# GeoAI-VLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18169685.svg)](https://doi.org/10.5281/zenodo.18169685)
[![PyPI version](https://img.shields.io/pypi/v/geoai-vlm)](https://pypi.org/project/geoai-vlm/)
[![PyPI downloads](https://static.pepy.tech/badge/geoai-vlm)](https://pepy.tech/project/geoai-vlm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yunusserhat/GeoAI-VLM?style=social)](https://github.com/yunusserhat/GeoAI-VLM)

**Geospatial Vision-Language Model analysis for street-level imagery.**

GeoAI-VLM combines [ZenSVI](https://github.com/koito19960406/ZenSVI)'s Mapillary downloading capabilities with Vision-Language Models (VLMs) and a high-performance [vLLM](https://github.com/vllm-project/vllm) backend to generate structured descriptions of street-level images. Starting with v0.2.0, GeoAI-VLM also supports **multimodal embedding** with [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), enabling **semantic clustering**, **spatial autocorrelation analysis**, and **vector similarity search** over geotagged imagery. It's designed for GeoAI research.

## Features

### Core

- 🗺️ **Geospatial Queries**: Point, line, polygon, and bounding box queries with automatic buffering
- 📸 **Mapillary Integration**: Download street-level imagery via ZenSVI
- 🤖 **VLM Analysis**: Generate structured descriptions using Qwen-VL, and other image-text-to-text models
- 📊 **GeoParquet Output**: Native geometry columns for seamless GIS integration
- 📏 **Distance Calculations**: Automatic distance-to-query computation using haversine
- ⚡ **High Performance**: [vLLM](https://github.com/vllm-project/vllm) backend for fast batch inference ([Transformers](https://github.com/huggingface/transformers) fallback available)
- 🔄 **Resume Support**: Skip already-processed images for incremental workflows

### Embedding & Analysis (v0.2.0)

- 🧬 **Multimodal Embeddings**: Generate dense vector representations from text and images using [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (2B & 8B variants)
- 🔍 **Vector Search**: Build searchable indices with ChromaDB or FAISS and retrieve semantically similar places by text or image query
- 📈 **Semantic Clustering**: K-Means clustering over embeddings with automatic keyword extraction per cluster
- 🌐 **Spatial Autocorrelation**: Global and local Moran's I to detect spatial patterns in cluster assignments
- 📉 **Visualization**: Elbow curves, cluster maps, LISA significance maps, category distributions, and full HTML reports


## Requirements & Platform Support

- Python 3.9-3.12 supported
- **Windows is NOT supported** due to the [vLLM](https://github.com/vllm-project/vllm) dependency. Please use Linux or macOS.
- CUDA-compatible GPU (recommended for VLM inference)
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
from geoai_vlm import ImageDescriber, describe_place

# Use custom system/user prompts
custom_system = """You are an urban safety analyst. Describe safety-relevant features."""
custom_user = """Analyze this street image for: lighting, visibility, foot traffic, escape routes."""

results = describe_place(
    query="Fatih, Istanbul",
    mly_api_key="YOUR_API_KEY",
    system_prompt=custom_system,
    user_prompt=custom_user,
    output_path="safety_analysis.parquet"
)
```

### Using Different Backends

```python
from geoai_vlm import ImageDescriber

# VLLM backend (default, fastest)
describer = ImageDescriber(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    backend="vllm",
    gpu_memory_utilization=0.8
)

# Transformers backend (fallback)
describer = ImageDescriber(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    backend="transformers",
    device="cuda"
)

# Describe images
results = describer.describe(
    image_dir="./my_images",
    output_path="descriptions.parquet",
    batch_size=8
)
```

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

## Multimodal Embeddings

Generate dense vector representations from VLM descriptions and street-level images using [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):

```python
from geoai_vlm import ImageEmbedder

# Initialize the embedder (auto-selects vLLM or Transformers backend)
embedder = ImageEmbedder(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    backend="auto"
)

# Embed text descriptions
vectors = embedder.embed_texts(["A busy commercial street with shops"])
print(vectors.shape)  # (1, hidden_dim)

# Embed images directly
img_vectors = embedder.embed_images(["path/to/image.jpg"])

# Multimodal: combine text + image into a single embedding
mm_vectors = embedder.embed_multimodal(
    texts=["A quiet residential area"],
    image_paths=["path/to/image.jpg"]
)
```

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

# Find the optimal number of clusters
k_values, inertias = clusterer.find_optimal_k(gdf, k_range=range(2, 20))

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

# Global Moran's I — is there overall spatial clustering?
global_result = analyzer.moran_global(gdf, column="cluster")
print(f"Moran's I = {global_result.I:.3f}, p = {global_result.p_sim:.4f}")

# Local Moran's I (LISA) — where are the hot/cold spots?
gdf = analyzer.moran_local(gdf, column="cluster")
# Adds 'lisa_Is', 'lisa_q', 'lisa_p_sim' columns to the GeoDataFrame
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
    metadata_columns=["land_use_character", "cluster"]
)

# Search by natural language
results = vdb.search(query_text="tree-lined residential street", n_results=5)
print(results[["scene_narrative", "distance"]])

# Search by image
results = vdb.search(query_image="query_photo.jpg", n_results=5)
```

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
plot_category_distribution(gdf, category_columns=["land_use_character"])

# Full HTML report
generate_report(gdf, output_dir="./report")
```

## One-Line Pipeline

Run the entire workflow — download, describe, embed, cluster, analyze — in a single call:

```python
from geoai_vlm import embed_place, cluster_descriptions, analyze_spatial

# 1. Download + embed
gdf = embed_place(
    place_name="Sultanahmet, Istanbul",
    mly_api_key="YOUR_API_KEY",
    embedding_model="Qwen/Qwen3-Embedding-0.6B"
)

# 2. Cluster
gdf = cluster_descriptions(gdf, n_clusters=8)

# 3. Spatial analysis
gdf = analyze_spatial(gdf, column="cluster", k_neighbors=8)
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

## Requirements

- Python 3.9-3.12 supported
- Mapillary API key ([get one here](https://www.mapillary.com/developer))
- GPU recommended for VLM inference and embedding generation

## Dependencies

- **Core**: geopandas, pandas, shapely, pyarrow, haversine
- **Downloading**: zensvi (Mapillary integration)
- **VLM (choose one)**:
  - vLLM + qwen-vl-utils (recommended)
  - Transformers + torch + accelerate
- **Embedding & Analysis**: chromadb, faiss-cpu, scikit-learn, matplotlib, libpysal, esda

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use GeoAI-VLM in your research, please cite:

```bibtex
@software{geoai_vlm,
  author  = {B{\i}cak{\c{c}}{\i}, Yunus Serhat},
  title   = {GeoAI-VLM: Geospatial Vision-Language Model Analysis},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.18169685},
  url     = {https://github.com/yunusserhat/GeoAI-VLM}
}
```

## Acknowledgments

- [ZenSVI](https://github.com/koito19960406/ZenSVI) for Mapillary integration
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for vision-language models
- [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) for multimodal embeddings
- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
- [ChromaDB](https://github.com/chroma-core/chroma) and [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [PySAL](https://pysal.org/) for spatial statistics
