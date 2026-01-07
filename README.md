# GeoAI-VLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18169685.svg)](https://doi.org/10.5281/zenodo.18169685)
[![PyPI version](https://img.shields.io/pypi/v/geoai-vlm.svg)](https://pypi.org/project/geoai-vlm/)
[![PyPI downloads](https://static.pepy.tech/badge/geoai-vlm)](https://pepy.tech/project/geoai-vlm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yunusserhat/GeoAI-VLM?style=social)](https://github.com/yunusserhat/GeoAI-VLM)

**Geospatial Vision-Language Model analysis for street-level imagery.**

GeoAI-VLM combines [ZenSVI](https://github.com/koito19960406/ZenSVI)'s Mapillary downloading capabilities with Vision-Language Models (VLMs) and a high-performance [vLLM](https://github.com/vllm-project/vllm) backend to generate structured descriptions of street-level images. It's designed for GeoAI research.

## Features

- 🗺️ **Geospatial Queries**: Point, line, polygon, and bounding box queries with automatic buffering
- 📸 **Mapillary Integration**: Download street-level imagery via ZenSVI
- 🤖 **VLM Analysis**: Generate structured descriptions using Qwen-VL, and other image-text-to-text models
- 📊 **GeoParquet Output**: Native geometry columns for seamless GIS integration
- 📏 **Distance Calculations**: Automatic distance-to-query computation using haversine
- ⚡ **High Performance**: [vLLM](https://github.com/vllm-project/vllm) backend for fast batch inference ([Transformers](https://github.com/huggingface/transformers) fallback available)
- 🔄 **Resume Support**: Skip already-processed images for incremental workflows


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
- GPU recommended for VLM inference

## Dependencies

- **Core**: geopandas, pandas, shapely, pyarrow, haversine
- **Downloading**: zensvi (Mapillary integration)
- **VLM (choose one)**:
  - vLLM + qwen-vl-utils (recommended)
  - Transformers + torch + accelerate

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
- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
