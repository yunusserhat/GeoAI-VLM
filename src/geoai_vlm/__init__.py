# -*- coding: utf-8 -*-
"""
GeoAI-VLM: Geospatial Vision-Language Model Analysis
=====================================================

A Python package for downloading street-level imagery from Mapillary
and generating structured descriptions using Vision-Language Models.

Features:
- Geospatial queries: Point, Line, Polygon, BBox, Place name
- VLM backends: VLLM (high-performance) and Transformers (fallback)
- GeoParquet output with native geometry columns
- Automatic distance calculations (haversine)
- Resume support for incremental processing
- Multimodal embedding (Qwen3-VL-Embedding) with vector search
- Semantic clustering with spatial autocorrelation analysis

Example:
    >>> from geoai_vlm import describe_place
    >>> results = describe_place(
    ...     place_name="Sultanahmet, Istanbul",
    ...     mly_api_key="YOUR_MAPILLARY_API_KEY",
    ...     buffer_m=100
    ... )
"""

__version__ = "0.3"
__author__ = "GeoAI Research"

# Core classes
from .describer import ImageDescriber, VLLMBackend, TransformersBackend, parse_json_response
from .downloader import MapillaryDownloader, download_images
from .geometry import (
    BaseQuery,
    PointQuery,
    LineQuery,
    BBoxQuery,
    PolygonQuery,
    PlaceQuery,
    calculate_point_distance,
    calculate_line_distance,
    add_distance_columns,
)
from .io import (
    save_geoparquet,
    load_geoparquet,
    load_results,
    list_results,
    summarize_results,
    to_geodataframe,
    export_formats,
    merge_metadata_and_descriptions,
)
from .pipeline import (
    describe_query,
    describe_place,
    describe_point,
    describe_line,
    describe_bbox,
    describe_polygon,
    embed_place,
    cluster_descriptions,
    analyze_spatial,
    build_search_index,
    search_similar,
)
from .prompts import (
    GEOAI_SYSTEM_PROMPT,
    GEOAI_USER_PROMPT,
    GEOAI_SCHEMA,
    SIMPLE_SYSTEM_PROMPT,
    SIMPLE_USER_PROMPT,
    get_prompt_template,
    create_custom_prompt,
)

# New modules – embedding, vector store, clustering, spatial, visualization, preparation
from .embedding import ImageEmbedder, TransformersEmbeddingBackend, VLLMEmbeddingBackend
from .vectorstore import VectorDB, ChromaVectorStore, FAISSVectorStore
from .clustering import SemanticClusterer, ClusterConfig
from .spatial import SpatialAnalyzer, MoranResult
from .slope import (
    DEFAULT_SEGMENTATION_MODEL,
    DEFAULT_ROAD_CLASS_IDS,
    ImageSlopeEstimator,
    SlopeConfig,
    SlopeResult,
    aggregate_pano_slopes,
    angle_difference,
    compute_signed_slope,
    create_road_mask,
    estimate_image_slope,
    estimate_slope_from_mask,
    estimate_slope_from_semantic_map,
    estimate_slopes_from_images,
    extract_pano_id,
    extract_perspective_angle,
    extract_road_edge,
    fit_road_edge_line,
)
# ---------------------------------------------------------------------------
# Vision2Slope is imported lazily (PEP 562).
#
# The slope pipeline pulls in the heavy vision stack (cv2, torch,
# transformers, scikit-image, zensvi). Importing it eagerly made every core
# data operation -- geospatial queries, GeoParquet I/O, description parsing,
# clustering -- unusable without a full GPU-capable install. These names stay
# in the public API and resolve on first attribute access instead.
# ---------------------------------------------------------------------------
_VISION2SLOPE_EXPORTS = frozenset(
    {
        "AnalysisConfig",
        "CorrectionProvider",
        "DetectionConfig",
        "GSVDownloader",
        "ImageCorrector",
        "ImageProcessor",
        "ModelConfig",
        "PanoramaTransformer",
        "PipelineConfig",
        "ProcessingConfig",
        "ProcessingError",
        "ProcessingResult",
        "ProcessingStage",
        "ProcessingStatus",
        "SegmentationModel",
        "SegmentationProvider",
        "SkewDetectionProvider",
        "SkewDetector",
        "SlopeAnalysisProvider",
        "StandardImageProcessor",
        "Utils",
        "Vision2SlopeException",
        "Vision2SlopePipeline",
        "VisualizationConfig",
        "VisualizationProvider",
        "Visualizer",
        "ConfigurationError",
        "RoadSlopeAnalyzer",
    }
)


def __getattr__(name):  # noqa: D103 - module level lazy attribute access
    """Resolve Vision2Slope exports on demand (PEP 562)."""
    if name in _VISION2SLOPE_EXPORTS:
        from importlib import import_module

        try:
            module = import_module('.vision2slope', __name__)
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                f"'{name}' requires the optional Vision2Slope dependencies "
                '(opencv-python, torch, transformers, scikit-image, zensvi). '
                "Install them with: pip install 'geoai-vlm[slope]'"
            ) from exc
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():  # noqa: D103
    return sorted(set(globals()) | _VISION2SLOPE_EXPORTS)


from .visualization import (
    plot_elbow_curve,
    plot_cluster_map,
    plot_lisa_map,
    plot_category_distribution,
    generate_report,
)
from .preparation import (
    parse_vlm_descriptions,
    merge_data_sources,
    extract_image_id,
    build_embedding_text,
)


__all__ = [
    # Version
    "__version__",
    
    # Pipeline functions (main API)
    "describe_place",
    "describe_point",
    "describe_line",
    "describe_bbox",
    "describe_polygon",
    "describe_query",
    
    # New pipeline functions
    "embed_place",
    "cluster_descriptions",
    "analyze_spatial",
    "build_search_index",
    "search_similar",
    
    # Core classes
    "ImageDescriber",
    "MapillaryDownloader",
    
    # Query classes
    "PointQuery",
    "LineQuery",
    "BBoxQuery",
    "PolygonQuery",
    "PlaceQuery",
    "BaseQuery",
    
    # Backends
    "VLLMBackend",
    "TransformersBackend",
    
    # Embedding
    "ImageEmbedder",
    "TransformersEmbeddingBackend",
    "VLLMEmbeddingBackend",
    
    # Vector store
    "VectorDB",
    "ChromaVectorStore",
    "FAISSVectorStore",
    
    # Clustering
    "SemanticClusterer",
    "ClusterConfig",
    
    # Spatial analysis
    "SpatialAnalyzer",
    "MoranResult",

    # Road slope estimation
    "SlopeConfig",
    "SlopeResult",
    "ImageSlopeEstimator",
    "DEFAULT_SEGMENTATION_MODEL",
    "DEFAULT_ROAD_CLASS_IDS",
    "create_road_mask",
    "extract_road_edge",
    "fit_road_edge_line",
    "estimate_slope_from_mask",
    "estimate_slope_from_semantic_map",
    "estimate_image_slope",
    "estimate_slopes_from_images",
    "compute_signed_slope",
    "aggregate_pano_slopes",
    "angle_difference",
    "extract_pano_id",
    "extract_perspective_angle",

    # Full Vision2Slope pipeline
    "Vision2SlopePipeline",
    "PipelineConfig",
    "ModelConfig",
    "DetectionConfig",
    "AnalysisConfig",
    "VisualizationConfig",
    "ProcessingConfig",
    "StandardImageProcessor",
    "SegmentationModel",
    "SkewDetector",
    "ImageCorrector",
    "RoadSlopeAnalyzer",
    "Visualizer",
    "Utils",
    "PanoramaTransformer",
    "GSVDownloader",
    "ImageProcessor",
    "SegmentationProvider",
    "SkewDetectionProvider",
    "CorrectionProvider",
    "SlopeAnalysisProvider",
    "VisualizationProvider",
    "Vision2SlopeException",
    "ConfigurationError",
    "ProcessingError",
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingStage",
    
    # Visualization
    "plot_elbow_curve",
    "plot_cluster_map",
    "plot_lisa_map",
    "plot_category_distribution",
    "generate_report",
    
    # Data preparation
    "parse_vlm_descriptions",
    "merge_data_sources",
    "extract_image_id",
    "build_embedding_text",
    
    # I/O functions
    "save_geoparquet",
    "load_geoparquet",
    "load_results",
    "list_results",
    "summarize_results",
    "to_geodataframe",
    "export_formats",
    "download_images",
    "merge_metadata_and_descriptions",
    
    # Geometry utilities
    "calculate_point_distance",
    "calculate_line_distance",
    "add_distance_columns",
    
    # Prompts
    "GEOAI_SYSTEM_PROMPT",
    "GEOAI_USER_PROMPT",
    "GEOAI_SCHEMA",
    "SIMPLE_SYSTEM_PROMPT",
    "SIMPLE_USER_PROMPT",
    "get_prompt_template",
    "create_custom_prompt",
    
    # Utilities
    "parse_json_response",
]
