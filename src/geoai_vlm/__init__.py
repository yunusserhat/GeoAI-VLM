# -*- coding: utf-8 -*-
"""
GeoAI-VLM: Geospatial Vision-Language Model Analysis
=====================================================

A Python package for downloading street-level imagery from Mapillary
and generating structured descriptions using Vision-Language Models.

Features:
- Geospatial queries: Point, Line, Polygon, BBox, Place name
- Model-agnostic VLM backends: vLLM (LLM.chat), Transformers
  (AutoModelForImageTextToText) and any OpenAI-compatible server
- GeoParquet output with native geometry columns
- Automatic distance calculations (haversine)
- Resume support for incremental processing
- Multimodal embedding (Qwen3-VL-Embedding, or any CLIP-family dual
  encoder such as CLIP, SigLIP 2, MetaCLIP or StreetCLIP) with vector search
- Semantic clustering with spatial autocorrelation analysis

Example:
    >>> from geoai_vlm import describe_place
    >>> results = describe_place(
    ...     place_name="Sultanahmet, Istanbul",
    ...     mly_api_key="YOUR_MAPILLARY_API_KEY",
    ...     buffer_m=100
    ... )
"""

__version__ = "0.4.0"
__author__ = "GeoAI Research"

# Core classes
from .describer import (
    BaseBackend,
    DESCRIPTION_COLUMNS,
    ImageDescriber,
    RemoteCodeRequiredError,
    TransformersBackend,
    VLLMBackend,
    parse_json_response,
)
from .openai_compat import OpenAICompatibleBackend, OpenAICompatibleError
from .chat import (
    GenerationOutput,
    build_chat_messages,
    make_synthetic_street_image,
    resolve_system_prompt_mode,
)
from .models import ModelCheckReport, check_model
from .provenance import compute_processing_id, resolve_model_revision
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
    GEOAI_JSON_SCHEMA,
    SIMPLE_SYSTEM_PROMPT,
    SIMPLE_USER_PROMPT,
    get_prompt_template,
    list_prompt_templates,
    create_custom_prompt,
)
from .schemas import validate_json
from .audit import (
    AUDIT_COLUMNS,
    AUDIT_ITEMS,
    AUDIT_JSON_SCHEMA,
    audit_table,
    flatten_audit_response,
    normalize_audit_response,
    validate_audit_response,
)

# New modules – embedding, vector store, clustering, spatial, visualization, preparation
from .embedding import (
    ClipEmbeddingBackend,
    ImageEmbedder,
    TransformersEmbeddingBackend,
    VLLMEmbeddingBackend,
)
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


from .segments import (
    SEGMENT_DEFAULTS,
    aggregate_segments,
    load_osm_segments,
    prepare_segments,
    read_segment_metadata,
    segments_from_graph,
    snap_images_to_segments,
    write_segment_parquet,
)
from .coverage import coverage_by_area, network_coverage, recency_report
from .segmentation import (
    CITYSCAPES_19,
    LabelMapping,
    LabelMappingError,
    SemanticSegmenter,
    check_label_mapping,
    class_pixel_fractions,
    mapping_from_id2label,
)
from .service import DemoService, GroundedAnswer, SceneIndex
from .evaluation import (
    SplitLeakageError,
    agreement_report,
    assign_spatial_blocks,
    bland_altman,
    check_split_leakage,
    cluster_bootstrap_ci,
    cohen_kappa,
    generation_stability,
    grouped_split,
    icc,
    light_condition,
    percent_agreement,
    r_squared,
    repeated_generation,
    sequence_split,
    solar_elevation,
    spatial_block_split,
    stratified_reference_sample,
)

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
    "BaseBackend",
    "VLLMBackend",
    "TransformersBackend",
    "OpenAICompatibleBackend",
    "OpenAICompatibleError",
    "RemoteCodeRequiredError",
    "DESCRIPTION_COLUMNS",

    # Chat construction and model checks
    "GenerationOutput",
    "build_chat_messages",
    "resolve_system_prompt_mode",
    "make_synthetic_street_image",
    "ModelCheckReport",
    "check_model",
    "compute_processing_id",
    "resolve_model_revision",
    
    # Embedding
    "ImageEmbedder",
    "TransformersEmbeddingBackend",
    "VLLMEmbeddingBackend",
    "ClipEmbeddingBackend",
    
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
    
    # Street-segment indicators and coverage
    "SEGMENT_DEFAULTS",
    "prepare_segments",
    "segments_from_graph",
    "load_osm_segments",
    "snap_images_to_segments",
    "aggregate_segments",
    "write_segment_parquet",
    "read_segment_metadata",
    "network_coverage",
    "coverage_by_area",
    "recency_report",

    # Segmentation measurements
    "CITYSCAPES_19",
    "LabelMapping",
    "LabelMappingError",
    "SemanticSegmenter",
    "check_label_mapping",
    "class_pixel_fractions",
    "mapping_from_id2label",

    # Demo services (interface-independent; the Gradio app is geoai_vlm.app)
    "SceneIndex",
    "DemoService",
    "GroundedAnswer",

    # Evaluation
    "SplitLeakageError",
    "grouped_split",
    "sequence_split",
    "assign_spatial_blocks",
    "spatial_block_split",
    "check_split_leakage",
    "percent_agreement",
    "cohen_kappa",
    "icc",
    "r_squared",
    "bland_altman",
    "cluster_bootstrap_ci",
    "agreement_report",
    "repeated_generation",
    "generation_stability",
    "solar_elevation",
    "light_condition",
    "stratified_reference_sample",

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
    "GEOAI_JSON_SCHEMA",
    "SIMPLE_SYSTEM_PROMPT",
    "SIMPLE_USER_PROMPT",
    "get_prompt_template",
    "list_prompt_templates",
    "create_custom_prompt",
    "validate_json",

    # Active mobility audit (observation template)
    "AUDIT_ITEMS",
    "AUDIT_COLUMNS",
    "AUDIT_JSON_SCHEMA",
    "audit_table",
    "flatten_audit_response",
    "normalize_audit_response",
    "validate_audit_response",
    
    # Utilities
    "parse_json_response",
]
