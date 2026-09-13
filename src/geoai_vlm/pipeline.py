# -*- coding: utf-8 -*-
"""
Pipeline Module for GeoAI-VLM
==============================
End-to-end functions for downloading and describing street-level imagery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon

from .describer import DESCRIPTION_COLUMNS, ImageDescriber
from .downloader import MapillaryDownloader
from .geometry import (
    BaseQuery,
    BBoxQuery,
    LineQuery,
    PlaceQuery,
    PointQuery,
    PolygonQuery,
)
from .io import load_geoparquet, merge_metadata_and_descriptions, save_geoparquet, to_geodataframe


__all__ = [
    "describe_place",
    "describe_point",
    "describe_line",
    "describe_bbox",
    "describe_polygon",
    "describe_query",
    "embed_place",
    "cluster_descriptions",
    "analyze_spatial",
    "build_search_index",
    "search_similar",
]


def describe_query(
    query: BaseQuery,
    mly_api_key: str,
    output_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    backend: str = "auto",
    batch_size: int = 8,
    resolution: int = 1024,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    prompt_template: str = "geoai",
    resume: bool = True,
    metadata_only: bool = False,
    verbosity: int = 1,
    export_formats: str = "geoparquet",
    max_images: Optional[int] = None,
    **backend_kwargs,
) -> gpd.GeoDataFrame:
    """
    End-to-end pipeline: download images → describe with VLM → save as GeoParquet.
    
    Args:
        query: Geospatial query (PointQuery, LineQuery, BBoxQuery, PolygonQuery, PlaceQuery)
        mly_api_key: Mapillary API key
        output_dir: Directory for downloaded images and cache
        output_path: Path for output GeoParquet (default: output_dir/results.parquet)
        model_name: HuggingFace model name
        backend: VLM backend ("vllm", "transformers", or "auto")
        batch_size: Batch size for VLM inference
        resolution: Image resolution (256, 1024, 2048)
        start_date: Filter images after this date (YYYY-MM-DD)
        end_date: Filter images before this date (YYYY-MM-DD)
        system_prompt: Custom system prompt (overrides template)
        user_prompt: Custom user prompt (overrides template)
        prompt_template: Prompt template name ("geoai" or "simple")
        resume: Skip already-processed images
        metadata_only: Only download metadata, skip VLM description
        verbosity: Verbosity level
        export_formats: Space-separated export formats (geoparquet, geojson, csv, gpkg)
        max_images: Maximum number of images to process (None for all, sorted by distance)
        **backend_kwargs: Additional kwargs for VLM backend
        
    Returns:
        GeoDataFrame with image metadata, VLM descriptions, and distances
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_path is None:
        output_path = output_dir / "results.parquet"
    output_path = Path(output_path)
    
    # Step 1: Download images
    if verbosity > 0:
        print(f"Step 1: Downloading images from Mapillary...")
    
    downloader = MapillaryDownloader(
        mly_api_key=mly_api_key,
        verbosity=verbosity,
    )
    
    metadata_gdf = downloader.download(
        query=query,
        output_dir=output_dir,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
        metadata_only=metadata_only,
        max_images=max_images,
    )
    
    if len(metadata_gdf) == 0:
        print("No images found for the given query.")
        return gpd.GeoDataFrame()
    
    if verbosity > 0:
        print(f"Downloaded metadata for {len(metadata_gdf)} images")
    
    # If metadata_only, save and return
    if metadata_only:
        save_geoparquet(metadata_gdf, output_path)
        return metadata_gdf
    
    # Step 2: Describe images with VLM
    if verbosity > 0:
        print(f"\nStep 2: Describing images with {model_name}...")
    
    describer = ImageDescriber(
        model_name=model_name,
        backend=backend,
        prompt_template=prompt_template if system_prompt is None else None,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        **backend_kwargs,
    )
    
    # Use a temporary path for descriptions
    desc_path = output_dir / "descriptions_temp.parquet"

    # Describe exactly the images this query selected. Scanning output_dir
    # instead would also pick up images left by an earlier, wider run, so the
    # result would silently contain rows the query never asked for.
    selected_paths = None
    if "image_path" in metadata_gdf.columns:
        candidates = [
            Path(p) for p in metadata_gdf["image_path"].dropna().tolist()
        ]
        selected_paths = [p for p in candidates if p.exists()]
        n_missing = len(candidates) - len(selected_paths)
        if n_missing and verbosity > 0:
            print(
                f"   Note: {n_missing} of {len(candidates)} selected images are "
                "not present on disk (download failed or filtered) and will "
                "have no description."
            )

    if selected_paths is not None:
        # The query produced a selection. An *empty* selection means nothing
        # was downloadable -- which is not an invitation to scan the work
        # directory, since that is exactly how unrelated images got described.
        if selected_paths:
            descriptions_df = describer.describe(
                image_paths=selected_paths,
                output_path=desc_path,
                batch_size=batch_size,
                resume=resume,
            )
        else:
            if verbosity > 0:
                print("   No images available to describe for this query.")
            descriptions_df = pd.DataFrame(columns=list(DESCRIPTION_COLUMNS))
    else:
        # No image_path column (e.g. an older metadata frame): fall back to
        # scanning the directory, as before.
        descriptions_df = describer.describe(
            image_dir=output_dir,
            output_path=desc_path,
            batch_size=batch_size,
            resume=resume,
        )
    
    if verbosity > 0:
        print(f"Generated descriptions for {len(descriptions_df)} images")
    
    # Step 3: Merge metadata with descriptions
    if verbosity > 0:
        print(f"\nStep 3: Merging metadata and descriptions...")
    
    result_gdf = merge_metadata_and_descriptions(
        metadata_gdf=metadata_gdf,
        descriptions_df=descriptions_df,
        on="image_id",
    )
    
    # Step 4: Save results
    if verbosity > 0:
        print(f"\nStep 4: Saving results...")
    
    from .io import export_formats as do_export

    # Export next to the requested output_path, under its name. Previously this
    # branch always wrote output_dir/results.*, so a caller-supplied
    # output_path was accepted and then quietly ignored.
    do_export(result_gdf, output_path.parent, output_path.stem, export_formats)
    
    # Clean up temp file
    if desc_path.exists():
        desc_path.unlink()
    
    if verbosity > 0:
        print(f"\n✅ Complete! {len(result_gdf)} images processed.")
        print(f"   Output: {output_path}")
    
    return result_gdf


def describe_place(
    place_name: str,
    mly_api_key: str,
    output_dir: Optional[Union[str, Path]] = None,
    buffer_m: float = 0,
    max_images: Optional[int] = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Download and describe images from a place name.
    
    Args:
        place_name: OSM-compatible place name (e.g., "Sultanahmet, Istanbul")
        mly_api_key: Mapillary API key
        output_dir: Output directory (default: ./{place_name_slug}/)
        buffer_m: Additional buffer in meters
        max_images: Maximum number of images to process (None for all, sorted by distance)
        **kwargs: Additional arguments passed to describe_query
        
    Returns:
        GeoDataFrame with images, descriptions, and distances
        
    Example:
        >>> results = describe_place(
        ...     place_name="Fatih, Istanbul",
        ...     mly_api_key="YOUR_KEY",
        ...     buffer_m=0,
        ...     max_images=50  # Only process 50 nearest images
        ... )
    """
    # Create default output directory from place name
    if output_dir is None:
        slug = place_name.lower().replace(" ", "_").replace(",", "")
        output_dir = Path(f"./{slug}")
    
    query = PlaceQuery(place_name=place_name, buffer_m=buffer_m)
    
    return describe_query(
        query=query,
        mly_api_key=mly_api_key,
        output_dir=output_dir,
        max_images=max_images,
        **kwargs,
    )


def describe_point(
    lat: float,
    lon: float,
    mly_api_key: str,
    buffer_m: float = 50,
    output_dir: Optional[Union[str, Path]] = None,
    nearest_only: bool = False,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Download and describe images near a point.
    
    Args:
        lat: Latitude of the query point
        lon: Longitude of the query point
        mly_api_key: Mapillary API key
        buffer_m: Search radius in meters
        output_dir: Output directory (default: ./point_{lat}_{lon}/)
        nearest_only: Return only the single closest image
        **kwargs: Additional arguments passed to describe_query
        
    Returns:
        GeoDataFrame with images, descriptions, and distance_to_query_m
        
    Example:
        >>> results = describe_point(
        ...     lat=41.0082, lon=28.9784,
        ...     mly_api_key="YOUR_KEY",
        ...     buffer_m=100
        ... )
    """
    if output_dir is None:
        output_dir = Path(f"./point_{lat:.4f}_{lon:.4f}")
    
    query = PointQuery(lat=lat, lon=lon, buffer_m=buffer_m, nearest_only=nearest_only)
    
    return describe_query(
        query=query,
        mly_api_key=mly_api_key,
        output_dir=output_dir,
        **kwargs,
    )


def describe_line(
    geometry: Union[LineString, list],
    mly_api_key: str,
    buffer_m: float = 25,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Download and describe images along a line (street, route, path).
    
    Args:
        geometry: Shapely LineString or list of (lon, lat) coordinate tuples
        mly_api_key: Mapillary API key
        buffer_m: Buffer distance from line in meters
        output_dir: Output directory (default: ./line_query/)
        **kwargs: Additional arguments passed to describe_query
        
    Returns:
        GeoDataFrame with images, descriptions, distance_to_line_m, and distance_along_line_m
        
    Example:
        >>> from shapely.geometry import LineString
        >>> street = LineString([(28.97, 41.01), (28.98, 41.02)])
        >>> results = describe_line(
        ...     geometry=street,
        ...     mly_api_key="YOUR_KEY",
        ...     buffer_m=25
        ... )
    """
    if output_dir is None:
        output_dir = Path("./line_query")
    
    query = LineQuery(geometry=geometry, buffer_m=buffer_m)
    
    return describe_query(
        query=query,
        mly_api_key=mly_api_key,
        output_dir=output_dir,
        **kwargs,
    )


def describe_bbox(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    mly_api_key: str,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Download and describe images within a bounding box.
    
    Args:
        minx: Minimum longitude (west)
        miny: Minimum latitude (south)
        maxx: Maximum longitude (east)
        maxy: Maximum latitude (north)
        mly_api_key: Mapillary API key
        output_dir: Output directory (default: ./bbox_query/)
        **kwargs: Additional arguments passed to describe_query
        
    Returns:
        GeoDataFrame with images, descriptions, and distance_to_centroid_m
        
    Example:
        >>> results = describe_bbox(
        ...     minx=28.97, miny=41.00, maxx=28.99, maxy=41.02,
        ...     mly_api_key="YOUR_KEY"
        ... )
    """
    if output_dir is None:
        output_dir = Path("./bbox_query")
    
    query = BBoxQuery(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    
    return describe_query(
        query=query,
        mly_api_key=mly_api_key,
        output_dir=output_dir,
        **kwargs,
    )


def describe_polygon(
    geometry: Union[Polygon, list],
    mly_api_key: str,
    buffer_m: float = 0,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Download and describe images within a polygon.
    
    Args:
        geometry: Shapely Polygon or list of (lon, lat) coordinate tuples
        mly_api_key: Mapillary API key
        buffer_m: Additional buffer in meters
        output_dir: Output directory (default: ./polygon_query/)
        **kwargs: Additional arguments passed to describe_query
        
    Returns:
        GeoDataFrame with images, descriptions, and inside_query flag
        
    Example:
        >>> from shapely.geometry import Polygon
        >>> area = Polygon([(28.97, 41.00), (28.99, 41.00), (28.99, 41.02), (28.97, 41.02)])
        >>> results = describe_polygon(
        ...     geometry=area,
        ...     mly_api_key="YOUR_KEY"
        ... )
    """
    if output_dir is None:
        output_dir = Path("./polygon_query")
    
    query = PolygonQuery(geometry=geometry, buffer_m=buffer_m)
    
    return describe_query(
        query=query,
        mly_api_key=mly_api_key,
        output_dir=output_dir,
        **kwargs,
    )


# =========================================================================
# New high-level functions – embedding, clustering & spatial analysis
# =========================================================================


def embed_place(
    place_name: str,
    mly_api_key: str,
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    embedding_backend: str = "auto",
    output_dir: Optional[Union[str, Path]] = None,
    buffer_m: float = 0,
    max_images: Optional[int] = None,
    embedding_modality: str = "multimodal",
    text_column: str = "scene_narrative",
    image_column: str = "image_path",
    on_missing_image: str = "error",
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Download, describe **and embed** images from a place name.

    Extends :func:`describe_place` by additionally generating multimodal
    embeddings for every image, stored in a NumPy-friendly column.

    Args:
        place_name: OSM-compatible place name.
        mly_api_key: Mapillary API key.
        model_name: Embedding model name.
        embedding_backend: ``"vllm"``, ``"transformers"`` or ``"auto"``.
        output_dir: Output directory.
        buffer_m: Buffer in metres.
        max_images: Maximum images to process.
        **kwargs: Forwarded to :func:`describe_query`.

        embedding_modality: Which representation to build.

            ``"multimodal"`` (default) encodes the image together with its
            generated description -- the joint representation used by the Fatih
            study. ``"text"`` encodes the description only. ``"image"`` encodes
            the image only.
        text_column: Column holding the description text.
        image_column: Column holding the local image path.
        on_missing_image: Behaviour when an image file is absent and the
            modality needs it. ``"error"`` (default) raises, ``"skip"`` leaves
            the embedding empty and records ``missing_image``, ``"text"``
            downgrades that row to a text-only embedding and records it as such.

    Returns:
        GeoDataFrame with VLM descriptions, an ``embedding`` column and an
        ``embedding_modality`` column recording how each row was encoded.

    Note:
        Prior to v0.4 this function always produced a *text-only* embedding
        despite documenting a multimodal one. Pass ``embedding_modality="text"``
        to reproduce the old numbers exactly.
    """
    from .embedding import ImageEmbedder

    valid_modalities = {"multimodal", "text", "image"}
    if embedding_modality not in valid_modalities:
        raise ValueError(
            f"embedding_modality must be one of {sorted(valid_modalities)}"
        )
    valid_missing = {"error", "skip", "text"}
    if on_missing_image not in valid_missing:
        raise ValueError(f"on_missing_image must be one of {sorted(valid_missing)}")

    gdf = describe_place(
        place_name=place_name,
        mly_api_key=mly_api_key,
        output_dir=output_dir,
        buffer_m=buffer_m,
        max_images=max_images,
        **kwargs,
    )

    if len(gdf) == 0:
        return gdf

    gdf = gdf.copy()
    embedder = ImageEmbedder(model_name=model_name, backend=embedding_backend)

    texts = gdf[text_column].fillna("").astype(str).tolist() if text_column in gdf else [""] * len(gdf)

    if embedding_modality == "text":
        embeddings = embedder.embed_texts(texts)
        gdf["embedding"] = list(embeddings)
        gdf["embedding_modality"] = "text"
        return gdf

    # Image is required from here on.
    if image_column not in gdf.columns:
        raise ValueError(
            f"embedding_modality={embedding_modality!r} needs an image path "
            f"column {image_column!r}, which is not present."
        )

    paths = [
        Path(p) if isinstance(p, (str, Path)) and str(p) else None
        for p in gdf[image_column].tolist()
    ]
    present = [p is not None and p.exists() for p in paths]
    n_missing = len(present) - sum(present)

    if n_missing and on_missing_image == "error":
        first = next(
            (str(p) for p, ok in zip(paths, present) if not ok), "<empty>"
        )
        raise FileNotFoundError(
            f"{n_missing} of {len(paths)} image files are missing, so a "
            f"{embedding_modality} embedding cannot be built for them. "
            f"First missing: {first}. Pass on_missing_image='skip' to leave "
            "them empty, or 'text' to downgrade those rows explicitly."
        )

    # Never downgrade silently: every row records how it was actually encoded.
    modality_col = [None] * len(gdf)
    embedding_col = [None] * len(gdf)

    usable_idx = [i for i, ok in enumerate(present) if ok]
    if usable_idx:
        if embedding_modality == "image":
            vectors = embedder.embed_images([str(paths[i]) for i in usable_idx])
        else:
            vectors = embedder.embed_multimodal(
                [
                    {"image": str(paths[i]), "text": texts[i]}
                    for i in usable_idx
                ]
            )
        for slot, i in enumerate(usable_idx):
            embedding_col[i] = vectors[slot]
            modality_col[i] = embedding_modality

    missing_idx = [i for i, ok in enumerate(present) if not ok]
    if missing_idx:
        if on_missing_image == "text":
            fallback = embedder.embed_texts([texts[i] for i in missing_idx])
            for slot, i in enumerate(missing_idx):
                embedding_col[i] = fallback[slot]
                modality_col[i] = "text"
        else:  # skip
            for i in missing_idx:
                modality_col[i] = "missing_image"

    gdf["embedding"] = embedding_col
    gdf["embedding_modality"] = modality_col

    return gdf


def cluster_descriptions(
    gdf_or_path: Union[gpd.GeoDataFrame, str, Path],
    n_clusters: Optional[int] = 10,
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B",
    embedding_backend: str = "auto",
    embedding_columns: Optional[list] = None,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    """
    Cluster VLM descriptions in a GeoDataFrame and return enriched results.

    Args:
        gdf_or_path: A GeoDataFrame or path to a GeoParquet file.
        n_clusters: Number of clusters (``None`` for auto-elbow).
        embedding_model: HuggingFace model for embedding.
        embedding_backend: ``"vllm"``, ``"transformers"`` or ``"auto"``.
        embedding_columns: Columns to join for embedding text.
        random_state: Random seed.

    Returns:
        GeoDataFrame with ``cluster`` and ``embedding_text`` columns.
    """
    from .clustering import ClusterConfig, SemanticClusterer
    from .embedding import ImageEmbedder

    if isinstance(gdf_or_path, (str, Path)):
        gdf = load_geoparquet(Path(gdf_or_path))
    else:
        gdf = gdf_or_path

    config = ClusterConfig(
        n_clusters=n_clusters,
        random_state=random_state,
        embedding_columns=embedding_columns
        or ["scene_narrative", "semantic_tags", "place_character"],
    )
    embedder = ImageEmbedder(model_name=embedding_model, backend=embedding_backend)
    clusterer = SemanticClusterer(embedder=embedder, config=config)

    gdf = gdf.copy()
    gdf["embedding_text"] = clusterer.build_embedding_text(gdf)
    gdf = clusterer.cluster(gdf, n_clusters=n_clusters)

    return gdf


def analyze_spatial(
    gdf: gpd.GeoDataFrame,
    column: str = "cluster",
    k_neighbors: int = 8,
) -> dict:
    """
    Run Global & Local Moran's I on a clustered GeoDataFrame.

    Args:
        gdf: GeoDataFrame with a *column* of cluster labels.
        column: Name of the cluster column.
        k_neighbors: Neighbours for KNN weight matrix.

    Returns:
        Dict with ``global`` (:class:`~geoai_vlm.spatial.MoranResult` per cluster)
        and ``gdf`` (enriched with LISA columns).
    """
    from .spatial import SpatialAnalyzer

    sa = SpatialAnalyzer(k_neighbors=k_neighbors)
    global_results = sa.moran_global(gdf, column=column)
    lisa_gdf = sa.moran_local(gdf, column=column)

    return {"global": global_results, "gdf": lisa_gdf}


def build_search_index(
    gdf_or_path: Union[gpd.GeoDataFrame, str, Path],
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B",
    embedding_backend: str = "auto",
    store_backend: str = "chromadb",
    text_column: str = "scene_narrative",
    image_dir: Optional[Union[str, Path]] = None,
    metadata_columns: Optional[list] = None,
    **store_kwargs,
):
    """
    Build a searchable vector index from pipeline output.

    Args:
        gdf_or_path: GeoDataFrame or path to GeoParquet.
        embedding_model: HuggingFace embedding model name.
        embedding_backend: ``"vllm"``, ``"transformers"`` or ``"auto"``.
        store_backend: ``"chromadb"`` or ``"faiss"``.
        text_column: Column with text to embed.
        image_dir: Directory with original images for multimodal embedding.
        metadata_columns: Extra metadata columns to store.
        **store_kwargs: Forwarded to the vector store backend.

    Returns:
        A :class:`~geoai_vlm.vectorstore.VectorDB` instance with a populated store.
    """
    from .embedding import ImageEmbedder
    from .vectorstore import VectorDB

    if isinstance(gdf_or_path, (str, Path)):
        gdf = load_geoparquet(Path(gdf_or_path))
    else:
        gdf = gdf_or_path

    embedder = ImageEmbedder(model_name=embedding_model, backend=embedding_backend)
    vdb = VectorDB(embedder=embedder, store_backend=store_backend, **store_kwargs)
    vdb.build(
        gdf,
        text_column=text_column,
        image_dir=image_dir,
        metadata_columns=metadata_columns,
    )
    return vdb


def search_similar(
    vector_db,
    query_text: Optional[str] = None,
    query_image: Optional[Union[str, Path]] = None,
    n_results: int = 10,
):
    """
    Search a vector index for similar items.

    Args:
        vector_db: A :class:`~geoai_vlm.vectorstore.VectorDB` instance.
        query_text: Text query string.
        query_image: Path to a query image.
        n_results: Number of results.

    Returns:
        DataFrame with ``id``, ``distance`` and stored metadata columns.
    """
    return vector_db.search(
        query_text=query_text,
        query_image=query_image,
        n_results=n_results,
    )
