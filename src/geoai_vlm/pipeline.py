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
from shapely.geometry import LineString, Polygon

from .describer import ImageDescriber
from .downloader import MapillaryDownloader
from .geometry import (
    BaseQuery,
    BBoxQuery,
    LineQuery,
    PlaceQuery,
    PointQuery,
    PolygonQuery,
)
from .io import merge_metadata_and_descriptions, save_geoparquet, to_geodataframe


__all__ = [
    "describe_place",
    "describe_point",
    "describe_line",
    "describe_bbox",
    "describe_polygon",
    "describe_query",
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
    do_export(result_gdf, output_dir, "results", export_formats)
    
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
