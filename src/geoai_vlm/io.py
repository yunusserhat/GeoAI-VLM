# -*- coding: utf-8 -*-
"""
I/O Module for GeoAI-VLM
========================
GeoParquet I/O utilities with native geometry support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


__all__ = [
    "save_geoparquet",
    "load_geoparquet",
    "load_results",
    "list_results",
    "summarize_results",
    "to_geodataframe",
    "export_formats",
    "merge_metadata_and_descriptions",
]


def to_geodataframe(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with lat/lon columns to GeoDataFrame.
    
    Args:
        df: Input DataFrame
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        crs: Coordinate reference system
        
    Returns:
        GeoDataFrame with Point geometry
    """
    if len(df) == 0:
        return gpd.GeoDataFrame(df, geometry=[], crs=crs)
    
    # Check if already a GeoDataFrame with geometry
    if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None:
        if df.crs is None:
            df = df.set_crs(crs)
        return df
    
    # Create geometry from lat/lon
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"DataFrame must have '{lat_col}' and '{lon_col}' columns")
    
    geometry = [
        Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])
    ]
    
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


def save_geoparquet(
    gdf: gpd.GeoDataFrame,
    path: Union[str, Path],
    compression: str = "snappy",
    index: bool = False,
) -> Path:
    """
    Save GeoDataFrame as GeoParquet with native geometry.
    
    Args:
        gdf: GeoDataFrame to save
        path: Output file path
        compression: Compression algorithm (snappy, gzip, zstd, etc.)
        index: Whether to include index in output
        
    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure it's a GeoDataFrame with valid geometry
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame")
    
    # Set CRS if missing
    if gdf.crs is None and len(gdf) > 0:
        gdf = gdf.set_crs("EPSG:4326")
    
    # Save as GeoParquet
    gdf.to_parquet(
        path,
        compression=compression,
        index=index,
    )
    
    print(f"Saved GeoParquet: {path} ({len(gdf)} features)")
    return path


def load_geoparquet(
    path: Union[str, Path],
) -> gpd.GeoDataFrame:
    """
    Load GeoParquet file as GeoDataFrame.
    
    Args:
        path: Path to GeoParquet file
        
    Returns:
        GeoDataFrame with geometry
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return gpd.read_parquet(path)


def load_results(
    path: Union[str, Path],
    columns: Optional[list] = None,
    filters: Optional[list] = None,
) -> gpd.GeoDataFrame:
    """
    Load GeoAI-VLM results from a parquet file with optional filtering.
    
    Args:
        path: Path to parquet file (geoparquet or regular parquet)
        columns: List of columns to load (None for all)
        filters: PyArrow filters for row-level filtering
                 e.g., [("land_use_primary", "==", "commercial")]
                 
    Returns:
        GeoDataFrame with results
        
    Example:
        >>> # Load only specific columns
        >>> gdf = load_results("results.parquet", columns=["image_id", "land_use_primary"])
        >>> 
        >>> # Load with filters
        >>> gdf = load_results("results.parquet", filters=[("distance_to_query_m", "<", 50)])
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Try loading as geoparquet first
    try:
        gdf = gpd.read_parquet(path, columns=columns, filters=filters)
        return gdf
    except Exception:
        # Fallback to regular parquet and convert
        df = pd.read_parquet(path, columns=columns, filters=filters)
        if "lat" in df.columns and "lon" in df.columns:
            return to_geodataframe(df)
        return gpd.GeoDataFrame(df)


def list_results(
    directory: Union[str, Path],
    pattern: str = "**/*.parquet",
) -> list:
    """
    List all parquet files in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for matching files
        
    Returns:
        List of Path objects for matching files
        
    Example:
        >>> files = list_results("./output")
        >>> for f in files:
        ...     print(f"{f.name}: {f.stat().st_size / 1024:.1f} KB")
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    return sorted(directory.glob(pattern))


def summarize_results(
    path: Union[str, Path],
    show_columns: bool = True,
    show_stats: bool = True,
) -> dict:
    """
    Get summary statistics for a parquet results file.
    
    Args:
        path: Path to parquet file
        show_columns: Print column information
        show_stats: Print basic statistics
        
    Returns:
        Dictionary with summary information
        
    Example:
        >>> summary = summarize_results("results.parquet")
        >>> print(f"Total images: {summary['n_rows']}")
    """
    path = Path(path)
    gdf = load_geoparquet(path)
    
    summary = {
        "path": str(path),
        "n_rows": len(gdf),
        "n_columns": len(gdf.columns),
        "columns": list(gdf.columns),
        "crs": str(gdf.crs) if gdf.crs else None,
        "file_size_kb": path.stat().st_size / 1024,
    }
    
    # Add value counts for key categorical columns
    categorical_cols = ["land_use_primary", "street_type", "place_character"]
    for col in categorical_cols:
        if col in gdf.columns:
            summary[f"{col}_distribution"] = gdf[col].value_counts().to_dict()
    
    # Add statistics for distance columns
    distance_cols = ["distance_to_query_m", "distance_to_line_m", "distance_along_line_m"]
    for col in distance_cols:
        if col in gdf.columns:
            summary[f"{col}_stats"] = {
                "min": float(gdf[col].min()),
                "max": float(gdf[col].max()),
                "mean": float(gdf[col].mean()),
                "median": float(gdf[col].median()),
            }
    
    if show_columns:
        print(f"\n📁 {path.name}")
        print(f"   {summary['n_rows']} rows × {summary['n_columns']} columns")
        print(f"   Size: {summary['file_size_kb']:.1f} KB")
        print(f"   CRS: {summary['crs']}")
        print(f"\n   Columns: {', '.join(summary['columns'][:10])}", end="")
        if len(summary['columns']) > 10:
            print(f" ... (+{len(summary['columns']) - 10} more)")
        else:
            print()
    
    if show_stats:
        for col in categorical_cols:
            if f"{col}_distribution" in summary:
                print(f"\n   {col}:")
                for val, count in list(summary[f"{col}_distribution"].items())[:5]:
                    print(f"      {val}: {count}")
        
        for col in distance_cols:
            if f"{col}_stats" in summary:
                stats = summary[f"{col}_stats"]
                print(f"\n   {col}:")
                print(f"      min: {stats['min']:.1f}m, max: {stats['max']:.1f}m, mean: {stats['mean']:.1f}m")
    
    return summary


def export_formats(
    gdf: gpd.GeoDataFrame,
    output_dir: Union[str, Path],
    basename: str = "results",
    formats: str = "geoparquet geojson csv",
) -> dict:
    """
    Export GeoDataFrame to multiple formats.
    
    Args:
        gdf: GeoDataFrame to export
        output_dir: Output directory
        basename: Base filename (without extension)
        formats: Space-separated list of formats to export
                 Options: geoparquet, geojson, csv, gpkg (GeoPackage)
                 
    Returns:
        Dictionary mapping format names to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    format_list = formats.lower().split()
    output_paths = {}
    
    for fmt in format_list:
        if fmt == "geoparquet":
            path = output_dir / f"{basename}.parquet"
            save_geoparquet(gdf, path)
            output_paths["geoparquet"] = path
            
        elif fmt == "geojson":
            path = output_dir / f"{basename}.geojson"
            gdf.to_file(path, driver="GeoJSON")
            print(f"Saved GeoJSON: {path}")
            output_paths["geojson"] = path
            
        elif fmt == "csv":
            path = output_dir / f"{basename}.csv"
            # Drop geometry for CSV, keep lat/lon
            df = pd.DataFrame(gdf.drop(columns="geometry"))
            df.to_csv(path, index=False)
            print(f"Saved CSV: {path}")
            output_paths["csv"] = path
            
        elif fmt == "gpkg":
            path = output_dir / f"{basename}.gpkg"
            gdf.to_file(path, driver="GPKG")
            print(f"Saved GeoPackage: {path}")
            output_paths["gpkg"] = path
            
        else:
            print(f"Warning: Unknown format '{fmt}', skipping")
    
    return output_paths


def merge_metadata_and_descriptions(
    metadata_gdf: gpd.GeoDataFrame,
    descriptions_df: pd.DataFrame,
    on: str = "image_id",
) -> gpd.GeoDataFrame:
    """
    Merge image metadata with VLM descriptions.
    
    Args:
        metadata_gdf: GeoDataFrame with image metadata (from downloader)
        descriptions_df: DataFrame with VLM descriptions (from describer)
        on: Column to join on
        
    Returns:
        Merged GeoDataFrame
    """
    # Ensure both have the join column
    if on not in metadata_gdf.columns:
        raise ValueError(f"metadata_gdf missing column: {on}")
    if on not in descriptions_df.columns:
        raise ValueError(f"descriptions_df missing column: {on}")
    
    # Get description columns (exclude image_path which may conflict)
    desc_cols = [c for c in descriptions_df.columns if c not in metadata_gdf.columns or c == on]
    desc_subset = descriptions_df[desc_cols]
    
    # Merge
    merged = metadata_gdf.merge(desc_subset, on=on, how="left")
    
    # Ensure result is GeoDataFrame
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=metadata_gdf.crs)
