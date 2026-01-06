# -*- coding: utf-8 -*-
"""
Downloader Module for GeoAI-VLM
================================
Wrapper around ZenSVI's MLYDownloader for geometry-based queries.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

from .geometry import (
    BaseQuery,
    PointQuery,
    LineQuery,
    BBoxQuery,
    PolygonQuery,
    PlaceQuery,
)


__all__ = [
    "MapillaryDownloader",
    "download_images",
]


class MapillaryDownloader:
    """
    Wrapper around ZenSVI's MLYDownloader for geometry-based image downloading.
    
    Args:
        mly_api_key: Mapillary API key
        verbosity: Verbosity level (0=silent, 1=progress, 2=debug)
        max_workers: Number of parallel download workers
        
    Example:
        >>> downloader = MapillaryDownloader(mly_api_key="YOUR_KEY")
        >>> gdf = downloader.download(
        ...     query=PointQuery(lat=41.0082, lon=28.9784, buffer_m=100),
        ...     output_dir="./images"
        ... )
    """
    
    def __init__(
        self,
        mly_api_key: str,
        verbosity: int = 1,
        max_workers: Optional[int] = None,
    ):
        self.mly_api_key = mly_api_key
        self.verbosity = verbosity
        self.max_workers = max_workers
        
        self._downloader = None
    
    @property
    def downloader(self):
        """Lazy-load ZenSVI MLYDownloader."""
        if self._downloader is None:
            from zensvi.download import MLYDownloader
            
            self._downloader = MLYDownloader(
                mly_api_key=self.mly_api_key,
                max_workers=self.max_workers,
                verbosity=self.verbosity,
            )
        return self._downloader
    
    def download(
        self,
        query: BaseQuery,
        output_dir: Union[str, Path],
        resolution: int = 1024,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metadata_only: bool = False,
        use_cache: bool = True,
        max_images: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Download Mapillary images based on a geospatial query.
        
        Args:
            query: Geospatial query (PointQuery, LineQuery, BBoxQuery, PolygonQuery, PlaceQuery)
            output_dir: Directory to save downloaded images
            resolution: Image resolution (256, 1024, 2048)
            start_date: Filter images after this date (YYYY-MM-DD)
            end_date: Filter images before this date (YYYY-MM-DD)
            metadata_only: If True, only download metadata without images
            use_cache: Use cached tile results
            max_images: Maximum number of images to download (None for all, sorted by distance)
            
        Returns:
            GeoDataFrame with image metadata and geometry
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare base download parameters based on query type
        download_kwargs = {
            "dir_output": str(output_dir),
            "resolution": resolution,
            "start_date": start_date,
            "end_date": end_date,
            "use_cache": use_cache,
            "buffer": query.buffer_m,
        }
        
        # Set input based on query type
        if isinstance(query, PointQuery):
            download_kwargs["lat"] = query.lat
            download_kwargs["lon"] = query.lon
            
        elif isinstance(query, PlaceQuery):
            download_kwargs["input_place_name"] = query.place_name
            
        elif isinstance(query, (LineQuery, BBoxQuery, PolygonQuery)):
            # Save geometry to temporary file
            gdf = query.to_geodataframe()
            temp_shp = output_dir / "cache" / "query_geometry.geojson"
            temp_shp.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(temp_shp, driver="GeoJSON")
            download_kwargs["input_shp_file"] = str(temp_shp)
        
        else:
            raise TypeError(f"Unknown query type: {type(query)}")
        
        # If max_images is set, use optimized download (get PIDs first, filter, then download)
        if max_images is not None and not metadata_only:
            return self._download_with_limit(
                query=query,
                output_dir=output_dir,
                resolution=resolution,
                max_images=max_images,
                download_kwargs=download_kwargs,
            )
        
        # Standard download (no max_images limit)
        download_kwargs["metadata_only"] = metadata_only
        self.downloader.download_svi(**download_kwargs)
        
        # Load and enhance metadata
        gdf = self._load_metadata(output_dir, query)
        
        return gdf
    
    def _download_with_limit(
        self,
        query: BaseQuery,
        output_dir: Path,
        resolution: int,
        max_images: int,
        download_kwargs: dict,
    ) -> gpd.GeoDataFrame:
        """
        Optimized download: get PIDs via tile API (fast), filter by distance, 
        then download only selected images directly from Mapillary API.
        Skips the slow URL-fetching step for all images.
        """
        if self.verbosity > 0:
            print(f"Getting image IDs from Mapillary (will select {max_images} nearest)...")
        
        # Step 1: Get only PIDs via ZenSVI (fast - only tile API, we'll interrupt before URL fetching)
        # We use a workaround: call _get_pids through a fresh downloader with proper setup
        from zensvi.download import MLYDownloader
        
        temp_downloader = MLYDownloader(
            mly_api_key=self.mly_api_key,
            max_workers=self.max_workers,
            verbosity=self.verbosity,
        )
        
        # Set up cache paths that _get_pids expects
        temp_downloader.dir_output = output_dir
        temp_downloader.cache_dir = output_dir / "cache"
        temp_downloader.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_downloader.cache_pids_raw = temp_downloader.cache_dir / "pids_raw.csv"
        temp_downloader.dir_cache = temp_downloader.cache_dir
        
        pids_path = output_dir / "mly_pids.csv"
        
        # Prepare kwargs for _get_pids - must include all expected keys
        # Use empty strings for file paths when not provided (ZenSVI expects this)
        pids_kwargs = {
            "lat": download_kwargs.get("lat"),
            "lon": download_kwargs.get("lon"),
            "input_place_name": download_kwargs.get("input_place_name", ""),
            "input_shp_file": download_kwargs.get("input_shp_file", ""),
            "input_csv_file": "",
            "buffer": download_kwargs.get("buffer", 0),
            "start_date": download_kwargs.get("start_date"),
            "end_date": download_kwargs.get("end_date"),
            "use_cache": True,
        }
        
        # Call _get_pids (only fetches tile data - FAST)
        temp_downloader._get_pids(path_pid=str(pids_path), **pids_kwargs)
        
        # Step 2: Load metadata and calculate distances
        gdf = self._load_metadata(output_dir, query)
        
        if len(gdf) == 0:
            return gdf
        
        if self.verbosity > 0:
            print(f"Found {len(gdf)} images total")
        
        # Step 3: Sort by distance and filter to max_images
        gdf = self._filter_by_distance(gdf, max_images)
        
        if self.verbosity > 0:
            print(f"Selected {len(gdf)} nearest images")
        
        # Step 4: Download only selected images directly from Mapillary API
        # (skips the slow URL-fetching step for all 22k+ images)
        self._download_selected_images_direct(gdf, output_dir, resolution)
        
        return gdf
        
        # Step 2: Load metadata and calculate distances
        gdf = self._load_metadata(output_dir, query)
        
        if len(gdf) == 0:
            return gdf
        
        if self.verbosity > 0:
            print(f"Found {len(gdf)} images total")
        
        # Step 3: Sort by distance and filter to max_images
        gdf = self._filter_by_distance(gdf, max_images)
        
        if self.verbosity > 0:
            print(f"Selected {len(gdf)} nearest images")
        
        # Step 4: Get URLs and download only selected images
        self._download_selected_images_direct(
            gdf=gdf,
            output_dir=output_dir,
            resolution=resolution,
        )
        
        return gdf
    
    def _filter_by_distance(
        self,
        gdf: gpd.GeoDataFrame,
        max_images: int,
    ) -> gpd.GeoDataFrame:
        """Sort by distance and return top N images."""
        if len(gdf) <= max_images:
            return gdf
        
        # Find distance column
        distance_col = None
        for col in ["distance_to_query_m", "distance_to_line_m", "distance_to_polygon_m"]:
            if col in gdf.columns:
                distance_col = col
                break
        
        if distance_col:
            gdf = gdf.sort_values(distance_col).head(max_images).copy()
        else:
            gdf = gdf.head(max_images).copy()
        
        return gdf
    
    def _download_selected_images(
        self,
        gdf: gpd.GeoDataFrame,
        output_dir: Path,
        resolution: int,
    ) -> None:
        """
        Download only selected images using URLs from pids_urls.csv.
        """
        if len(gdf) == 0:
            return
        
        # Resolution to URL field mapping (ZenSVI uses 'url' for 1024)
        res_field = {256: "thumb_256_url", 1024: "url", 2048: "thumb_2048_url"}
        url_field = res_field.get(resolution, "url")
        
        # Try to load URLs from pids_urls.csv (created by ZenSVI)
        urls_path = output_dir / "pids_urls.csv"
        
        if urls_path.exists():
            urls_df = pd.read_csv(urls_path)
            urls_df["id"] = urls_df["id"].astype(str)
            
            # Filter to selected image IDs
            selected_ids = set(gdf["image_id"].tolist())
            urls_df = urls_df[urls_df["id"].isin(selected_ids)]
            
            # Create output directory
            img_output_dir = output_dir / "mly_svi"
            img_output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.verbosity > 0:
                print(f"Downloading {len(urls_df)} images...")
            
            # Download images
            for _, row in tqdm(urls_df.iterrows(), total=len(urls_df), 
                              desc="Downloading images", disable=self.verbosity < 1):
                img_id = row["id"]
                
                # Try the specified resolution field, fallback to 'url'
                url = row.get(url_field)
                if pd.isna(url) or url is None:
                    url = row.get("url")
                if pd.isna(url) or url is None:
                    continue
                
                img_path = img_output_dir / f"{img_id}.jpg"
                
                if img_path.exists():
                    continue
                
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                except Exception as e:
                    if self.verbosity > 1:
                        print(f"Failed to download {img_id}: {e}")
        else:
            # Fallback: download directly using Mapillary API
            self._download_selected_images_direct(gdf, output_dir, resolution)
    
    def _download_selected_images_direct(
        self,
        gdf: gpd.GeoDataFrame,
        output_dir: Path,
        resolution: int,
    ) -> None:
        """
        Fallback: Download images directly using Mapillary API (when pids_urls.csv not available).
        """
        res_field = {256: "thumb_256_url", 1024: "thumb_1024_url", 2048: "thumb_2048_url"}
        url_field = res_field.get(resolution, "thumb_1024_url")
        
        img_output_dir = output_dir / "mly_svi"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbosity > 0:
            print(f"Downloading {len(gdf)} images via API...")
        
        for img_id in tqdm(gdf["image_id"].tolist(), desc="Downloading", disable=self.verbosity < 1):
            img_path = img_output_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                continue
            
            try:
                url = self._get_image_url(img_id, url_field)
                if url:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    with open(img_path, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                if self.verbosity > 1:
                    print(f"Failed to download {img_id}: {e}")
    
    def _get_image_url(self, image_id: str, url_field: str) -> Optional[str]:
        """Get image URL from Mapillary API."""
        api_url = f"https://graph.mapillary.com/{image_id}"
        params = {
            "access_token": self.mly_api_key,
            "fields": url_field,
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get(url_field)
        except Exception:
            return None
    
    def _load_metadata(
        self,
        output_dir: Path,
        query: BaseQuery,
    ) -> gpd.GeoDataFrame:
        """Load metadata CSV and convert to GeoDataFrame with distances."""
        # Find the metadata file
        metadata_path = output_dir / "mly_pids.csv"
        
        if not metadata_path.exists():
            # Try alternative paths
            cache_path = output_dir / "cache" / "pids_raw.csv"
            if cache_path.exists():
                metadata_path = cache_path
            else:
                # Also check old cache_zensvi path for backwards compatibility
                legacy_path = output_dir / "cache_zensvi" / "pids_raw.csv"
                if legacy_path.exists():
                    metadata_path = legacy_path
                else:
                    print(f"Warning: No metadata file found in {output_dir}")
                    return gpd.GeoDataFrame()
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        if len(df) == 0:
            return gpd.GeoDataFrame()
        
        # Create geometry from lat/lon
        geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Add image paths (will be in mly_svi subfolder)
        gdf["image_path"] = gdf["id"].apply(
            lambda x: str(output_dir / "mly_svi" / f"{x}.jpg")
        )
        gdf["image_id"] = gdf["id"].astype(str)
        
        # Convert captured_at to datetime
        if "captured_at" in gdf.columns:
            gdf["captured_at"] = pd.to_datetime(gdf["captured_at"], unit="ms")
        
        # Add distance columns based on query type
        gdf = query.add_distances(gdf)
        
        # Filter to nearest if applicable
        if isinstance(query, PointQuery) and query.nearest_only:
            gdf = query.filter_nearest(gdf)
        
        return gdf
    
    def get_metadata_only(
        self,
        query: BaseQuery,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Get metadata without downloading images.
        
        Args:
            query: Geospatial query
            start_date: Filter images after this date (YYYY-MM-DD)
            end_date: Filter images before this date (YYYY-MM-DD)
            
        Returns:
            GeoDataFrame with image metadata
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            return self.download(
                query=query,
                output_dir=temp_dir,
                metadata_only=True,
                start_date=start_date,
                end_date=end_date,
            )


def download_images(
    query: BaseQuery,
    mly_api_key: str,
    output_dir: Union[str, Path],
    resolution: int = 1024,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metadata_only: bool = False,
    max_images: Optional[int] = None,
    verbosity: int = 1,
) -> gpd.GeoDataFrame:
    """
    Convenience function to download Mapillary images.
    
    Args:
        query: Geospatial query (PointQuery, LineQuery, etc.)
        mly_api_key: Mapillary API key
        output_dir: Directory to save images
        resolution: Image resolution (256, 1024, 2048)
        start_date: Filter images after this date
        end_date: Filter images before this date
        metadata_only: Only download metadata
        max_images: Maximum number of images (sorted by distance)
        verbosity: Verbosity level
        
    Returns:
        GeoDataFrame with image metadata and distances
        
    Example:
        >>> from geoai_vlm import PointQuery, download_images
        >>> gdf = download_images(
        ...     query=PointQuery(lat=41.0082, lon=28.9784, buffer_m=100),
        ...     mly_api_key="YOUR_KEY",
        ...     output_dir="./images",
        ...     max_images=50  # Only download 50 nearest images
        ... )
    """
    downloader = MapillaryDownloader(
        mly_api_key=mly_api_key,
        verbosity=verbosity,
    )
    
    return downloader.download(
        query=query,
        output_dir=output_dir,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
        metadata_only=metadata_only,
        max_images=max_images,
    )
