# -*- coding: utf-8 -*-
"""
Geometry Module for GeoAI-VLM
=============================
Provides geospatial query classes and distance calculation utilities
for point, line, polygon, and bounding box queries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from haversine import haversine, Unit
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import nearest_points, transform
from shapely import get_coordinates


__all__ = [
    "BaseQuery",
    "PointQuery",
    "LineQuery",
    "BBoxQuery",
    "PolygonQuery",
    "PlaceQuery",
    "calculate_point_distance",
    "calculate_line_distance",
    "add_distance_columns",
]


def calculate_point_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, unit: Unit = Unit.METERS
) -> float:
    """
    Calculate haversine distance between two points.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        unit: Distance unit (default: meters)
        
    Returns:
        Distance in specified unit
    """
    return haversine((lat1, lon1), (lat2, lon2), unit=unit)


def calculate_line_distance(
    point: Point, line: LineString, unit: Unit = Unit.METERS
) -> Tuple[float, float]:
    """
    Calculate perpendicular distance from point to line and distance along line.
    
    Args:
        point: Shapely Point geometry
        line: Shapely LineString geometry
        unit: Distance unit (default: meters)
        
    Returns:
        Tuple of (perpendicular_distance, distance_along_line)
    """
    # Find nearest point on line
    nearest_on_line = nearest_points(point, line)[1]
    
    # Calculate perpendicular distance using haversine
    perp_distance = calculate_point_distance(
        point.y, point.x,
        nearest_on_line.y, nearest_on_line.x,
        unit=unit
    )
    
    # Calculate distance along line from start to nearest point
    # Project the line to calculate distance along it
    line_coords = list(line.coords)
    total_distance = 0.0
    
    for i in range(len(line_coords) - 1):
        seg_start = Point(line_coords[i])
        seg_end = Point(line_coords[i + 1])
        
        # Check if nearest point is on this segment
        segment = LineString([seg_start, seg_end])
        if segment.distance(nearest_on_line) < 1e-8:  # Small tolerance
            # Add distance from segment start to nearest point
            total_distance += calculate_point_distance(
                seg_start.y, seg_start.x,
                nearest_on_line.y, nearest_on_line.x,
                unit=unit
            )
            break
        else:
            # Add full segment length
            total_distance += calculate_point_distance(
                seg_start.y, seg_start.x,
                seg_end.y, seg_end.x,
                unit=unit
            )
    
    return perp_distance, total_distance


def add_distance_columns(
    gdf: gpd.GeoDataFrame,
    query_geometry: Union[Point, LineString, Polygon],
    query_type: str = "point"
) -> gpd.GeoDataFrame:
    """
    Add distance columns to GeoDataFrame based on query geometry.
    
    Args:
        gdf: GeoDataFrame with image locations (must have geometry column)
        query_geometry: The query geometry to calculate distances from
        query_type: Type of query ("point", "line", "polygon", "bbox")
        
    Returns:
        GeoDataFrame with added distance column(s)
    """
    if len(gdf) == 0:
        return gdf
    
    gdf = gdf.copy()
    
    if query_type == "point":
        # Distance to query point
        query_lat, query_lon = query_geometry.y, query_geometry.x
        gdf["distance_to_query_m"] = gdf.geometry.apply(
            lambda geom: calculate_point_distance(
                geom.y, geom.x, query_lat, query_lon
            )
        )
        
    elif query_type == "line":
        # Distance to line and along line
        distances = gdf.geometry.apply(
            lambda geom: calculate_line_distance(geom, query_geometry)
        )
        gdf["distance_to_line_m"] = distances.apply(lambda x: x[0])
        gdf["distance_along_line_m"] = distances.apply(lambda x: x[1])
        
    elif query_type in ("polygon", "bbox"):
        # Distance to polygon centroid (or 0 if inside)
        centroid = query_geometry.centroid
        gdf["distance_to_centroid_m"] = gdf.geometry.apply(
            lambda geom: calculate_point_distance(
                geom.y, geom.x, centroid.y, centroid.x
            )
        )
        # Also add flag for whether point is inside polygon
        gdf["inside_query"] = gdf.geometry.within(query_geometry)
    
    return gdf


@dataclass
class BaseQuery(ABC):
    """Abstract base class for geospatial queries."""
    
    buffer_m: float = 50.0
    """Buffer distance in meters for the query."""
    
    @abstractmethod
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert query to GeoDataFrame for ZenSVI."""
        pass
    
    @abstractmethod
    def get_geometry(self) -> Union[Point, LineString, Polygon]:
        """Get the raw query geometry."""
        pass
    
    @property
    @abstractmethod
    def query_type(self) -> str:
        """Return the query type string."""
        pass
    
    def get_buffered_geometry(self) -> Polygon:
        """
        Get buffered geometry using metric CRS for accurate buffering.
        
        Returns:
            Buffered polygon in EPSG:4326
        """
        geom = self.get_geometry()
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        
        if self.buffer_m > 0:
            # Project to UTM for accurate metric buffer
            gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
            gdf_projected["geometry"] = gdf_projected.buffer(self.buffer_m)
            gdf = gdf_projected.to_crs("EPSG:4326")
        
        return gdf.geometry.iloc[0]
    
    def add_distances(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add distance columns to results based on query type."""
        return add_distance_columns(gdf, self.get_geometry(), self.query_type)


@dataclass
class PointQuery(BaseQuery):
    """
    Query images near a specific point.
    
    Args:
        lat: Latitude of the query point
        lon: Longitude of the query point
        buffer_m: Search radius in meters (default: 50)
        nearest_only: If True, return only the single closest image
        
    Example:
        >>> query = PointQuery(lat=41.0082, lon=28.9784, buffer_m=100)
        >>> gdf = query.to_geodataframe()
    """
    
    lat: float = 0.0
    lon: float = 0.0
    nearest_only: bool = False
    buffer_m: float = 50.0
    
    def __post_init__(self):
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.lon}")
        if self.buffer_m < 0:
            raise ValueError(f"Buffer must be non-negative, got {self.buffer_m}")
    
    def get_geometry(self) -> Point:
        return Point(self.lon, self.lat)
    
    @property
    def query_type(self) -> str:
        return "point"
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"lat": [self.lat], "lon": [self.lon]},
            geometry=[self.get_geometry()],
            crs="EPSG:4326"
        )
    
    def filter_nearest(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Filter to only the nearest image if nearest_only is True."""
        if not self.nearest_only or len(gdf) == 0:
            return gdf
        
        # Ensure distance column exists
        if "distance_to_query_m" not in gdf.columns:
            gdf = self.add_distances(gdf)
        
        # Return only the closest image
        return gdf.nsmallest(1, "distance_to_query_m")


@dataclass
class LineQuery(BaseQuery):
    """
    Query images along a line (street, route, path).
    
    Args:
        geometry: Shapely LineString or list of (lon, lat) coordinate tuples
        buffer_m: Buffer distance from line in meters (default: 25)
        
    Example:
        >>> from shapely.geometry import LineString
        >>> line = LineString([(28.97, 41.01), (28.98, 41.02)])
        >>> query = LineQuery(geometry=line, buffer_m=25)
    """
    
    geometry: Union[LineString, List[Tuple[float, float]]] = None
    buffer_m: float = 25.0
    
    def __post_init__(self):
        if self.geometry is None:
            raise ValueError("geometry is required for LineQuery")
        
        # Convert list of tuples to LineString
        if isinstance(self.geometry, list):
            if len(self.geometry) < 2:
                raise ValueError("LineQuery requires at least 2 coordinate pairs")
            self.geometry = LineString(self.geometry)
        
        if not isinstance(self.geometry, (LineString, MultiLineString)):
            raise TypeError(f"geometry must be LineString, got {type(self.geometry)}")
        
        if self.buffer_m <= 0:
            raise ValueError(f"Buffer must be positive for LineQuery, got {self.buffer_m}")
    
    def get_geometry(self) -> LineString:
        return self.geometry
    
    @property
    def query_type(self) -> str:
        return "line"
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=[self.geometry],
            crs="EPSG:4326"
        )


@dataclass
class BBoxQuery(BaseQuery):
    """
    Query images within a bounding box.
    
    Args:
        minx: Minimum longitude (west)
        miny: Minimum latitude (south)
        maxx: Maximum longitude (east)
        maxy: Maximum latitude (north)
        
    Example:
        >>> query = BBoxQuery(minx=28.97, miny=41.00, maxx=28.99, maxy=41.02)
    """
    
    minx: float = 0.0
    miny: float = 0.0
    maxx: float = 0.0
    maxy: float = 0.0
    buffer_m: float = 0.0  # No buffer needed for bbox
    
    def __post_init__(self):
        if self.minx >= self.maxx:
            raise ValueError(f"minx ({self.minx}) must be less than maxx ({self.maxx})")
        if self.miny >= self.maxy:
            raise ValueError(f"miny ({self.miny}) must be less than maxy ({self.maxy})")
    
    def get_geometry(self) -> Polygon:
        return box(self.minx, self.miny, self.maxx, self.maxy)
    
    @property
    def query_type(self) -> str:
        return "bbox"
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=[self.get_geometry()],
            crs="EPSG:4326"
        )


@dataclass
class PolygonQuery(BaseQuery):
    """
    Query images within a polygon.
    
    Args:
        geometry: Shapely Polygon or list of (lon, lat) coordinate tuples
        buffer_m: Additional buffer in meters (default: 0)
        
    Example:
        >>> from shapely.geometry import Polygon
        >>> poly = Polygon([(28.97, 41.00), (28.99, 41.00), (28.99, 41.02), (28.97, 41.02)])
        >>> query = PolygonQuery(geometry=poly)
    """
    
    geometry: Union[Polygon, List[Tuple[float, float]]] = None
    buffer_m: float = 0.0
    
    def __post_init__(self):
        if self.geometry is None:
            raise ValueError("geometry is required for PolygonQuery")
        
        # Convert list of tuples to Polygon
        if isinstance(self.geometry, list):
            if len(self.geometry) < 3:
                raise ValueError("PolygonQuery requires at least 3 coordinate pairs")
            self.geometry = Polygon(self.geometry)
        
        if not isinstance(self.geometry, (Polygon, MultiPolygon)):
            raise TypeError(f"geometry must be Polygon, got {type(self.geometry)}")
    
    def get_geometry(self) -> Polygon:
        return self.geometry
    
    @property
    def query_type(self) -> str:
        return "polygon"
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            geometry=[self.geometry],
            crs="EPSG:4326"
        )


@dataclass
class PlaceQuery(BaseQuery):
    """
    Query images by place name using Nominatim/OSMnx geocoding.
    
    Args:
        place_name: OSM-compatible place name (e.g., "Sultanahmet, Istanbul")
        buffer_m: Additional buffer in meters (default: 0 for polygon results)
        
    Example:
        >>> query = PlaceQuery(place_name="Fatih, Istanbul", buffer_m=0)
    """
    
    place_name: str = ""
    buffer_m: float = 0.0
    _geocoded_gdf: gpd.GeoDataFrame = field(default=None, repr=False)
    
    def __post_init__(self):
        if not self.place_name:
            raise ValueError("place_name is required for PlaceQuery")
    
    def _geocode(self) -> gpd.GeoDataFrame:
        """Geocode the place name using OSMnx."""
        if self._geocoded_gdf is not None:
            return self._geocoded_gdf
        
        try:
            import osmnx as ox
            self._geocoded_gdf = ox.geocoder.geocode_to_gdf(self.place_name)
            if len(self._geocoded_gdf) == 0:
                raise ValueError(f"Place not found: {self.place_name}")
            return self._geocoded_gdf
        except Exception as e:
            raise ValueError(f"Failed to geocode '{self.place_name}': {e}")
    
    def get_geometry(self) -> Union[Point, Polygon]:
        gdf = self._geocode()
        return gdf.geometry.iloc[0]
    
    @property
    def query_type(self) -> str:
        geom = self.get_geometry()
        if isinstance(geom, Point):
            return "point"
        return "polygon"
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return self._geocode()
