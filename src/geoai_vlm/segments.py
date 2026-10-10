# -*- coding: utf-8 -*-
"""
Street-segment indicators for GeoAI-VLM
=======================================
Move image-level measurements to the street segment, the unit that walking
and cycling research usually analyses, and say how well each segment is
actually observed.

Two steps:

1. :func:`snap_images_to_segments` matches every image point to the nearest
   street segment within ``max_distance_m`` (default 20 m). Equal distances
   are resolved by a fixed rule (smaller segment id), and a match whose
   runner-up lies within ``ambiguity_margin_m`` (default 2 m) is flagged as
   ambiguous rather than silently trusted.
2. :func:`aggregate_segments` summarises the matched images per segment:
   image and sequence counts, capture-date range, the share of the segment's
   length within ``support_length_m / 2`` of an image (default 25 m support),
   and summaries of the measurement columns you name -- means for numbers,
   rates for booleans, and for audit states the ``present`` rate among
   definite observations with every other state counted separately.

Every segment of the network is kept in the output, including segments with
no image, so coverage always has its full denominator. The analysis unit and
all matching rules are written to the output metadata.

Image counts are not network coverage: twenty images at one corner cover one
corner. Both are reported, separately.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, substring


__all__ = [
    "SEGMENT_DEFAULTS",
    "STATE_COUNT_VALUES",
    "prepare_segments",
    "segments_from_graph",
    "load_osm_segments",
    "snap_images_to_segments",
    "aggregate_segments",
    "write_segment_parquet",
    "read_segment_metadata",
]

#: Default matching and support parameters, in metres.
SEGMENT_DEFAULTS = {
    "max_distance_m": 20.0,
    "support_length_m": 25.0,
    "ambiguity_margin_m": 2.0,
}

#: Values counted separately for an audit state column.
STATE_COUNT_VALUES = (
    "present", "absent", "not_visible", "uncertain", "not_assessed", "invalid", "failed",
)

METADATA_KEY = b"geoai_vlm"
_TIE_TOLERANCE_M = 1e-3


def _version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except ImportError:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# Network preparation
# ---------------------------------------------------------------------------
def _as_linestring(geom) -> Optional[LineString]:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
    raise ValueError(
        f"segment geometry must be a LineString (or a mergeable MultiLineString), "
        f"got {geom.geom_type}; explode multi-part streets into separate segments first"
    )


def prepare_segments(
    lines: gpd.GeoDataFrame,
    id_column: Optional[str] = None,
    segment_id_column: str = "segment_id",
) -> gpd.GeoDataFrame:
    """Validate a street-line layer and give every segment a stable id.

    Args:
        lines: Line geometries with a CRS.
        id_column: Existing unique id column to use; by default the row
            order is used (``"0"``, ``"1"``, ...).
        segment_id_column: Name of the id column in the output.

    Returns:
        A copy with string ids in ``segment_id_column`` and LineString
        geometries. Duplicate geometries (a street stored once per direction)
        raise a warning, since they make every nearby image ambiguous.
    """
    if lines.crs is None:
        raise ValueError("segments need a CRS")
    out = lines.copy()
    if id_column is not None:
        ids = out[id_column].astype(str)
    elif segment_id_column in out.columns:
        ids = out[segment_id_column].astype(str)
    else:
        ids = pd.Series([str(i) for i in range(len(out))], index=out.index)
    if ids.duplicated().any():
        raise ValueError(f"segment ids must be unique; duplicated: {sorted(ids[ids.duplicated()].unique())[:5]}")
    out[segment_id_column] = ids.values
    out = out.set_geometry(out.geometry.apply(_as_linestring))
    out = out[out.geometry.notna()].copy()

    normalised = out.geometry.apply(lambda g: shapely.normalize(g).wkb)
    if normalised.duplicated().any():
        warnings.warn(
            f"{int(normalised.duplicated().sum())} segment(s) duplicate another segment's "
            "geometry (e.g. one street stored per direction); nearby images will be "
            "flagged ambiguous. Deduplicate with segments_from_graph(dedupe_bidirectional=True).",
            stacklevel=2,
        )
    return out


def segments_from_graph(
    graph: Any,
    dedupe_bidirectional: bool = True,
    class_attribute: str = "highway",
    crs: Any = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convert an OSMnx-style street graph to one row per street segment.

    Works with any ``networkx`` MultiDiGraph whose nodes carry ``x``/``y``
    and whose edges may carry a ``geometry`` (as OSMnx produces). An edge
    without geometry becomes a straight line between its nodes.

    Args:
        graph: The street graph.
        dedupe_bidirectional: Keep one segment per physical street. OSMnx
            stores a two-way street twice (u->v and v->u); without this every
            image beside it would match two identical segments.
        class_attribute: Edge attribute holding the road class; lists (OSM
            ways merged during simplification) are joined with ``"|"``.
        crs: CRS of the node coordinates (OSMnx graphs are EPSG:4326 unless
            projected).

    Returns:
        GeoDataFrame with ``segment_id`` (``"u-v-key"``), ``u``, ``v``,
        ``key``, ``road_class`` and LineString geometry.
    """
    graph_crs = getattr(graph, "graph", {}).get("crs") if hasattr(graph, "graph") else None
    nodes = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}
    rows, seen = [], set()
    for u, v, key, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([nodes[u], nodes[v]])
        if dedupe_bidirectional:
            signature = shapely.normalize(geom).wkb
            if signature in seen:
                continue
            seen.add(signature)
        road_class = data.get(class_attribute)
        if isinstance(road_class, (list, tuple)):
            road_class = "|".join(sorted(str(c) for c in road_class))
        rows.append(
            {
                "segment_id": f"{u}-{v}-{key}",
                "u": u,
                "v": v,
                "key": key,
                "road_class": road_class,
                "osmid": data.get("osmid") if not isinstance(data.get("osmid"), list) else "|".join(map(str, data.get("osmid"))),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=graph_crs or crs)


def load_osm_segments(
    place: Optional[str] = None,
    polygon: Any = None,
    network_type: str = "walk",
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Download a street network with OSMnx and return its segments.

    Needs the optional ``network`` extra (``pip install 'geoai-vlm[network]'``)
    and network access to OpenStreetMap services. Give a place name or a
    polygon in EPSG:4326; nothing is fixed to any city.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "load_osm_segments needs OSMnx. Install it with: pip install 'geoai-vlm[network]'"
        ) from exc
    if (place is None) == (polygon is None):
        raise ValueError("give exactly one of place or polygon")
    if place is not None:
        graph = ox.graph_from_place(place, network_type=network_type, **kwargs)
    else:
        graph = ox.graph_from_polygon(polygon, network_type=network_type, **kwargs)
    return segments_from_graph(graph)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _metric_crs(segments: gpd.GeoDataFrame, metric_crs: Any):
    if metric_crs is not None:
        return metric_crs
    if segments.crs is not None and segments.crs.is_projected:
        return segments.crs
    return segments.estimate_utm_crs()


def snap_images_to_segments(
    images: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    max_distance_m: float = SEGMENT_DEFAULTS["max_distance_m"],
    ambiguity_margin_m: float = SEGMENT_DEFAULTS["ambiguity_margin_m"],
    segment_id_column: str = "segment_id",
    metric_crs: Any = None,
) -> gpd.GeoDataFrame:
    """Match each image point to its nearest street segment.

    Rule: the nearest segment within ``max_distance_m`` wins. Distances equal
    to within 1 mm are a tie, resolved by the smaller segment id (string
    order), so the result does not depend on row order. When the runner-up
    lies within ``ambiguity_margin_m`` of the winner the match is kept but
    flagged (``snap_status="ambiguous"``).

    Args:
        images: Image points with a CRS.
        segments: Street segments with a CRS and a ``segment_id_column``
            (see :func:`prepare_segments`).
        max_distance_m: Matching radius in metres.
        ambiguity_margin_m: Runner-up margin in metres.
        segment_id_column: Segment id column.
        metric_crs: Projected CRS for distances; by default the segments' own
            projected CRS or a local UTM zone.

    Returns:
        A copy of *images* with ``segment_id``, ``snap_distance_m``,
        ``snap_offset_m`` (position along the segment from its start),
        ``second_segment_id``, ``second_distance_m``, ``snap_ambiguous`` and
        ``snap_status`` (``matched`` / ``ambiguous`` / ``unmatched``).
    """
    if images.crs is None or segments.crs is None:
        raise ValueError("images and segments both need a CRS")
    if segment_id_column not in segments.columns:
        raise ValueError(f"segments have no {segment_id_column!r} column; run prepare_segments() first")
    if max_distance_m <= 0 or ambiguity_margin_m < 0:
        raise ValueError("max_distance_m must be > 0 and ambiguity_margin_m >= 0")

    crs = _metric_crs(segments, metric_crs)
    seg = segments.to_crs(crs)
    pts = images.to_crs(crs)
    seg_ids = seg[segment_id_column].astype(str).to_numpy()
    seg_geoms = seg.geometry.to_numpy()
    pt_geoms = pts.geometry.to_numpy()

    n = len(pts)
    best_id = np.full(n, None, dtype=object)
    best_dist = np.full(n, np.nan)
    best_offset = np.full(n, np.nan)
    second_id = np.full(n, None, dtype=object)
    second_dist = np.full(n, np.nan)

    valid = np.array([g is not None and not g.is_empty for g in pt_geoms], dtype=bool)
    if valid.any() and len(seg):
        pt_idx, seg_idx = seg.sindex.query(pt_geoms[valid], predicate="dwithin", distance=max_distance_m)
        pt_idx = np.flatnonzero(valid)[pt_idx]
        dists = shapely.distance(pt_geoms[pt_idx], seg_geoms[seg_idx])
        candidates = pd.DataFrame(
            {
                "pt": pt_idx,
                "seg": seg_idx,
                "dist": dists,
                # Round to the tie tolerance so floating noise cannot reorder ties.
                "dist_key": np.round(dists / _TIE_TOLERANCE_M).astype(np.int64),
                "sid": seg_ids[seg_idx],
            }
        )
        candidates = candidates[candidates["dist"] <= max_distance_m]
        candidates = candidates.sort_values(["pt", "dist_key", "sid"], kind="mergesort")
        candidates["rank"] = candidates.groupby("pt").cumcount()
        first = candidates[candidates["rank"] == 0]
        runner = candidates[candidates["rank"] == 1]
        best_id[first["pt"].to_numpy()] = first["sid"].to_numpy()
        best_dist[first["pt"].to_numpy()] = first["dist"].to_numpy()
        best_offset[first["pt"].to_numpy()] = shapely.line_locate_point(
            seg_geoms[first["seg"].to_numpy()], pt_geoms[first["pt"].to_numpy()]
        )
        second_id[runner["pt"].to_numpy()] = runner["sid"].to_numpy()
        second_dist[runner["pt"].to_numpy()] = runner["dist"].to_numpy()

    matched = pd.notna(pd.Series(best_id)).to_numpy()
    ambiguous = matched & ~np.isnan(second_dist) & ((second_dist - best_dist) <= ambiguity_margin_m)
    status = np.where(~matched, "unmatched", np.where(ambiguous, "ambiguous", "matched"))

    out = images.copy()
    out[segment_id_column] = best_id
    out["snap_distance_m"] = best_dist
    out["snap_offset_m"] = best_offset
    out["second_segment_id"] = second_id
    out["second_distance_m"] = second_dist
    out["snap_ambiguous"] = ambiguous
    out["snap_status"] = status
    out.attrs["snap_rules"] = {
        "max_distance_m": max_distance_m,
        "ambiguity_margin_m": ambiguity_margin_m,
        "tie_rule": "nearest; ties within 1 mm resolved by the smaller segment id (string order)",
        "metric_crs": str(crs),
    }
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _to_datetime(values: pd.Series) -> pd.Series:
    if values is None:
        return values
    if pd.api.types.is_datetime64_any_dtype(values):
        out = values
    elif pd.api.types.is_numeric_dtype(values):
        out = pd.to_datetime(values, unit="ms", errors="coerce", utc=True)  # Mapillary epoch ms
    else:
        out = pd.to_datetime(values, errors="coerce", utc=True)
    if getattr(out.dt, "tz", None) is None:
        out = out.dt.tz_localize("UTC")
    return out


def _union_length(intervals: List[Tuple[float, float]]) -> Tuple[float, List[Tuple[float, float]]]:
    if not intervals:
        return 0.0, []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return float(sum(e - s for s, e in merged)), [tuple(m) for m in merged]


def _state_summary(values: pd.Series, column: str) -> Dict[str, Any]:
    counts = values.value_counts(dropna=True)
    row = {f"{column}__n_{state}": int(counts.get(state, 0)) for state in STATE_COUNT_VALUES}
    row[f"{column}__n_missing"] = int(values.isna().sum())
    other = values.dropna()[~values.dropna().isin(STATE_COUNT_VALUES)]
    row[f"{column}__n_other"] = int(len(other))
    definite = row[f"{column}__n_present"] + row[f"{column}__n_absent"]
    row[f"{column}__present_share"] = (row[f"{column}__n_present"] / definite) if definite else np.nan
    return row


def _numeric_summary(values: pd.Series, column: str) -> Dict[str, Any]:
    numbers = pd.to_numeric(values, errors="coerce")
    return {
        f"{column}__mean": float(numbers.mean()) if numbers.notna().any() else np.nan,
        f"{column}__n": int(numbers.notna().sum()),
        f"{column}__n_missing": int(numbers.isna().sum()),
    }


def _boolean_summary(values: pd.Series, column: str) -> Dict[str, Any]:
    known = values[values.map(lambda v: isinstance(v, (bool, np.bool_)))]
    return {
        f"{column}__share_true": float(known.astype(bool).mean()) if len(known) else np.nan,
        f"{column}__n": int(len(known)),
        f"{column}__n_missing": int(len(values) - len(known)),
    }


def aggregate_segments(
    snapped: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    numeric_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
    state_columns: Sequence[str] = (),
    support_length_m: float = SEGMENT_DEFAULTS["support_length_m"],
    include_ambiguous: bool = True,
    sequence_column: Optional[str] = "sequence_id",
    time_column: Optional[str] = "captured_at",
    segment_id_column: str = "segment_id",
    metric_crs: Any = None,
) -> gpd.GeoDataFrame:
    """Summarise matched images per street segment.

    Args:
        snapped: Output of :func:`snap_images_to_segments`.
        segments: The same segments the images were matched to.
        numeric_columns: Columns summarised by mean, count and missing count
            (e.g. ``vegetation_pixel_fraction``).
        boolean_columns: Columns summarised as the share of ``True`` among
            real booleans; anything else counts as missing.
        state_columns: Audit state columns (e.g. ``audit_sidewalk_state``).
            Each gets counts of every state and ``present_share`` =
            present / (present + absent); ``not_visible``, ``uncertain``,
            ``not_assessed``, ``invalid``, ``failed`` and missing values never
            enter that denominator.
        support_length_m: Street length one image is taken to observe,
            centred on its snapped position. Coverage is the union of these
            stretches, clipped to the segment.
        include_ambiguous: Use ambiguous matches (default). Set False to keep
            only unambiguous ones; the ambiguous count is reported either way.
        sequence_column: Capture-sequence id column (distinct count), or None.
        time_column: Capture time column (datetime, ISO text, or epoch
            milliseconds as Mapillary returns), or None.
        segment_id_column: Segment id column.
        metric_crs: Projected CRS for lengths (default as in matching).

    Returns:
        One row per segment (all segments, imaged or not) in the segments'
        CRS, with ``segment_length_m``, ``n_images``, ``n_ambiguous_images``,
        ``n_sequences``, ``first_capture``, ``last_capture``,
        ``median_capture``, ``covered_length_m``, ``covered_length_share``,
        a ``covered_geometry`` column, and the requested summaries. The
        analysis unit and rules are in ``attrs["geoai_vlm"]``.
    """
    if support_length_m <= 0:
        raise ValueError("support_length_m must be > 0")
    missing = [c for c in list(numeric_columns) + list(boolean_columns) + list(state_columns) if c not in snapped.columns]
    if missing:
        raise ValueError(f"columns not found in the image table: {missing}")
    if "snap_status" not in snapped.columns:
        raise ValueError("run snap_images_to_segments() first")

    crs = _metric_crs(segments, metric_crs)
    seg_m = segments.to_crs(crs)
    lengths = dict(zip(seg_m[segment_id_column].astype(str), seg_m.geometry.length))
    lines = dict(zip(seg_m[segment_id_column].astype(str), seg_m.geometry))

    usable_status = {"matched", "ambiguous"} if include_ambiguous else {"matched"}
    used = snapped[snapped["snap_status"].isin(usable_status)].copy()
    used[segment_id_column] = used[segment_id_column].astype(str)
    ambiguous_counts = (
        snapped[snapped["snap_status"] == "ambiguous"][segment_id_column].astype(str).value_counts()
    )
    times = _to_datetime(used[time_column]) if time_column and time_column in used.columns else None
    if times is not None:
        used["_capture"] = times
    half = support_length_m / 2.0

    rows = []
    covered_geoms = []
    groups = dict(tuple(used.groupby(segment_id_column))) if len(used) else {}
    for sid in seg_m[segment_id_column].astype(str):
        group = groups.get(sid)
        length = float(lengths[sid])
        row: Dict[str, Any] = {
            segment_id_column: sid,
            "segment_length_m": length,
            "n_images": 0 if group is None else int(len(group)),
            "n_ambiguous_images": int(ambiguous_counts.get(sid, 0)),
        }
        if group is not None and sequence_column and sequence_column in group.columns:
            row["n_sequences"] = int(group[sequence_column].dropna().nunique())
        elif sequence_column:
            row["n_sequences"] = 0 if group is None else np.nan

        if times is not None:
            caps = group["_capture"].dropna() if group is not None else pd.Series(dtype="datetime64[ns, UTC]")
            row["first_capture"] = caps.min() if len(caps) else pd.NaT
            row["last_capture"] = caps.max() if len(caps) else pd.NaT
            row["median_capture"] = caps.median() if len(caps) else pd.NaT

        intervals = []
        if group is not None and length > 0:
            for offset in group["snap_offset_m"].dropna():
                intervals.append((max(0.0, offset - half), min(length, offset + half)))
        covered, merged = _union_length(intervals)
        row["covered_length_m"] = covered
        row["covered_length_share"] = covered / length if length > 0 else np.nan
        pieces = [substring(lines[sid], s, e) for s, e in merged if e > s]
        covered_geoms.append(
            MultiLineString([p for p in pieces if isinstance(p, LineString) and p.length > 0]) if pieces else None
        )

        empty = pd.Series(dtype=object)
        for column in numeric_columns:
            row.update(_numeric_summary(group[column] if group is not None else empty, column))
        for column in boolean_columns:
            row.update(_boolean_summary(group[column] if group is not None else empty, column))
        for column in state_columns:
            row.update(_state_summary(group[column] if group is not None else empty, column))
        rows.append(row)

    table = pd.DataFrame(rows)
    keep = [c for c in segments.columns if c not in table.columns and c != segments.geometry.name]
    base = segments[[segment_id_column] + keep + [segments.geometry.name]].copy()
    base[segment_id_column] = base[segment_id_column].astype(str)
    result = base.merge(table, on=segment_id_column, how="left")
    result = gpd.GeoDataFrame(result, geometry=segments.geometry.name, crs=segments.crs)
    result["covered_geometry"] = gpd.GeoSeries(covered_geoms, crs=crs).to_crs(segments.crs).values

    rules = dict(snapped.attrs.get("snap_rules", {}))
    rules.update(
        {
            "support_length_m": support_length_m,
            "coverage_rule": (
                "each image observes support_length_m of street centred on its snapped "
                "position, clipped to the segment; covered length is the union"
            ),
            "include_ambiguous": include_ambiguous,
            "present_share_rule": "present / (present + absent); other states counted separately",
            "length_crs": str(crs),
        }
    )
    result.attrs["geoai_vlm"] = {
        "analysis_unit": "street_segment",
        "segment_id_column": segment_id_column,
        "rules": rules,
        "measures": {
            "numeric": list(numeric_columns),
            "boolean": list(boolean_columns),
            "state": list(state_columns),
        },
        "n_images_input": int(len(snapped)),
        "n_images_used": int(len(used)),
        "n_images_unmatched": int((snapped["snap_status"] == "unmatched").sum()),
        "notes": [
            "Image counts are not network coverage; see covered_length_share.",
            "Values summarise image observations, not physical measurements of the street.",
        ],
        "geoai_vlm_version": _version(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return result


# ---------------------------------------------------------------------------
# GeoParquet with metadata
# ---------------------------------------------------------------------------
def write_segment_parquet(
    gdf: gpd.GeoDataFrame,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write *gdf* as GeoParquet with the analysis metadata embedded.

    The metadata (by default ``gdf.attrs["geoai_vlm"]``) is stored as JSON
    under the ``geoai_vlm`` key of the Parquet schema metadata, next to the
    standard ``geo`` key, so the analysis unit and rules travel with the file.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata if metadata is not None else gdf.attrs.get("geoai_vlm", {})
    gdf.to_parquet(path, index=False)
    table = pq.read_table(path)
    existing = dict(table.schema.metadata or {})
    existing[METADATA_KEY] = json.dumps(meta, default=str, ensure_ascii=False).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(existing), path)
    return path


def read_segment_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """The ``geoai_vlm`` metadata written by :func:`write_segment_parquet`."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(METADATA_KEY)
    return json.loads(raw.decode("utf-8")) if raw else {}
