# -*- coding: utf-8 -*-
"""
Coverage and recency of street-level imagery
============================================
How much of a street network is actually observed, and how recently --
reported by segment group (e.g. road class), by user-supplied areas (e.g.
neighbourhoods) and by capture-year range.

Inputs are the per-segment table from
:func:`geoai_vlm.segments.aggregate_segments`. Two quantities are kept apart
in every table, because they answer different questions:

* **network coverage** -- the share of street *length* within the support
  distance of at least one matched image (``covered_length_share``);
* **image counts** -- how many photographs there are (``n_images``,
  ``images_per_km``).

A thousand images of one square cover one square. A high image count says
nothing about coverage, and a coverage share says nothing about how many
viewpoints back it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd


__all__ = [
    "COVERAGE_NOTE",
    "network_coverage",
    "coverage_by_area",
    "recency_report",
]

COVERAGE_NOTE = (
    "covered_length_share is the share of street length within support_length_m/2 "
    "of a matched image; n_images counts photographs. They measure different things "
    "and neither implies the other."
)


def _check(summary: pd.DataFrame) -> None:
    needed = {"segment_length_m", "covered_length_m", "n_images"}
    missing = needed - set(summary.columns)
    if missing:
        raise ValueError(
            f"not a segment summary (missing {sorted(missing)}); use aggregate_segments() output"
        )


def _summarise(frame: pd.DataFrame) -> pd.Series:
    length = float(frame["segment_length_m"].sum())
    covered = float(frame["covered_length_m"].sum())
    with_images = int((frame["n_images"] > 0).sum())
    n_images = int(frame["n_images"].sum())
    return pd.Series(
        {
            "n_segments": int(len(frame)),
            "network_length_m": length,
            "covered_length_m": covered,
            "covered_length_share": covered / length if length > 0 else np.nan,
            "n_segments_with_images": with_images,
            "segments_with_images_share": with_images / len(frame) if len(frame) else np.nan,
            "n_images": n_images,
            "images_per_km": n_images / (length / 1000.0) if length > 0 else np.nan,
        }
    )


def network_coverage(
    summary: pd.DataFrame,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Network coverage overall or per group (e.g. ``group_column="road_class"``).

    Returns:
        One row per group (or one ``"all"`` row) with ``n_segments``,
        ``network_length_m``, ``covered_length_m``, ``covered_length_share``,
        ``n_segments_with_images``, ``segments_with_images_share``,
        ``n_images`` and ``images_per_km``. Missing group values are reported
        as ``"(unknown)"`` rather than dropped.
    """
    _check(summary)
    frame = pd.DataFrame(summary.drop(columns=[c for c in ("geometry", "covered_geometry") if c in summary.columns]))
    if group_column is None:
        out = _summarise(frame).to_frame().T
        out.insert(0, "group", "all")
    else:
        if group_column not in frame.columns:
            raise ValueError(f"no column {group_column!r}")
        keys = frame[group_column].astype(object).where(frame[group_column].notna(), "(unknown)")
        out = frame.groupby(keys, sort=True).apply(_summarise, include_groups=False).reset_index()
        out = out.rename(columns={out.columns[0]: group_column})
    out.attrs["note"] = COVERAGE_NOTE
    return out.reset_index(drop=True)


def coverage_by_area(
    summary: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    area_id_column: str = "area_id",
    images: Optional[gpd.GeoDataFrame] = None,
    metric_crs: Any = None,
) -> pd.DataFrame:
    """Network coverage within user-supplied areas (neighbourhoods, districts...).

    Street length and covered length are clipped to each area, so a segment
    crossing a boundary is split between areas by length. Images are counted
    by their own location when *images* is given (otherwise the image count
    is not reported per area, rather than guessed from segments).

    Args:
        summary: :func:`~geoai_vlm.segments.aggregate_segments` output (needs
            its ``covered_geometry`` column).
        areas: Polygons with an ``area_id_column``.
        area_id_column: Area id column.
        images: Optional image points, for per-area image counts.
        metric_crs: Projected CRS for lengths (default: local UTM zone).
    """
    _check(summary)
    if "covered_geometry" not in summary.columns:
        raise ValueError("summary has no covered_geometry column")
    if area_id_column not in areas.columns:
        raise ValueError(f"areas have no {area_id_column!r} column")
    if areas.crs is None or summary.crs is None:
        raise ValueError("summary and areas need a CRS")

    crs = metric_crs or (summary.crs if summary.crs.is_projected else summary.estimate_utm_crs())
    streets = summary.to_crs(crs)
    # to_crs() only reprojects the active geometry, so covered_geometry may
    # still be in the CRS it was built in: use its own CRS when it has one.
    covered_values = summary["covered_geometry"]
    covered_crs = getattr(covered_values, "crs", None) or summary.crs
    covered = gpd.GeoSeries(covered_values.values, crs=covered_crs).to_crs(crs)
    zones = areas.to_crs(crs)

    rows = []
    for _, zone in zones.iterrows():
        polygon = zone.geometry
        network = float(streets.geometry.intersection(polygon).length.sum())
        observed = float(covered.intersection(polygon).length.sum())
        row = {
            area_id_column: zone[area_id_column],
            "network_length_m": network,
            "covered_length_m": observed,
            "covered_length_share": observed / network if network > 0 else np.nan,
        }
        if images is not None:
            points = images.to_crs(crs)
            row["n_images"] = int(points.within(polygon).sum())
            row["images_per_km"] = row["n_images"] / (network / 1000.0) if network > 0 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["note"] = COVERAGE_NOTE
    return out


def _bucket_labels(edges: Sequence[int]) -> list:
    labels = [f"<= {edges[0]}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo + 1}-{hi}")
    labels.append(f">= {edges[-1] + 1}")
    return labels


def recency_report(
    summary: pd.DataFrame,
    year_edges: Sequence[int] = (2015, 2018, 2021),
    time_column: str = "last_capture",
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Share of network length by the year of its most recent image.

    Segments are bucketed by the capture year in *time_column* (by default the
    newest image on the segment) using inclusive upper year edges, plus an
    ``"unknown capture time"`` bucket for segments whose images have no
    readable capture time and a ``"no imagery"`` bucket for segments without
    images. Shares are of network length, so they add up to one within each
    group.

    Args:
        summary: :func:`~geoai_vlm.segments.aggregate_segments` output.
        year_edges: Increasing years closing each bucket: ``(2015, 2018, 2021)``
            gives ``<= 2015``, ``2016-2018``, ``2019-2021``, ``>= 2022``.
        time_column: Capture-time column of the summary.
        group_column: Optional grouping (e.g. ``"road_class"``).

    Returns:
        Long table with ``recency``, ``network_length_m``, ``length_share``
        and ``n_segments`` (per group when *group_column* is given).
    """
    _check(summary)
    edges = list(year_edges)
    if edges != sorted(set(edges)) or not edges:
        raise ValueError("year_edges must be strictly increasing years")
    if time_column not in summary.columns:
        raise ValueError(f"no column {time_column!r}")

    frame = pd.DataFrame(summary.drop(columns=[c for c in ("geometry", "covered_geometry") if c in summary.columns]))
    years = pd.to_datetime(frame[time_column], errors="coerce", utc=True).dt.year
    labels = _bucket_labels(edges)
    bins = [-np.inf] + [e + 0.5 for e in edges] + [np.inf]
    bucket = pd.cut(years, bins=bins, labels=labels, right=False).astype(object)
    bucket = bucket.where(years.notna(), "unknown capture time")
    bucket = bucket.where(frame["n_images"] > 0, "no imagery")
    frame["recency"] = pd.Categorical(
        bucket, categories=labels + ["unknown capture time", "no imagery"], ordered=True
    )

    keys = ["recency"] if group_column is None else [group_column, "recency"]
    grouped = (
        frame.groupby(keys, observed=False)
        .agg(network_length_m=("segment_length_m", "sum"), n_segments=("segment_length_m", "size"))
        .reset_index()
    )
    totals = grouped.groupby(group_column)["network_length_m"].transform("sum") if group_column else grouped["network_length_m"].sum()
    grouped["length_share"] = grouped["network_length_m"] / totals
    grouped.attrs["note"] = (
        "Recency is the capture year of the newest image on each segment; "
        "'unknown capture time' segments have images without a readable capture "
        "time; 'no imagery' segments have none. Shares are of network length."
    )
    return grouped
