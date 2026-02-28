# -*- coding: utf-8 -*-
"""
Visualization Module for GeoAI-VLM
====================================
Plotting utilities for semantic clustering, elbow curves, LISA maps,
category distributions and markdown report generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd


__all__ = [
    "plot_elbow_curve",
    "plot_cluster_map",
    "plot_lisa_map",
    "plot_category_distribution",
    "generate_report",
]


# ---------------------------------------------------------------------------
# Elbow curve
# ---------------------------------------------------------------------------
def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    optimal_k: Optional[int] = None,
    title: str = "Elbow Method for Optimal k",
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot an elbow curve for K-Means cluster selection.

    Args:
        k_values: Evaluated *k* values.
        inertias: Corresponding inertia (WCSS) values.
        optimal_k: If given, a vertical line highlights the chosen *k*.
        title: Plot title.
        figsize: Matplotlib figure size.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_values, inertias, "bo-", linewidth=2, markersize=8)
    if optimal_k is not None:
        ax.axvline(x=optimal_k, color="red", linestyle="--", label=f"k = {optimal_k}")
        ax.legend()
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (Within-cluster sum of squares)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster scatter map
# ---------------------------------------------------------------------------
def plot_cluster_map(
    gdf: gpd.GeoDataFrame,
    cluster_column: str = "cluster",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    cmap: Optional[str] = None,
    markersize: float = 3,
    alpha: float = 0.7,
):
    """
    Scatter-plot the spatial distribution of clusters.

    Args:
        gdf: GeoDataFrame with a cluster column and point geometry.
        cluster_column: Column containing cluster labels.
        title: Plot title (auto-generated when ``None``).
        figsize: Matplotlib figure size.
        cmap: Colormap name (auto-selected when ``None``).
        markersize: Marker size for points.
        alpha: Marker transparency.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    n_clusters = gdf[cluster_column].nunique()
    used_cmap = cmap or ("tab20" if n_clusters > 10 else "tab10")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(
        column=cluster_column,
        categorical=True,
        legend=True,
        markersize=markersize,
        ax=ax,
        cmap=used_cmap,
        alpha=alpha,
        legend_kwds={"title": "Semantic Cluster", "loc": "lower right"},
    )

    ax.set_title(title or "Spatial Distribution of Semantic Clusters", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LISA cluster map
# ---------------------------------------------------------------------------
_LISA_COLORS = {
    1: "#d7191c",  # High-High (hotspot)
    2: "#2c7bb6",  # Low-Low (coldspot)
    3: "#abd9e9",  # Low-High
    4: "#fdae61",  # High-Low
    0: "#ffffbf",  # Not significant
}

_LISA_LABELS = {
    1: "High-High (Hotspot)",
    2: "Low-Low (Coldspot)",
    3: "Low-High",
    4: "High-Low",
    0: "Not Significant",
}


def plot_lisa_map(
    gdf: gpd.GeoDataFrame,
    lisa_column: str = "lisa_cluster",
    significance_column: str = "lisa_significant",
    p_threshold: float = 0.05,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    markersize: float = 3,
    alpha: float = 0.7,
):
    """
    Plot a LISA (Local Moran's I) cluster classification map.

    Requires the GeoDataFrame to have been processed by
    :meth:`~geoai_vlm.spatial.SpatialAnalyzer.moran_local`.

    Args:
        gdf: GeoDataFrame with LISA columns.
        lisa_column: Column with LISA quadrant codes.
        significance_column: Column with boolean significance flag.
        p_threshold: (unused – for API compatibility).
        title: Plot title.
        figsize: Matplotlib figure size.
        markersize: Marker size.
        alpha: Marker transparency.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)

    for lc, color in _LISA_COLORS.items():
        subset = gdf[gdf[lisa_column] == lc]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, markersize=markersize, alpha=alpha)

    legend_elements = [
        Patch(facecolor=c, label=_LISA_LABELS[lc]) for lc, c in _LISA_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_title(title or "LISA Cluster Map", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Category distribution (stacked bar)
# ---------------------------------------------------------------------------
def plot_category_distribution(
    gdf: gpd.GeoDataFrame,
    category_columns: Optional[List[str]] = None,
    cluster_column: str = "cluster",
    figsize: Tuple[int, int] = (18, 6),
    cmap: str = "tab20",
):
    """
    Stacked bar charts showing category distributions per cluster.

    Args:
        gdf: GeoDataFrame with cluster column and category columns.
        category_columns: Category columns to plot (default:
            ``land_use_primary``, ``street_type``, ``place_character``).
        cluster_column: Column containing cluster labels.
        figsize: Matplotlib figure size.
        cmap: Colormap for the bars.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    cats = category_columns or ["land_use_primary", "street_type", "place_character"]
    # Only keep cols that actually exist
    cats = [c for c in cats if c in gdf.columns]
    if not cats:
        raise ValueError("No category columns found in the GeoDataFrame.")

    fig, axes = plt.subplots(1, len(cats), figsize=figsize)
    if len(cats) == 1:
        axes = [axes]

    for idx, col in enumerate(cats):
        ax = axes[idx]
        ct = pd.crosstab(gdf[cluster_column], gdf[col], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, colormap=cmap)
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_title(f'{col.replace("_", " ").title()} by Cluster', fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Markdown findings report
# ---------------------------------------------------------------------------
def generate_report(
    gdf: gpd.GeoDataFrame,
    keywords: Dict[int, List[str]],
    moran_results: Optional[Dict] = None,
    category_profiles: Optional[Dict[int, Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    n_clusters: Optional[int] = None,
) -> str:
    """
    Generate a Markdown findings report.

    Args:
        gdf: Clustered GeoDataFrame (must have a ``cluster`` column).
        keywords: ``{cluster_id: [keyword, …]}`` from
            :meth:`~geoai_vlm.clustering.SemanticClusterer.extract_keywords`.
        moran_results: Global Moran's I results (dict of
            :class:`~geoai_vlm.spatial.MoranResult` or plain dicts).
        category_profiles: Optional cluster profile dicts from
            :meth:`~geoai_vlm.clustering.SemanticClusterer.analyze_categories`.
        output_path: If given, the report is written to this file.
        n_clusters: Number of clusters (inferred from *gdf* when ``None``).

    Returns:
        The full report as a Markdown string.
    """
    import datetime

    k = n_clusters or gdf["cluster"].nunique()
    lines: List[str] = []

    lines.append("# Semantic Clustering Analysis Report\n")
    lines.append(f"**Analysis Date:** {datetime.date.today().isoformat()}\n")
    lines.append(f"**Total Images Analysed:** {len(gdf):,}\n\n")

    # ---- Cluster summary table -----------------------------------------
    lines.append("## 1. Semantic Cluster Identification\n")
    lines.append(f"K-Means clustering with k={k} identified the following groups:\n\n")

    header = "| Cluster | Images | % | Top Keywords |"
    if category_profiles:
        header += " Dominant Land Use | Dominant Street Type |"
    lines.append(header + "\n")

    sep = "|---------|--------|---|--------------|"
    if category_profiles:
        sep += "-------------------|----------------------|"
    lines.append(sep + "\n")

    counts = gdf["cluster"].value_counts().sort_index()
    for cid in sorted(keywords.keys()):
        cnt = counts.get(cid, 0)
        pct = cnt / len(gdf) * 100
        kws = ", ".join(keywords[cid][:4])
        row = f"| {cid} | {cnt:,} | {pct:.1f}% | {kws} |"
        if category_profiles:
            prof = category_profiles.get(cid, {})
            lu = prof.get("land_use_primary_top", "N/A")
            st = prof.get("street_type_top", "N/A")
            row += f" {lu} | {st} |"
        lines.append(row + "\n")

    lines.append("\n")

    # ---- Spatial autocorrelation ----------------------------------------
    lines.append("## 2. Spatial Autocorrelation Analysis\n")

    if moran_results:
        lines.append("### Global Moran's I Statistics\n")
        lines.append("| Cluster | Moran's I | Expected I | p-value | Sig. |\n")
        lines.append("|---------|-----------|------------|---------|------|\n")

        for cid, mr in sorted(moran_results.items(), key=lambda x: x[0]):
            # Support both MoranResult dataclass and plain dict
            I = mr.I if hasattr(mr, "I") else mr["I"]
            eI = mr.expected_I if hasattr(mr, "expected_I") else mr["expected_I"]
            p = mr.p_value if hasattr(mr, "p_value") else mr["p_value"]

            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            lines.append(f"| {cid} | {I:.4f} | {eI:.4f} | {p:.4f} | {sig} |\n")

        lines.append(
            "\n*Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, * p<0.05, "
            "ns = not significant*\n\n"
        )

        # Interpretation
        all_I = [
            (mr.I if hasattr(mr, "I") else mr["I"])
            for mr in moran_results.values()
        ]
        all_p = [
            (mr.p_value if hasattr(mr, "p_value") else mr["p_value"])
            for mr in moran_results.values()
        ]
        avg_I = float(np.mean(all_I))
        sig_n = sum(1 for p in all_p if p < 0.05)

        lines.append("### Interpretation\n")
        lines.append(
            f"- **{sig_n} out of {len(moran_results)} clusters** show "
            f"statistically significant spatial clustering (p < 0.05)\n"
        )
        lines.append(f"- **Average Moran's I = {avg_I:.4f}**\n\n")
    else:
        lines.append("*Spatial analysis not performed.*\n\n")

    report_text = "".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        print(f"Saved report to {output_path}")

    return report_text
