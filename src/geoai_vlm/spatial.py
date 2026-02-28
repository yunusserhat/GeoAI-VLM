# -*- coding: utf-8 -*-
"""
Spatial Analysis Module for GeoAI-VLM
=======================================
Global and Local Moran's I spatial autocorrelation analysis for
semantically clustered street-view data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd


__all__ = [
    "SpatialAnalyzer",
    "MoranResult",
]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------
try:
    from libpysal.weights import KNN as _KNN        # noqa: F401
    from esda.moran import Moran as _Moran          # noqa: F401
    from esda.moran import Moran_Local as _MoranLocal  # noqa: F401
    _SPATIAL_LIBS_AVAILABLE = True
except ImportError:
    _SPATIAL_LIBS_AVAILABLE = False


def _require_spatial_libs() -> None:
    """Raise an informative error if spatial libraries are missing."""
    if not _SPATIAL_LIBS_AVAILABLE:
        raise ImportError(
            "Spatial autocorrelation analysis requires 'libpysal' and 'esda'.  "
            "Install them with:  pip install libpysal esda"
        )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class MoranResult:
    """
    Container for Global Moran's I results for one cluster (or variable).

    Attributes:
        I: Moran's I statistic.
        expected_I: Expected I under the null hypothesis.
        p_value: Pseudo p-value from permutation inference.
        z_score: Standardised z-score.
    """

    I: float
    expected_I: float
    p_value: float
    z_score: float


# ---------------------------------------------------------------------------
# Analyser class
# ---------------------------------------------------------------------------
class SpatialAnalyzer:
    """
    Spatial autocorrelation analyser for clustered GeoDataFrames.

    Computes **Global Moran's I** and **Local Moran's I (LISA)** using
    K-nearest-neighbour spatial weights.

    Args:
        k_neighbors: Number of neighbours for the KNN weight matrix.

    Example:
        >>> from geoai_vlm import SpatialAnalyzer
        >>> sa = SpatialAnalyzer(k_neighbors=8)
        >>> results = sa.moran_global(gdf, column="cluster")
        >>> print(results)
    """

    def __init__(self, k_neighbors: int = 8):
        self.k_neighbors = k_neighbors

    # ------------------------------------------------------------------
    # Spatial weights
    # ------------------------------------------------------------------
    def spatial_weights(self, gdf: gpd.GeoDataFrame, k: Optional[int] = None):
        """
        Build a row-standardised KNN spatial weight matrix.

        Args:
            gdf: GeoDataFrame with point geometry.
            k: Override ``self.k_neighbors``.

        Returns:
            A ``libpysal.weights.KNN`` weight matrix.
        """
        _require_spatial_libs()
        from libpysal.weights import KNN

        k_val = k or self.k_neighbors
        w = KNN.from_dataframe(gdf, k=k_val)
        w.transform = "r"  # row-standardisation
        return w

    # ------------------------------------------------------------------
    # Global Moran's I
    # ------------------------------------------------------------------
    def moran_global(
        self,
        gdf: gpd.GeoDataFrame,
        column: str = "cluster",
    ) -> Dict[int, MoranResult]:
        """
        Compute Global Moran's I for each cluster as a binary indicator.

        For a categorical *column* the statistic is computed once per unique
        value by encoding the column as ``1`` (in cluster) vs ``0`` (not in
        cluster).

        Args:
            gdf: GeoDataFrame with the cluster column and point geometry.
            column: Name of the cluster column.

        Returns:
            ``{cluster_id: MoranResult}`` mapping.
        """
        _require_spatial_libs()
        from esda.moran import Moran

        w = self.spatial_weights(gdf)
        results: Dict[int, MoranResult] = {}

        for cid in sorted(gdf[column].unique()):
            binary = (gdf[column] == cid).astype(int)
            moran = Moran(binary, w)
            results[cid] = MoranResult(
                I=float(moran.I),
                expected_I=float(moran.EI),
                p_value=float(moran.p_sim),
                z_score=float(moran.z_sim),
            )
            print(
                f"  Cluster {cid}: Moran's I = {moran.I:.4f}, "
                f"p-value = {moran.p_sim:.4f}"
            )

        return results

    # ------------------------------------------------------------------
    # Local Moran's I (LISA)
    # ------------------------------------------------------------------
    def moran_local(
        self,
        gdf: gpd.GeoDataFrame,
        column: str = "cluster",
        p_threshold: float = 0.05,
    ) -> gpd.GeoDataFrame:
        """
        Compute Local Moran's I (LISA) and add classification columns.

        New columns added to the returned GeoDataFrame:

        * ``lisa_cluster``: LISA quadrant classification per cluster
          (1 = HH, 2 = LL, 3 = LH, 4 = HL, 0 = not significant).
        * ``lisa_p_value``: Local pseudo p-value.
        * ``lisa_significant``: Boolean flag (``True`` when *p < p_threshold*).

        Since *column* is categorical, the LISA statistics are computed for
        the **majority cluster** at each observation.

        Args:
            gdf: GeoDataFrame with the cluster column and point geometry.
            column: Name of the cluster column.
            p_threshold: Significance threshold.

        Returns:
            Copy of *gdf* with LISA columns.
        """
        _require_spatial_libs()
        from esda.moran import Moran_Local

        w = self.spatial_weights(gdf)
        gdf = gdf.copy()

        # Accumulate LISA results across all clusters and pick the result
        # from the cluster that each observation belongs to.
        n = len(gdf)
        lisa_class = np.zeros(n, dtype=int)
        lisa_pval = np.ones(n, dtype=float)

        for cid in sorted(gdf[column].unique()):
            binary = (gdf[column] == cid).astype(int)
            lisa = Moran_Local(binary, w)

            mask = gdf[column].values == cid
            sig = lisa.p_sim < p_threshold
            q = lisa.q

            # Assign quadrant for this cluster's own members
            for obs_idx in np.where(mask)[0]:
                if sig[obs_idx]:
                    lisa_class[obs_idx] = int(q[obs_idx])
                lisa_pval[obs_idx] = float(lisa.p_sim[obs_idx])

        gdf["lisa_cluster"] = lisa_class
        gdf["lisa_p_value"] = lisa_pval
        gdf["lisa_significant"] = lisa_pval < p_threshold

        return gdf
