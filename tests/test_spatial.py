# -*- coding: utf-8 -*-
"""Tests for geoai_vlm.spatial module."""

from __future__ import annotations

import numpy as np
import pytest


class TestSpatialAnalyzer:
    """Verify MoranResult and global / local Moran's I calculations."""

    @pytest.fixture
    def analyzer(self):
        from geoai_vlm.spatial import SpatialAnalyzer

        return SpatialAnalyzer(k_neighbors=5)

    # ---- Moran's I global ----
    def test_moran_global_returns_dict(self, analyzer, sample_gdf):
        # Add a cluster column
        rng = np.random.RandomState(42)
        sample_gdf = sample_gdf.copy()
        sample_gdf["cluster"] = rng.randint(0, 3, size=len(sample_gdf))

        results = analyzer.moran_global(sample_gdf)
        assert isinstance(results, dict)
        assert len(results) > 0

        # Each value should be a MoranResult
        from geoai_vlm.spatial import MoranResult

        for key, val in results.items():
            assert isinstance(val, MoranResult)
            assert hasattr(val, "I")
            assert hasattr(val, "p_value")

    # ---- Moran's I local (LISA) ----
    def test_moran_local_adds_columns(self, analyzer, sample_gdf):
        rng = np.random.RandomState(42)
        sample_gdf = sample_gdf.copy()
        sample_gdf["cluster"] = rng.randint(0, 3, size=len(sample_gdf))

        result_gdf = analyzer.moran_local(sample_gdf)
        assert "lisa_cluster" in result_gdf.columns
        assert "lisa_p_value" in result_gdf.columns
        assert "lisa_significant" in result_gdf.columns

    def test_moran_local_preserves_rows(self, analyzer, sample_gdf):
        rng = np.random.RandomState(42)
        sample_gdf = sample_gdf.copy()
        sample_gdf["cluster"] = rng.randint(0, 3, size=len(sample_gdf))

        result_gdf = analyzer.moran_local(sample_gdf)
        assert len(result_gdf) == len(sample_gdf)


class TestMoranResult:
    def test_dataclass_fields(self):
        from geoai_vlm.spatial import MoranResult

        mr = MoranResult(I=0.25, expected_I=-0.05, p_value=0.01, z_score=2.5)
        assert mr.I == 0.25
        assert mr.expected_I == -0.05
        assert mr.p_value == 0.01
        assert mr.z_score == 2.5
