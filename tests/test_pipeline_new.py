# -*- coding: utf-8 -*-
"""Integration tests for new pipeline convenience functions."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from tests.conftest import MOCK_EMBED_DIM, MockImageEmbedder


class TestClusterDescriptions:
    """Test the cluster_descriptions pipeline helper."""

    def test_adds_cluster_column(self, sample_gdf):
        from geoai_vlm.clustering import ClusterConfig, SemanticClusterer

        mock = MockImageEmbedder()
        config = ClusterConfig(
            n_clusters=3,
            embedding_columns=["scene_narrative"],
        )
        clusterer = SemanticClusterer(embedder=mock, config=config)
        gdf = clusterer.cluster(sample_gdf, n_clusters=3)
        assert "cluster" in gdf.columns
        assert gdf["cluster"].nunique() <= 3

    def test_returns_geodataframe(self, sample_gdf):
        import geopandas as gpd
        from geoai_vlm.clustering import ClusterConfig, SemanticClusterer

        mock = MockImageEmbedder()
        config = ClusterConfig(
            n_clusters=2,
            embedding_columns=["scene_narrative"],
        )
        clusterer = SemanticClusterer(embedder=mock, config=config)
        gdf = clusterer.cluster(sample_gdf, n_clusters=2)
        assert isinstance(gdf, gpd.GeoDataFrame)


class TestAnalyzeSpatial:
    """Test the analyze_spatial pipeline helper."""

    def test_returns_dict(self, sample_gdf):
        from geoai_vlm.pipeline import analyze_spatial

        rng = np.random.RandomState(42)
        sample_gdf = sample_gdf.copy()
        sample_gdf["cluster"] = rng.randint(0, 3, size=len(sample_gdf))

        results = analyze_spatial(sample_gdf, k_neighbors=5)
        assert isinstance(results, dict)
        assert "global" in results
        assert "gdf" in results
        assert len(results) > 0
