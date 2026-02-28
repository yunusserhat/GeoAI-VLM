# -*- coding: utf-8 -*-
"""Tests for geoai_vlm.clustering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import MOCK_EMBED_DIM


class TestSemanticClusterer:
    """Verify SemanticClusterer clustering and analysis methods."""

    @pytest.fixture
    def clusterer(self, mock_embedder):
        from geoai_vlm.clustering import ClusterConfig, SemanticClusterer

        config = ClusterConfig(
            n_clusters=3,
            embedding_columns=["scene_narrative"],
        )
        return SemanticClusterer(embedder=mock_embedder, config=config)

    # ---- cluster ----
    def test_cluster_adds_column(self, clusterer, sample_gdf):
        gdf = clusterer.cluster(sample_gdf)
        assert "cluster" in gdf.columns
        assert gdf["cluster"].nunique() <= 3

    def test_cluster_preserves_rows(self, clusterer, sample_gdf):
        gdf = clusterer.cluster(sample_gdf)
        assert len(gdf) == len(sample_gdf)

    # ---- find_optimal_k ----
    def test_find_optimal_k_returns_tuple(self, clusterer, sample_gdf):
        k_values, inertias = clusterer.find_optimal_k(
            sample_gdf,
            k_range=(2, 6),
        )
        assert isinstance(k_values, list)
        assert len(k_values) == 4
        assert len(inertias) == 4

    # ---- extract_keywords ----
    def test_extract_keywords(self, clusterer, sample_gdf):
        gdf = clusterer.cluster(sample_gdf)
        # build embedding_text column for TF-IDF
        gdf["embedding_text"] = clusterer.build_embedding_text(gdf)
        keywords = clusterer.extract_keywords(gdf)
        assert isinstance(keywords, dict)
        assert len(keywords) == gdf["cluster"].nunique()

    # ---- analyze_categories ----
    def test_analyze_categories(self, clusterer, sample_gdf):
        gdf = clusterer.cluster(sample_gdf)
        profiles = clusterer.analyze_categories(
            gdf,
            category_columns=["land_use_primary"],
        )
        assert isinstance(profiles, dict)
        assert len(profiles) == gdf["cluster"].nunique()


class TestClusterConfig:
    """Verify ClusterConfig defaults and validation."""

    def test_defaults(self):
        from geoai_vlm.clustering import ClusterConfig

        config = ClusterConfig()
        assert config.n_clusters == 10
        assert config.random_state == 42
        assert config.n_keywords == 10

    def test_custom_values(self):
        from geoai_vlm.clustering import ClusterConfig

        config = ClusterConfig(n_clusters=10, random_state=0, n_keywords=20)
        assert config.n_clusters == 10
        assert config.random_state == 0
        assert config.n_keywords == 20
