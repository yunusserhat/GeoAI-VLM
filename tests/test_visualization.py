# -*- coding: utf-8 -*-
"""Tests for geoai_vlm.visualization module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt


class TestPlotElbowCurve:
    def test_returns_figure(self):
        from geoai_vlm.visualization import plot_elbow_curve

        inertias = [100, 80, 50, 40, 38, 37]
        k_range = range(2, 8)
        fig = plot_elbow_curve(inertias, k_range, optimal_k=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotClusterMap:
    def test_returns_figure(self, sample_gdf):
        from geoai_vlm.visualization import plot_cluster_map

        rng = np.random.RandomState(0)
        sample_gdf = sample_gdf.copy()
        sample_gdf["cluster"] = rng.randint(0, 3, size=len(sample_gdf))

        fig = plot_cluster_map(sample_gdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLisaMap:
    def test_returns_figure(self, sample_gdf):
        from geoai_vlm.visualization import plot_lisa_map

        rng = np.random.RandomState(0)
        sample_gdf = sample_gdf.copy()
        sample_gdf["lisa_cluster"] = rng.choice([0, 1, 2, 3, 4], size=len(sample_gdf))
        sample_gdf["lisa_significant"] = rng.choice([True, False], size=len(sample_gdf))

        fig = plot_lisa_map(sample_gdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCategoryDistribution:
    def test_returns_figure(self, sample_gdf):
        from geoai_vlm.visualization import plot_category_distribution

        rng = np.random.RandomState(0)
        sample_gdf = sample_gdf.copy()
        sample_gdf["cluster"] = rng.randint(0, 3, size=len(sample_gdf))

        fig = plot_category_distribution(
            sample_gdf,
            category_columns=["land_use_primary"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestGenerateReport:
    def test_returns_string(self, sample_gdf):
        from geoai_vlm.visualization import generate_report

        rng = np.random.RandomState(0)
        sample_gdf = sample_gdf.copy()
        sample_gdf["cluster"] = rng.randint(0, 3, size=len(sample_gdf))

        keywords = {0: ["word1", "word2"], 1: ["word3"], 2: ["word4", "word5"]}
        report = generate_report(sample_gdf, keywords)
        assert isinstance(report, str)
        assert "Cluster" in report
