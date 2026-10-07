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

        # Signature is plot_elbow_curve(k_values, inertias, ...). The previous
        # version of this test passed them the other way round; because it only
        # asserted the return type, the swapped axes went unnoticed.
        k_values = list(range(2, 8))
        inertias = [100, 80, 50, 40, 38, 37]
        fig = plot_elbow_curve(k_values, inertias, optimal_k=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plots_k_on_x_axis_and_inertia_on_y(self):
        """Checking the figure type alone cannot catch swapped axes."""
        from geoai_vlm.visualization import plot_elbow_curve

        k_values = list(range(2, 8))
        inertias = [100.0, 80.0, 50.0, 40.0, 38.0, 37.0]
        fig = plot_elbow_curve(k_values, inertias)

        ax = fig.axes[0]
        lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == len(k_values)]
        assert lines, "no data line drawn"
        xdata = list(lines[0].get_xdata())
        ydata = list(lines[0].get_ydata())

        assert xdata == [float(k) for k in k_values], (
            f"x axis must carry the cluster counts, got {xdata}"
        )
        assert ydata == inertias, f"y axis must carry the inertias, got {ydata}"
        plt.close(fig)

    def test_save_path_writes_a_file(self, tmp_path):
        from geoai_vlm.visualization import plot_elbow_curve

        out = tmp_path / "nested" / "elbow.png"
        fig = plot_elbow_curve([2, 3, 4], [10.0, 5.0, 4.0], save_path=out)
        assert out.exists() and out.stat().st_size > 0
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
