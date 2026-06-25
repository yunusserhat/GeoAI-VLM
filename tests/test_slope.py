# -*- coding: utf-8 -*-
"""Tests for geoai_vlm.slope module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _sloped_road_semantic_map(height=80, width=120, slope=0.2, intercept=18):
    semantic_map = np.zeros((height, width), dtype=np.uint8)
    for col in range(width):
        edge_row = int(round(slope * col + intercept))
        semantic_map[edge_row:, col] = 13
    return semantic_map


class TestSlopeFromSemanticMap:
    def test_estimates_road_edge_angle(self):
        from geoai_vlm.slope import SlopeConfig, estimate_slope_from_semantic_map

        semantic_map = _sloped_road_semantic_map(slope=0.2, intercept=18)
        result = estimate_slope_from_semantic_map(
            semantic_map,
            config=SlopeConfig(morphology_kernel_size=0, min_edge_points=20),
        )

        assert result.status == "success"
        assert result.road_area > 0
        assert result.edge_point_count == semantic_map.shape[1]
        assert result.road_edge_line_slope == pytest.approx(0.2, abs=0.02)
        assert result.road_edge_line_angle == pytest.approx(np.degrees(np.arctan(0.2)), abs=1)

    def test_no_road_returns_invalid_result(self):
        from geoai_vlm.slope import SlopeConfig, estimate_slope_from_semantic_map

        result = estimate_slope_from_semantic_map(
            np.zeros((20, 20), dtype=np.uint8),
            config=SlopeConfig(morphology_kernel_size=0, min_edge_points=5),
        )

        assert result.status == "no_road"
        assert result.road_area == 0
        assert result.road_edge_line_angle == -999.0

    def test_create_road_mask_combines_mapillary_road_labels(self):
        from geoai_vlm.slope import create_road_mask

        semantic_map = np.array(
            [
                [0, 13, 23],
                [24, 17, 30],
            ],
            dtype=np.uint8,
        )
        mask = create_road_mask(semantic_map, morphology_kernel_size=0)

        assert mask.tolist() == [[0, 1, 1], [1, 0, 0]]


class TestSignedSlope:
    def test_right_view_keeps_heading_sign(self):
        from geoai_vlm.slope import compute_signed_slope

        assert compute_signed_slope(5.0, perspective_angle=90, heading=10, edge_bearing=20) == 5.0
        assert compute_signed_slope(-5.0, perspective_angle=90, heading=10, edge_bearing=20) == -5.0

    def test_left_view_inverts_heading_sign(self):
        from geoai_vlm.slope import compute_signed_slope

        assert compute_signed_slope(5.0, perspective_angle=270, heading=10, edge_bearing=20) == -5.0
        assert compute_signed_slope(-5.0, perspective_angle=270, heading=10, edge_bearing=20) == 5.0

    def test_opposite_edge_bearing_flips_sign(self):
        from geoai_vlm.slope import compute_signed_slope

        assert compute_signed_slope(5.0, perspective_angle=90, heading=10, edge_bearing=210) == -5.0


class TestPanoSlopeAggregation:
    def test_aggregates_absolute_slope_by_pano(self):
        from geoai_vlm.slope import aggregate_pano_slopes

        df = pd.DataFrame(
            {
                "filename": [
                    "abc_Direction_90_FOV_90_aspect_10--10_raw.png",
                    "abc_Direction_270_FOV_90_aspect_10--10_raw.png",
                    "def_Direction_90_FOV_90_aspect_10--10_raw.png",
                ],
                "road_edge_line_angle": [4.0, -8.0, 20.0],
                "road_area": [1, 3, 1],
            }
        )

        result = aggregate_pano_slopes(df, angle_threshold=10)

        assert result.loc[0, "road_estimated_slope"] == pytest.approx(7.0)
        assert result.loc[1, "road_estimated_slope"] == pytest.approx(7.0)
        assert np.isnan(result.loc[2, "road_estimated_slope"])

    def test_aggregates_signed_slope_when_metadata_exists(self):
        from geoai_vlm.slope import aggregate_pano_slopes

        df = pd.DataFrame(
            {
                "filename": [
                    "abc_Direction_90_FOV_90_aspect_10--10_raw.png",
                    "abc_Direction_270_FOV_90_aspect_10--10_raw.png",
                ],
                "road_edge_line_angle": [4.0, 8.0],
                "road_area": [1, 1],
                "heading": [10.0, 10.0],
                "edge_bearing": [10.0, 10.0],
            }
        )

        result = aggregate_pano_slopes(df)

        assert result.loc[0, "signed_slope"] == pytest.approx(4.0)
        assert result.loc[1, "signed_slope"] == pytest.approx(-8.0)
        assert result.loc[0, "road_estimated_slope"] == pytest.approx(-2.0)
        assert result.loc[0, "road_estimated_slope_abs"] == pytest.approx(2.0)


class TestFilenameParsing:
    def test_extracts_pano_id_and_perspective_angle(self):
        from geoai_vlm.slope import extract_pano_id, extract_perspective_angle

        filename = "abc_Direction_270_FOV_90_aspect_10--10_raw.png"
        assert extract_pano_id(filename) == "abc"
        assert extract_perspective_angle(filename) == 270.0
