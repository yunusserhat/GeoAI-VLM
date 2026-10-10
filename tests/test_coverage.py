# -*- coding: utf-8 -*-
"""
Coverage and recency reports on the synthetic network of test_segments (P1-E).

Hand-computed expectations (metres; see tests/test_segments.py for the layout):

    network length 350 m = A 100 + B 100 + C 50 + D 100
    covered        113 m = A 47.5 + B 40.5 + C 25 + D 0  -> share 113/350
    images matched 5 (i6 is unmatched), segments with images 3 of 4

    by road class  residential (A + D): 47.5 / 200, 2 images
                   primary (B):         40.5 / 100, 2 images
                   footway (C):         25 / 50,    1 image

    areas          west  x in [-10, 60]:  A[0..60] + C   -> network 110, covered 22.5+22.5+25 = 70
                   east  x in [60, 400]:  A[60..100] + B + D -> network 240, covered 2.5+40.5 = 43

    recency (newest image per segment, edges 2018 and 2021)
                   A 2021 -> "2019-2021"; B 2023, C 2024 -> ">= 2022"; D -> "no imagery"

No real data is used.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from geoai_vlm.coverage import COVERAGE_NOTE, coverage_by_area, network_coverage, recency_report
from geoai_vlm.segments import aggregate_segments, snap_images_to_segments
from tests.test_segments import CRS, images, network  # noqa: F401 - fixtures


@pytest.fixture
def summary(network, images):  # noqa: F811
    return aggregate_segments(snap_images_to_segments(images, network), network)


class TestNetworkCoverage:
    def test_overall(self, summary):
        row = network_coverage(summary).iloc[0]
        assert row["group"] == "all"
        assert row["n_segments"] == 4
        assert row["network_length_m"] == pytest.approx(350)
        assert row["covered_length_m"] == pytest.approx(113)
        assert row["covered_length_share"] == pytest.approx(113 / 350)
        assert row["n_segments_with_images"] == 3
        assert row["segments_with_images_share"] == pytest.approx(0.75)
        assert row["n_images"] == 5
        assert row["images_per_km"] == pytest.approx(5 / 0.35)

    def test_by_road_class(self, summary):
        table = network_coverage(summary, group_column="road_class").set_index("road_class")
        assert table.loc["residential", "network_length_m"] == pytest.approx(200)
        assert table.loc["residential", "covered_length_share"] == pytest.approx(47.5 / 200)
        assert table.loc["residential", "n_segments_with_images"] == 1
        assert table.loc["primary", "covered_length_share"] == pytest.approx(0.405)
        assert table.loc["footway", "covered_length_share"] == pytest.approx(0.5)
        assert table["n_images"].sum() == 5

    def test_image_count_is_not_coverage(self, summary):
        """Many images on one spot must raise the count, not the coverage."""
        crowded = summary.copy()
        crowded.loc[crowded["segment_id"] == "C", "n_images"] = 500
        before = network_coverage(summary).iloc[0]
        after = network_coverage(crowded).iloc[0]
        assert after["n_images"] > before["n_images"]
        assert after["covered_length_share"] == pytest.approx(before["covered_length_share"])
        assert network_coverage(summary).attrs["note"] == COVERAGE_NOTE

    def test_missing_group_values_are_reported(self, summary):
        partial = summary.copy()
        partial.loc[partial["segment_id"] == "D", "road_class"] = None
        table = network_coverage(partial, group_column="road_class")
        assert "(unknown)" in set(table["road_class"])
        assert table["network_length_m"].sum() == pytest.approx(350)

    def test_rejects_non_summary_tables(self, images):  # noqa: F811
        with pytest.raises(ValueError, match="not a segment summary"):
            network_coverage(images)


class TestCoverageByArea:
    @pytest.fixture
    def areas(self):
        return gpd.GeoDataFrame(
            {"area_id": ["west", "east"]},
            geometry=[box(-10, -10, 60, 60), box(60, -10, 400, 160)],
            crs=CRS,
        )

    def test_lengths_are_clipped_to_areas(self, summary, areas):
        table = coverage_by_area(summary, areas).set_index("area_id")
        assert table.loc["west", "network_length_m"] == pytest.approx(110)
        assert table.loc["west", "covered_length_m"] == pytest.approx(70)
        assert table.loc["east", "network_length_m"] == pytest.approx(240)
        assert table.loc["east", "covered_length_m"] == pytest.approx(43)
        assert table.loc["west", "covered_length_share"] == pytest.approx(70 / 110)

    def test_images_are_counted_by_their_own_location(self, summary, areas, images):  # noqa: F811
        table = coverage_by_area(summary, areas, images=images).set_index("area_id")
        assert table.loc["west", "n_images"] == 3  # i1, i2, i5
        assert table.loc["east", "n_images"] == 3  # i3, i4 and the unmatched i6
        assert "n_images" not in coverage_by_area(summary, areas).columns

    def test_works_from_geographic_coordinates(self, summary, areas):
        geo_summary, geo_areas = summary.to_crs("EPSG:4326"), areas.to_crs("EPSG:4326")
        # to_crs() leaves covered_geometry in its own CRS; the report must cope.
        exact = coverage_by_area(geo_summary, geo_areas, metric_crs=CRS).set_index("area_id")
        assert exact.loc["west", "covered_length_m"] == pytest.approx(70, rel=1e-6)
        # With the default (an estimated UTM zone, here 30N rather than 31N)
        # lengths differ by the projections' scale factors, well under 1 %.
        estimated = coverage_by_area(geo_summary, geo_areas).set_index("area_id")
        assert estimated.loc["west", "covered_length_m"] == pytest.approx(70, rel=1e-2)


class TestRecency:
    def test_length_share_by_newest_capture_year(self, summary):
        table = recency_report(summary, year_edges=(2018, 2021)).set_index("recency")
        assert list(table.index) == ["<= 2018", "2019-2021", ">= 2022", "no imagery"]
        assert table.loc["<= 2018", "length_share"] == pytest.approx(0)
        assert table.loc["2019-2021", "length_share"] == pytest.approx(100 / 350)
        assert table.loc[">= 2022", "length_share"] == pytest.approx(150 / 350)
        assert table.loc["no imagery", "length_share"] == pytest.approx(100 / 350)
        assert table["length_share"].sum() == pytest.approx(1)
        assert table.loc[">= 2022", "n_segments"] == 2

    def test_per_group_shares_sum_to_one(self, summary):
        table = recency_report(summary, year_edges=(2018, 2021), group_column="road_class")
        sums = table.groupby("road_class")["length_share"].sum()
        assert np.allclose(sums.values, 1.0)
        residential = table[table["road_class"] == "residential"].set_index("recency")
        assert residential.loc["no imagery", "length_share"] == pytest.approx(0.5)

    def test_oldest_capture_can_be_used(self, summary):
        table = recency_report(summary, year_edges=(2018, 2021), time_column="first_capture").set_index("recency")
        # B's oldest image is from 2018.
        assert table.loc["<= 2018", "length_share"] == pytest.approx(100 / 350)

    def test_edges_must_increase(self, summary):
        with pytest.raises(ValueError, match="increasing"):
            recency_report(summary, year_edges=(2021, 2018))
