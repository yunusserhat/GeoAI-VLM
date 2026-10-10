# -*- coding: utf-8 -*-
"""
Street-segment matching and aggregation on a small synthetic network (P1-D).

Everything is in metres (EPSG:32631), so every expected number below is worked
out by hand from the coordinates. The network and images are synthetic.

    Segments                                   Images (x, y)  -> expected match
    A  (0,0)-(100,0)     100 m  residential    i1 (10, 5)     -> A, 5 m, offset 10
    B  (100,0)-(100,100) 100 m  primary        i2 (50, -3)    -> A, 3 m, offset 50
    C  (0,50)-(50,50)     50 m  footway        i3 (98, 3)     -> B, 2 m (A at 3 m: ambiguous)
    D  (200,0)-(300,0)   100 m  residential    i4 (105, 60)   -> B, 5 m, offset 60
                                               i5 (25, 35)    -> C, 15 m, offset 25
                                               i6 (150, 150)  -> unmatched (> 20 m)

With 25 m support (12.5 m either side of each image):
    A: [0, 22.5] + [37.5, 62.5]  -> 47.5 m of 100 m
    B: [0, 15.5] + [47.5, 72.5]  -> 40.5 m of 100 m
    C: [12.5, 37.5]              -> 25 m of 50 m
    D: nothing                   -> 0 m of 100 m
"""

from __future__ import annotations


import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from geoai_vlm.segments import (
    aggregate_segments,
    prepare_segments,
    read_segment_metadata,
    segments_from_graph,
    snap_images_to_segments,
    write_segment_parquet,
)

CRS = "EPSG:32631"


@pytest.fixture
def network():
    return prepare_segments(
        gpd.GeoDataFrame(
            {
                "segment_id": ["A", "B", "C", "D"],
                "road_class": ["residential", "primary", "footway", "residential"],
            },
            geometry=[
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (100, 100)]),
                LineString([(0, 50), (50, 50)]),
                LineString([(200, 0), (300, 0)]),
            ],
            crs=CRS,
        )
    )


@pytest.fixture
def images():
    return gpd.GeoDataFrame(
        {
            "image_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
            "sequence_id": ["s1", "s1", "s2", "s2", None, "s3"],
            "captured_at": pd.to_datetime(
                ["2019-05-01", "2021-06-01", "2023-01-01", "2018-01-01", "2024-07-01", "2020-01-01"]
            ),
            "green_fraction": [0.2, 0.4, 0.1, None, 0.5, 0.9],
            "usable": [True, None, False, True, "true", True],
            "audit_sidewalk_state": ["present", "absent", "not_visible", "present", "failed", "present"],
        },
        geometry=[Point(10, 5), Point(50, -3), Point(98, 3), Point(105, 60), Point(25, 35), Point(150, 150)],
        crs=CRS,
    )


def _by_id(frame, column="segment_id"):
    return frame.set_index(column)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
class TestSnapping:
    def test_nearest_segment_offsets_and_status(self, network, images):
        snapped = _by_id(snap_images_to_segments(images, network), "image_id")
        assert snapped.loc["i1", "segment_id"] == "A"
        assert snapped.loc["i1", "snap_distance_m"] == pytest.approx(5)
        assert snapped.loc["i1", "snap_offset_m"] == pytest.approx(10)
        assert snapped.loc["i2", "snap_offset_m"] == pytest.approx(50)
        assert snapped.loc["i4", "segment_id"] == "B"
        assert snapped.loc["i4", "snap_offset_m"] == pytest.approx(60)
        assert snapped.loc["i5", "segment_id"] == "C"
        assert snapped.loc["i5", "snap_distance_m"] == pytest.approx(15)
        assert snapped.loc["i1", "snap_status"] == "matched"

    def test_runner_up_within_two_metres_is_flagged(self, network, images):
        snapped = _by_id(snap_images_to_segments(images, network), "image_id")
        row = snapped.loc["i3"]
        assert row["segment_id"] == "B"  # 2 m beats 3 m
        assert row["second_segment_id"] == "A"
        assert row["second_distance_m"] == pytest.approx(3)
        assert bool(row["snap_ambiguous"]) is True
        assert row["snap_status"] == "ambiguous"

    def test_margin_is_a_parameter(self, network, images):
        snapped = _by_id(snap_images_to_segments(images, network, ambiguity_margin_m=0.5), "image_id")
        assert snapped.loc["i3", "snap_status"] == "matched"

    def test_beyond_max_distance_is_unmatched(self, network, images):
        snapped = _by_id(snap_images_to_segments(images, network), "image_id")
        assert snapped.loc["i6", "snap_status"] == "unmatched"
        assert pd.isna(snapped.loc["i6", "segment_id"])
        assert np.isnan(snapped.loc["i6", "snap_distance_m"])

    def test_max_distance_is_a_parameter(self, network, images):
        tight = _by_id(snap_images_to_segments(images, network, max_distance_m=10), "image_id")
        assert tight.loc["i5", "snap_status"] == "unmatched"  # 15 m away

    def test_equal_distances_resolve_to_the_smaller_id(self):
        segs = prepare_segments(
            gpd.GeoDataFrame(
                {"segment_id": ["F", "E"]},
                geometry=[LineString([(0, 10), (100, 10)]), LineString([(0, 0), (100, 0)])],
                crs=CRS,
            )
        )
        pts = gpd.GeoDataFrame({"image_id": ["t"]}, geometry=[Point(50, 5)], crs=CRS)
        row = snap_images_to_segments(pts, segs).iloc[0]
        assert row["segment_id"] == "E"
        assert row["second_segment_id"] == "F"
        assert row["snap_status"] == "ambiguous"

    def test_result_does_not_depend_on_row_order(self, network, images):
        reference = snap_images_to_segments(images, network).set_index("image_id")
        shuffled = snap_images_to_segments(
            images.sample(frac=1, random_state=3), network.sample(frac=1, random_state=4)
        ).set_index("image_id").loc[reference.index]
        pd.testing.assert_series_equal(reference["segment_id"], shuffled["segment_id"])
        pd.testing.assert_series_equal(reference["snap_status"], shuffled["snap_status"])

    def test_geographic_input_is_projected_for_distances(self, network, images):
        snapped = snap_images_to_segments(images.to_crs("EPSG:4326"), network.to_crs("EPSG:4326"))
        row = snapped.set_index("image_id").loc["i1"]
        assert row["segment_id"] == "A"
        assert row["snap_distance_m"] == pytest.approx(5, abs=0.05)

    def test_rules_are_recorded(self, network, images):
        rules = snap_images_to_segments(images, network).attrs["snap_rules"]
        assert rules["max_distance_m"] == 20 and rules["ambiguity_margin_m"] == 2
        assert "smaller segment id" in rules["tie_rule"]

    def test_crs_is_required(self, network, images):
        with pytest.raises(ValueError, match="CRS"):
            snap_images_to_segments(images.set_crs(None, allow_override=True), network)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@pytest.fixture
def summary(network, images):
    snapped = snap_images_to_segments(images, network)
    return aggregate_segments(
        snapped,
        network,
        numeric_columns=["green_fraction"],
        boolean_columns=["usable"],
        state_columns=["audit_sidewalk_state"],
    )


class TestAggregation:
    def test_every_segment_is_kept(self, summary):
        assert list(summary["segment_id"]) == ["A", "B", "C", "D"]

    def test_counts(self, summary):
        s = _by_id(summary)
        assert s.loc["A", "n_images"] == 2 and s.loc["B", "n_images"] == 2
        assert s.loc["C", "n_images"] == 1 and s.loc["D", "n_images"] == 0
        assert s.loc["B", "n_ambiguous_images"] == 1
        assert s.loc["A", "n_sequences"] == 1 and s.loc["B", "n_sequences"] == 1
        assert s.loc["C", "n_sequences"] == 0  # its only image has no sequence id

    def test_capture_dates(self, summary):
        s = _by_id(summary)
        assert s.loc["A", "first_capture"] == pd.Timestamp("2019-05-01", tz="UTC")
        assert s.loc["A", "last_capture"] == pd.Timestamp("2021-06-01", tz="UTC")
        assert s.loc["B", "first_capture"] == pd.Timestamp("2018-01-01", tz="UTC")
        assert s.loc["B", "median_capture"] == pd.Timestamp("2020-07-02", tz="UTC")
        assert pd.isna(s.loc["D", "last_capture"])

    def test_covered_length_share(self, summary):
        s = _by_id(summary)
        assert s.loc["A", "segment_length_m"] == pytest.approx(100)
        assert s.loc["A", "covered_length_m"] == pytest.approx(47.5)
        assert s.loc["B", "covered_length_m"] == pytest.approx(40.5)
        assert s.loc["C", "covered_length_share"] == pytest.approx(0.5)
        assert s.loc["D", "covered_length_share"] == pytest.approx(0.0)
        assert s.loc["A", "covered_geometry"].length == pytest.approx(47.5)

    def test_support_length_is_a_parameter(self, network, images):
        snapped = snap_images_to_segments(images, network)
        wide = _by_id(aggregate_segments(snapped, network, support_length_m=50))
        # A: [0, 35] + [25, 75] -> [0, 75]
        assert wide.loc["A", "covered_length_m"] == pytest.approx(75)

    def test_ambiguous_matches_can_be_excluded(self, network, images):
        snapped = snap_images_to_segments(images, network)
        strict = _by_id(aggregate_segments(snapped, network, include_ambiguous=False))
        assert strict.loc["B", "n_images"] == 1
        assert strict.loc["B", "n_ambiguous_images"] == 1  # still reported
        assert strict.loc["B", "covered_length_m"] == pytest.approx(25)

    def test_numeric_summary(self, summary):
        s = _by_id(summary)
        assert s.loc["A", "green_fraction__mean"] == pytest.approx(0.3)
        assert s.loc["B", "green_fraction__mean"] == pytest.approx(0.1)
        assert s.loc["B", "green_fraction__n"] == 1
        assert s.loc["B", "green_fraction__n_missing"] == 1
        assert np.isnan(s.loc["D", "green_fraction__mean"])

    def test_boolean_summary_counts_only_real_booleans(self, summary):
        s = _by_id(summary)
        assert s.loc["A", "usable__share_true"] == pytest.approx(1.0)
        assert s.loc["A", "usable__n_missing"] == 1
        assert s.loc["B", "usable__share_true"] == pytest.approx(0.5)
        assert np.isnan(s.loc["C", "usable__share_true"])  # the string "true" is not a verdict
        assert s.loc["C", "usable__n_missing"] == 1

    def test_state_summary_keeps_unknowns_out_of_the_rate(self, summary):
        s = _by_id(summary)
        assert s.loc["A", "audit_sidewalk_state__present_share"] == pytest.approx(0.5)
        # B: one present, one not_visible -> not_visible is not "absent"
        assert s.loc["B", "audit_sidewalk_state__present_share"] == pytest.approx(1.0)
        assert s.loc["B", "audit_sidewalk_state__n_not_visible"] == 1
        assert s.loc["C", "audit_sidewalk_state__n_failed"] == 1
        assert np.isnan(s.loc["C", "audit_sidewalk_state__present_share"])
        assert np.isnan(s.loc["D", "audit_sidewalk_state__present_share"])

    def test_metadata_records_unit_and_rules(self, summary):
        meta = summary.attrs["geoai_vlm"]
        assert meta["analysis_unit"] == "street_segment"
        assert meta["rules"]["support_length_m"] == 25
        assert meta["rules"]["max_distance_m"] == 20
        assert meta["n_images_unmatched"] == 1
        assert meta["measures"]["state"] == ["audit_sidewalk_state"]
        assert any("not network coverage" in n for n in meta["notes"])

    def test_unknown_measure_column_is_an_error(self, network, images):
        snapped = snap_images_to_segments(images, network)
        with pytest.raises(ValueError, match="not found"):
            aggregate_segments(snapped, network, numeric_columns=["nope"])

    def test_epoch_milliseconds_are_understood(self, network, images):
        epoch = images.copy()
        epoch["captured_at"] = (
            (images["captured_at"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
        ).astype("int64")  # Mapillary's captured_at is epoch milliseconds
        snapped = snap_images_to_segments(epoch, network)
        s = _by_id(aggregate_segments(snapped, network))
        assert s.loc["A", "first_capture"] == pd.Timestamp("2019-05-01", tz="UTC")


class TestGeoParquet:
    def test_metadata_travels_with_the_file(self, summary, tmp_path):
        path = write_segment_parquet(summary, tmp_path / "segments.parquet")
        meta = read_segment_metadata(path)
        assert meta["analysis_unit"] == "street_segment"
        assert meta["rules"]["ambiguity_margin_m"] == 2
        back = gpd.read_parquet(path)
        assert len(back) == 4
        assert back.crs == summary.crs
        assert "covered_length_share" in back.columns


# ---------------------------------------------------------------------------
# Network preparation
# ---------------------------------------------------------------------------
class _FakeGraph:
    """The two networkx views segments_from_graph uses."""

    def __init__(self, nodes, edges, crs=None):
        self._nodes, self._edges = nodes, edges
        self.graph = {"crs": crs} if crs else {}

    def nodes(self, data=False):
        return list(self._nodes.items())

    def edges(self, keys=False, data=False):
        return list(self._edges)


class TestNetworkPreparation:
    def test_graph_edges_become_segments_once_per_street(self):
        graph = _FakeGraph(
            {1: {"x": 0, "y": 0}, 2: {"x": 100, "y": 0}, 3: {"x": 100, "y": 100}},
            [
                (1, 2, 0, {"highway": "residential", "osmid": 10}),
                (2, 1, 0, {"highway": "residential", "osmid": 10}),
                (2, 3, 0, {"highway": ["primary", "secondary"], "osmid": [11, 12],
                           "geometry": LineString([(100, 0), (100, 50), (100, 100)])}),
            ],
            crs=CRS,
        )
        segs = segments_from_graph(graph)
        assert list(segs["segment_id"]) == ["1-2-0", "2-3-0"]
        assert list(segs["road_class"]) == ["residential", "primary|secondary"]
        assert segs.crs == CRS
        assert len(segments_from_graph(graph, dedupe_bidirectional=False)) == 3

    def test_duplicate_geometries_warn(self):
        lines = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (0, 0)])], crs=CRS
        )
        with pytest.warns(UserWarning, match="duplicate"):
            prepare_segments(lines)

    def test_ids_are_assigned_and_must_be_unique(self):
        lines = gpd.GeoDataFrame(
            {"name": ["x", "x"]},
            geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 1)])],
            crs=CRS,
        )
        assert list(prepare_segments(lines)["segment_id"]) == ["0", "1"]
        with pytest.raises(ValueError, match="unique"):
            prepare_segments(lines, id_column="name")

    def test_mergeable_multilines_are_merged_others_rejected(self):
        ok = gpd.GeoDataFrame(
            geometry=[MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])], crs=CRS
        )
        assert prepare_segments(ok).geometry.iloc[0].geom_type == "LineString"
        bad = gpd.GeoDataFrame(
            geometry=[MultiLineString([[(0, 0), (1, 0)], [(5, 5), (6, 5)]])], crs=CRS
        )
        with pytest.raises(ValueError, match="explode"):
            prepare_segments(bad)

    def test_crs_is_required(self):
        with pytest.raises(ValueError, match="CRS"):
            prepare_segments(gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])]))

    def test_osm_download_names_the_extra(self, monkeypatch):
        import sys

        from geoai_vlm.segments import load_osm_segments

        monkeypatch.setitem(sys.modules, "osmnx", None)
        with pytest.raises(ImportError, match=r"geoai-vlm\[network\]"):
            load_osm_segments(place="Anywhere")
