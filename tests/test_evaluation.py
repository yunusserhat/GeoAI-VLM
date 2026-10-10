# -*- coding: utf-8 -*-
"""
Evaluation tools (P1-F), tested against values worked out by hand.

    Cohen's kappa   2x2 table (both yes 20, A-yes/B-no 5, A-no/B-yes 10, both no 15):
                    po = 35/50 = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = 0.4
    ICC             Shrout & Fleiss (1979), 6 targets x 4 judges:
                    ICC(2,1) = 0.2898, ICC(3,1) = 0.7148 (published: .29, .71)
                    rater 2 = rater 1 + 1 on [1,2,3,4]: ICC(3,1) = 1, ICC(2,1) = 10/13
    Bland-Altman    a = [1,2,3,4,5], b = [1.5,2.5,2.5,4.5,5.5]: bias -0.3,
                    sd sqrt(0.2), limits -0.3 -/+ 1.96 sqrt(0.2)
    R-squared       y = 2x on [1,2,3,4]: pearson 1, identity 1 - 30/20 = -0.5

No real data and no unpublished numbers are used.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geoai_vlm.evaluation import (
    SplitLeakageError,
    agreement_report,
    assign_spatial_blocks,
    bland_altman,
    check_split_leakage,
    cluster_bootstrap_ci,
    cohen_kappa,
    generation_stability,
    grouped_split,
    icc,
    light_condition,
    percent_agreement,
    r_squared,
    repeated_generation,
    sequence_split,
    solar_elevation,
    spatial_block_split,
    stratified_reference_sample,
)

TWO_BY_TWO_A = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
TWO_BY_TWO_B = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15

SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
class TestCategoricalAgreement:
    def test_percent_agreement(self):
        assert percent_agreement(TWO_BY_TWO_A, TWO_BY_TWO_B) == pytest.approx(0.7)

    def test_kappa_hand_computed(self):
        assert cohen_kappa(TWO_BY_TWO_A, TWO_BY_TWO_B) == pytest.approx(0.4)

    def test_kappa_matches_scikit_learn(self):
        from sklearn.metrics import cohen_kappa_score

        rng = np.random.default_rng(1)
        a = rng.choice(["present", "absent", "uncertain"], 200)
        b = np.where(rng.random(200) < 0.7, a, rng.choice(["present", "absent", "uncertain"], 200))
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))

    @pytest.mark.parametrize("weights", ["linear", "quadratic"])
    def test_weighted_kappa_matches_scikit_learn(self, weights):
        from sklearn.metrics import cohen_kappa_score

        order = ["narrow", "medium", "wide"]
        rng = np.random.default_rng(2)
        a = rng.integers(0, 3, 150)
        b = np.clip(a + rng.integers(-1, 2, 150), 0, 2)
        ours = cohen_kappa([order[i] for i in a], [order[i] for i in b], labels=order, weights=weights)
        assert ours == pytest.approx(cohen_kappa_score(a, b, weights=weights))

    def test_missing_pairs_are_dropped(self):
        assert percent_agreement(["a", None, "b"], ["a", "b", "b"]) == pytest.approx(1.0)

    def test_kappa_is_undefined_without_variation(self):
        assert math.isnan(cohen_kappa(["a"] * 5, ["a"] * 5))

    def test_weighted_kappa_needs_an_order(self):
        with pytest.raises(ValueError, match="ordinal"):
            cohen_kappa(["a"], ["b"], weights="linear")

    def test_values_outside_labels_are_rejected(self):
        with pytest.raises(ValueError, match="not in labels"):
            cohen_kappa(["a", "c"], ["a", "b"], labels=["a", "b"])


class TestContinuousAgreement:
    def test_icc_shrout_fleiss(self):
        assert icc(SHROUT_FLEISS, "ICC(2,1)") == pytest.approx(0.2898, abs=1e-3)
        assert icc(SHROUT_FLEISS, "ICC(3,1)") == pytest.approx(0.7148, abs=1e-3)

    def test_icc_separates_consistency_from_absolute_agreement(self):
        offset = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        assert icc(offset, "ICC(3,1)") == pytest.approx(1.0)
        assert icc(offset, "ICC(2,1)") == pytest.approx(10 / 13)

    def test_icc_drops_incomplete_rows(self):
        with_gap = np.vstack([SHROUT_FLEISS, [np.nan, 1, 1, 1]])
        assert icc(with_gap, "ICC(2,1)") == pytest.approx(icc(SHROUT_FLEISS, "ICC(2,1)"))

    def test_icc_rejects_bad_input(self):
        with pytest.raises(ValueError):
            icc(np.ones((5, 1)))
        with pytest.raises(ValueError):
            icc(SHROUT_FLEISS, "ICC(1,1)")

    def test_bland_altman(self):
        result = bland_altman([1, 2, 3, 4, 5], [1.5, 2.5, 2.5, 4.5, 5.5])
        sd = math.sqrt(0.2)
        assert result["bias"] == pytest.approx(-0.3)
        assert result["sd"] == pytest.approx(sd)
        assert result["loa_lower"] == pytest.approx(-0.3 - 1.96 * sd)
        assert result["loa_upper"] == pytest.approx(-0.3 + 1.96 * sd)
        assert result["n"] == 5

    def test_r_squared_pearson_versus_identity(self):
        x, y = [1, 2, 3, 4], [2, 4, 6, 8]
        assert r_squared(x, y, "pearson") == pytest.approx(1.0)
        assert r_squared(x, y, "identity") == pytest.approx(-0.5)

    def test_r_squared_constant_series_is_undefined(self):
        assert math.isnan(r_squared([1, 1, 1], [1, 2, 3]))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def _images(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2000, n)
    y = rng.uniform(0, 2000, n)
    return gpd.GeoDataFrame(
        {
            "image_id": [f"img{i:03d}" for i in range(n)],
            "sequence_id": [f"seq{i // 4}" for i in range(n)],  # 4 images per drive
        },
        geometry=[Point(a, b) for a, b in zip(x, y)],
        crs="EPSG:32631",
    )


class TestSplits:
    @pytest.mark.parametrize("seed", range(5))
    def test_sequence_split_never_leaks(self, seed):
        split = sequence_split(_images(), test_size=0.25, seed=seed)
        check_split_leakage(split, ["sequence_id"])
        assert set(split["split"]) == {"train", "test"}
        assert split.attrs["split"]["seed"] == seed

    def test_split_is_reproducible(self):
        a = sequence_split(_images(), seed=7)["split"].tolist()
        b = sequence_split(_images(), seed=7)["split"].tolist()
        assert a == b

    def test_test_share_is_close_to_the_request(self):
        split = sequence_split(_images(200), test_size=0.2, seed=1)
        assert 0.2 <= split.attrs["split"]["test_size_achieved"] <= 0.25

    def test_spatial_blocks(self):
        pts = gpd.GeoDataFrame(geometry=[Point(10, 10), Point(499, 499), Point(501, 10)], crs="EPSG:32631")
        blocks = assign_spatial_blocks(pts, block_size_m=500)["block_id"].tolist()
        assert blocks[0] == blocks[1] != blocks[2]

    @pytest.mark.parametrize("seed", range(5))
    def test_block_split_with_sequences_leaks_neither(self, seed):
        split = spatial_block_split(_images(), block_size_m=400, test_size=0.3, seed=seed, sequence_column="sequence_id")
        check_split_leakage(split, ["block_id", "sequence_id"])

    def test_block_only_split_can_leak_sequences_and_the_check_says_so(self):
        """Sequences crossing blocks leak under a block-only split."""
        leaky_found = False
        for seed in range(10):
            split = spatial_block_split(_images(), block_size_m=400, test_size=0.3, seed=seed)
            check_split_leakage(split, ["block_id"])
            report = check_split_leakage(split, ["sequence_id"], raise_on_leak=False)
            leaky_found = leaky_found or bool(report["sequence_id"])
        assert leaky_found, "the fixture should contain drives that cross blocks"

    def test_leaky_split_is_rejected(self):
        df = pd.DataFrame({"sequence_id": ["s1", "s1", "s2"], "split": ["train", "test", "test"]})
        with pytest.raises(SplitLeakageError, match="sequence_id"):
            check_split_leakage(df, ["sequence_id"])

    def test_missing_group_values_do_not_link_rows(self):
        df = pd.DataFrame({"g": [None, None, "a", "a"]})
        split = grouped_split(df, ["g"], test_size=0.5, seed=0)
        assert split["split_component"].nunique() == 3

    def test_nullable_and_datetime_missing_values_do_not_link_rows(self):
        df = pd.DataFrame(
            {
                "seq": pd.array([pd.NA, pd.NA, 7, 7], dtype="Int64"),
                "day": pd.to_datetime([None, None, "2024-05-01", "2024-05-01"]),
            }
        )
        for column in ("seq", "day"):
            split = grouped_split(df, [column], test_size=0.5, seed=0)
            assert split["split_component"].nunique() == 3, column

    def test_bad_test_size(self):
        with pytest.raises(ValueError):
            grouped_split(pd.DataFrame({"g": [1]}), ["g"], test_size=1.0)


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------
class TestClusterBootstrap:
    @staticmethod
    def _clustered(n_clusters=20, size=10, seed=0):
        rng = np.random.default_rng(seed)
        levels = rng.normal(0, 1, n_clusters)
        return pd.DataFrame(
            {"cluster": np.repeat(np.arange(n_clusters), size), "value": np.repeat(levels, size)}
        )

    def test_reproducible_with_a_seed(self):
        data = self._clustered()
        a = cluster_bootstrap_ci(lambda d: d["value"].mean(), data, "cluster", n_boot=200, seed=3)
        b = cluster_bootstrap_ci(lambda d: d["value"].mean(), data, "cluster", n_boot=200, seed=3)
        assert a == b
        assert a["ci_low"] <= a["estimate"] <= a["ci_high"]
        assert a["n_clusters"] == 20 and a["seed"] == 3

    def test_clustered_data_get_wider_intervals_than_row_resampling(self):
        data = self._clustered()
        clustered = cluster_bootstrap_ci(lambda d: d["value"].mean(), data, "cluster", n_boot=400, seed=0)
        rows = data.assign(row=np.arange(len(data)))
        naive = cluster_bootstrap_ci(lambda d: d["value"].mean(), rows, "row", n_boot=400, seed=0)
        width = lambda r: r["ci_high"] - r["ci_low"]  # noqa: E731
        assert width(clustered) > 2 * width(naive)

    def test_agreement_report_with_intervals(self):
        rng = np.random.default_rng(5)
        truth = rng.choice(["present", "absent"], 120)
        model = np.where(rng.random(120) < 0.8, truth, rng.choice(["present", "absent"], 120))
        df = pd.DataFrame({"ref": truth, "model": model, "seq": np.repeat(np.arange(30), 4)})
        report = agreement_report(df, "ref", "model", cluster_column="seq", n_boot=200).set_index("metric")
        assert report.loc["cohen_kappa", "estimate"] == pytest.approx(cohen_kappa(truth, model))
        assert report.loc["cohen_kappa", "ci_low"] <= report.loc["cohen_kappa", "estimate"]
        assert report.loc["percent_agreement", "n_clusters"] == 30

    def test_continuous_report(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5.0], "b": [1.5, 2.5, 2.5, 4.5, 5.5]})
        report = agreement_report(df, "a", "b", kind="continuous").set_index("metric")
        assert report.loc["bland_altman_bias", "estimate"] == pytest.approx(-0.3)
        assert "ci_low" not in report.columns


# ---------------------------------------------------------------------------
# Repeated-generation stability
# ---------------------------------------------------------------------------
class TestGenerationStability:
    def test_modal_shares_by_hand(self):
        runs = pd.DataFrame(
            {
                "image_id": ["A"] * 4 + ["B"] * 4,
                "repeat": [0, 1, 2, 3] * 2,
                "state": ["x", "x", "x", "y", "z", "z", "z", "z"],
                "other": ["p", None, None, "p", "q", "q", "r", "s"],
            }
        )
        table = generation_stability(runs, ["state", "other"]).set_index("field")
        assert table.loc["state", "mean_modal_share"] == pytest.approx((0.75 + 1.0) / 2)
        assert table.loc["state", "share_fully_stable"] == pytest.approx(0.5)
        assert table.loc["state", "mean_n_distinct"] == pytest.approx(1.5)
        # A: p, <missing>, <missing>, p -> 0.5; B: q, q, r, s -> 0.5
        assert table.loc["other", "mean_modal_share"] == pytest.approx(0.5)
        assert table.loc["other", "min_modal_share"] == pytest.approx(0.5)

    def test_repeated_generation_runs_every_repeat(self, tmp_path):
        from PIL import Image

        from geoai_vlm.describer import ImageDescriber

        paths = []
        for i in range(2):
            p = tmp_path / f"r{i}.png"
            Image.new("RGB", (4, 4)).save(p)
            paths.append(p)

        class Sampler:
            temperature = 0.7
            seed = None

            def __init__(self):
                self.calls = 0

            def generate(self, image_paths, system_prompt, user_prompt):
                self.calls += 1
                tag = "quiet" if self.calls % 2 else "busy"
                return [f'{{"description": "{tag}", "tags": []}}'] * len(image_paths)

        d = ImageDescriber(model_name="org/model", prompt_template="simple")
        d._backend = Sampler()
        runs = repeated_generation(d, paths, n_repeats=3)
        assert len(runs) == 6 and sorted(runs["repeat"].unique()) == [0, 1, 2]
        table = generation_stability(runs, ["scene_narrative"]).set_index("field")
        assert table.loc["scene_narrative", "mean_modal_share"] == pytest.approx(2 / 3)

    def test_greedy_decoding_is_flagged(self, tmp_path):
        from PIL import Image

        from geoai_vlm.describer import ImageDescriber

        p = tmp_path / "g.png"
        Image.new("RGB", (4, 4)).save(p)

        class Greedy:
            temperature = 0.0

            def generate(self, image_paths, system_prompt, user_prompt):
                return ['{"description": "x", "tags": []}'] * len(image_paths)

        d = ImageDescriber(model_name="org/model", prompt_template="simple")
        d._backend = Greedy()
        with pytest.warns(UserWarning, match="greedy"):
            repeated_generation(d, [p], n_repeats=2)


# ---------------------------------------------------------------------------
# Daylight proxy and stratified sampling
# ---------------------------------------------------------------------------
class TestSolarProxy:
    def test_equator_equinox(self):
        # On 20 March the equation of time is about -7.4 min, so at 12:00 UTC
        # the hour angle at longitude 0 is about -1.85 degrees: elevation
        # about 88.1 at noon and -88.0 at midnight, not exactly +/-90.
        noon = solar_elevation(["2024-03-20T12:00:00Z"], 0.0, 0.0)[0]
        midnight = solar_elevation(["2024-03-20T00:00:00Z"], 0.0, 0.0)[0]
        assert noon == pytest.approx(88.1, abs=0.5)
        assert midnight == pytest.approx(-88.0, abs=0.5)

    def test_mid_latitude_solstice_noon(self):
        # 90 - 51.48 + 23.44 = 61.96 degrees at Greenwich on the June solstice.
        value = solar_elevation(["2024-06-21T12:00:00Z"], 51.48, 0.0)[0]
        assert value == pytest.approx(61.96, abs=0.5)

    def test_epoch_milliseconds_and_naive_times_are_utc(self):
        iso = solar_elevation(["2024-06-21T12:00:00Z"], 51.48, 0.0)[0]
        naive = solar_elevation([pd.Timestamp("2024-06-21 12:00:00")], 51.48, 0.0)[0]
        epoch_ms = solar_elevation([1718971200000], 51.48, 0.0)[0]
        assert iso == pytest.approx(naive) == pytest.approx(epoch_ms)

    def test_light_condition_classes(self):
        times = ["2024-06-21T12:00:00Z", "2024-06-21T20:30:00Z", "2024-06-21T23:59:00Z", None]
        labels = light_condition(times, 51.48, 0.0).tolist()
        assert labels == ["day", "twilight", "night", "unknown"]


class TestStratifiedSample:
    @staticmethod
    def _frame():
        rows = []
        for area in ("north", "south"):
            for road in ("primary", "residential"):
                n = 10 if road == "residential" else 3
                for i in range(n):
                    rows.append({"image_id": f"{area}-{road}-{i}", "area": area, "road_class": road})
        rows.append({"image_id": "x-0", "area": None, "road_class": "primary"})
        return pd.DataFrame(rows)

    def test_seed_makes_it_reproducible_and_is_recorded(self):
        df = self._frame()
        a = stratified_reference_sample(df, ["area", "road_class"], n_per_stratum=2, seed=11)
        b = stratified_reference_sample(df, ["area", "road_class"], n_per_stratum=2, seed=11)
        c = stratified_reference_sample(df, ["area", "road_class"], n_per_stratum=2, seed=12)
        assert a["image_id"].tolist() == b["image_id"].tolist()
        assert a["image_id"].tolist() != c["image_id"].tolist()
        assert set(a["sample_seed"]) == {11}
        assert a.attrs["sampling"]["seed"] == 11

    def test_quotas_and_shortfalls(self):
        df = self._frame()
        sample = stratified_reference_sample(df, ["area", "road_class"], n_per_stratum=5, seed=1)
        report = pd.DataFrame(sample.attrs["sampling"]["per_stratum"]).set_index("stratum")
        assert report.loc["north | residential", "selected"] == 5
        assert report.loc["north | primary", "selected"] == 3
        assert report.loc["north | primary", "shortfall"] == 2
        assert report.loc["(missing) | primary", "available"] == 1
        assert sample.groupby("stratum").size().to_dict() == report["selected"].to_dict()

    def test_proportional_allocation(self):
        df = self._frame()
        sample = stratified_reference_sample(df, ["road_class"], total=10, allocation="proportional", seed=1)
        report = pd.DataFrame(sample.attrs["sampling"]["per_stratum"]).set_index("stratum")
        assert report["quota"].sum() == 10
        assert report.loc["residential", "quota"] == 7  # 20 of 27 rows
        assert report.loc["primary", "quota"] == 3

    def test_equal_allocation_of_a_total(self):
        df = self._frame()
        sample = stratified_reference_sample(df, ["area"], total=7, seed=1)
        report = pd.DataFrame(sample.attrs["sampling"]["per_stratum"])
        assert report["quota"].tolist() == [3, 2, 2]

    def test_light_condition_as_a_stratum(self):
        df = pd.DataFrame(
            {
                "image_id": ["a", "b", "c", "d"],
                "captured_at": ["2024-06-21T12:00:00Z", "2024-06-21T12:30:00Z",
                                "2024-06-21T23:30:00Z", "2024-06-22T00:10:00Z"],
                "lat": [51.48] * 4,
                "lon": [0.0] * 4,
            }
        )
        df["light"] = light_condition(df["captured_at"], df["lat"], df["lon"])
        sample = stratified_reference_sample(df, ["light"], n_per_stratum=1, seed=3)
        assert sorted(sample["light"]) == ["day", "night"]

    def test_arguments_are_validated(self):
        with pytest.raises(ValueError, match="exactly one"):
            stratified_reference_sample(self._frame(), ["area"])
