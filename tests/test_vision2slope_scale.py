# -*- coding: utf-8 -*-
"""
Vision2Slope scale and record-integrity tests.

The 2026-09-12 review flagged the parallel slope path as unsuitable for
city-scale production: the worker could not be pickled, and its body rebuilt the
segmentation model for every image. It also flagged that model configuration was
not reaching the model, and that the evaluation table silently dropped failed and
filtered records from its own denominator.

These tests run on CPU. They never download a model: the segmentation model is
always replaced by a counting stub.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Vision2Slope needs the optional vision stack. Skip the whole module rather
# than fail when only the core package is installed; a dedicated CI job
# installs the slope extra and runs these for real.
pytest.importorskip("cv2", reason="requires the 'slope' extra")
pytest.importorskip("torch", reason="requires the 'slope' extra")
pytest.importorskip("transformers", reason="requires the 'slope' extra")
pytest.importorskip("skimage", reason="requires the 'slope' extra")


# ---------------------------------------------------------------------------
# The panorama step must not drag zensvi into every slope import
# ---------------------------------------------------------------------------
class TestOptionalPanoramaDependency:
    """zensvi resolves to ~109 packages; only the panorama step needs it."""

    def test_slope_pipeline_imports_without_zensvi(self):
        code = (
            "import sys\n"
            "class _Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'zensvi':\n"
            "            raise ModuleNotFoundError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Blocker())\n"
            "from geoai_vlm import Vision2SlopePipeline, PipelineConfig, ModelConfig\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, (
            "the slope pipeline still requires zensvi to import:\n"
            f"{proc.stderr[-1500:]}"
        )
        assert "OK" in proc.stdout

    def test_panorama_transform_names_the_extra_when_zensvi_is_absent(self):
        """Using the panorama step without zensvi must say what to install."""
        code = (
            "import sys\n"
            "class _Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'zensvi':\n"
            "            raise ModuleNotFoundError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Blocker())\n"
            "from geoai_vlm.vision2slope.pano2perspective import PanoramaTransformer\n"
            "t = PanoramaTransformer()\n"
            "try:\n"
            "    t.transform_panorama('in', 'out')\n"
            "except ImportError as exc:\n"
            "    assert 'zensvi' in str(exc), str(exc)\n"
            "    assert 'geoai-vlm[' in str(exc), str(exc)\n"
            "    print('OK')\n"
            "else:\n"
            "    raise AssertionError('expected an ImportError naming the extra')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# The parallel path
# ---------------------------------------------------------------------------
class _CountingSegmentationModel:
    """Stand-in for SegmentationModel that records how often it is built."""

    construction_count = 0

    def __init__(self, model_name, device=None, cache_dir=None):
        type(self).construction_count += 1
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

    def segment(self, image):
        return np.zeros((4, 4), dtype=np.int32)


class TestParallelWorkerContract:
    """A worker that cannot be pickled never runs; one that reloads the model
    per image cannot scale to a city."""

    def test_worker_function_is_picklable(self):
        """multiprocessing sends the callable to the child by pickle."""
        from geoai_vlm.vision2slope import pipeline as pl

        worker = getattr(pl, "_slope_worker", None)
        assert worker is not None, (
            "the parallel worker must be a module-level function; a closure "
            "defined inside the method cannot cross a process boundary"
        )
        assert pickle.loads(pickle.dumps(worker)) is worker

    def test_worker_survives_the_multiprocessing_pickler(self):
        """This is the exact serialiser Pool uses."""
        from multiprocessing.reduction import ForkingPickler

        from geoai_vlm.vision2slope import pipeline as pl

        data = ForkingPickler.dumps(pl._slope_worker)
        assert pickle.loads(data) is pl._slope_worker

    def test_component_config_is_picklable(self):
        """The worker initialiser has to carry the component settings across."""
        from geoai_vlm.vision2slope.pipeline import LegacyComponentConfig

        cfg = LegacyComponentConfig(canny_threshold1=1.0, canny_threshold2=2.0)
        restored = pickle.loads(pickle.dumps(cfg))
        assert restored.canny_threshold1 == 1.0
        assert restored.canny_threshold2 == 2.0

    def test_model_is_built_once_per_worker_not_once_per_image(self, monkeypatch):
        """The original worker body constructed the model inside every call."""
        from geoai_vlm.vision2slope import pipeline as pl

        monkeypatch.setattr(pl, "SegmentationModel", _CountingSegmentationModel)
        _CountingSegmentationModel.construction_count = 0

        class _StubProcessor:
            def __init__(self, **kwargs):
                self.segmentation_provider = kwargs.get("segmentation_provider")

            def process(self, image_path):
                return {"filename": image_path, "status": "success"}

        monkeypatch.setattr(pl, "StandardImageProcessor", _StubProcessor)

        pl._init_slope_worker(
            model_name="stub/model",
            device="cpu",
            cache_dir=None,
            component_config=pl.LegacyComponentConfig(),
        )

        for path in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
            pl._slope_worker(path)

        assert _CountingSegmentationModel.construction_count == 1, (
            "the segmentation model was rebuilt per image "
            f"({_CountingSegmentationModel.construction_count} times for 4 images)"
        )

    def test_worker_without_initialisation_fails_loudly(self):
        from geoai_vlm.vision2slope import pipeline as pl

        pl._WORKER_STATE.clear()
        with pytest.raises(RuntimeError):
            pl._slope_worker("a.jpg")


# ---------------------------------------------------------------------------
# Model configuration must reach the model
# ---------------------------------------------------------------------------
class TestModelConfigPlumbing:
    def test_segmentation_model_accepts_device_and_cache_dir(self, monkeypatch):
        from geoai_vlm.vision2slope.models import SegmentationModel

        monkeypatch.setattr(SegmentationModel, "_load_model", lambda self: None)
        model = SegmentationModel(
            "stub/model", device="cpu", cache_dir="/tmp/hf-cache"
        )
        assert str(model.device) == "cpu"
        assert model.cache_dir == "/tmp/hf-cache"

    def test_device_none_still_auto_detects(self, monkeypatch):
        from geoai_vlm.vision2slope.models import SegmentationModel

        monkeypatch.setattr(SegmentationModel, "_load_model", lambda self: None)
        model = SegmentationModel("stub/model")
        assert str(model.device) in ("cpu", "cuda")

    def test_worker_initialiser_forwards_device_and_cache_dir(self, monkeypatch):
        from geoai_vlm.vision2slope import pipeline as pl

        monkeypatch.setattr(pl, "SegmentationModel", _CountingSegmentationModel)
        _CountingSegmentationModel.construction_count = 0

        captured = {}

        class _StubProcessor:
            def __init__(self, **kwargs):
                captured["provider"] = kwargs.get("segmentation_provider")

            def process(self, image_path):
                return {}

        monkeypatch.setattr(pl, "StandardImageProcessor", _StubProcessor)

        pl._init_slope_worker(
            model_name="stub/model",
            device="cpu",
            cache_dir="/tmp/cache-x",
            component_config=pl.LegacyComponentConfig(),
        )

        provider = captured["provider"]
        assert provider.device == "cpu"
        assert provider.cache_dir == "/tmp/cache-x"


# ---------------------------------------------------------------------------
# Every processed image must stay in the record table
# ---------------------------------------------------------------------------
class TestRecordCompleteness:
    """Failures and filtered angles must not vanish from the denominator."""

    @pytest.fixture
    def results(self):
        return pd.DataFrame(
            {
                "filename": [
                    "ok_shallow.jpg",
                    "ok_shallow2.jpg",
                    "too_steep.jpg",
                    "failed.jpg",
                    "no_angle.jpg",
                ],
                "status": ["success", "success", "success", "failed", "success"],
                "road_edge_line_angle": [2.0, -3.0, 45.0, np.nan, np.nan],
            }
        )

    def test_every_input_appears_with_a_status(self, results):
        from geoai_vlm.vision2slope.pipeline import build_record_table

        records = build_record_table(results, angle_threshold=10.0)
        assert len(records) == len(results), "records were dropped from the table"
        assert set(records["filename"]) == set(results["filename"])
        assert "evaluation_status" in records.columns

    def test_the_four_outcomes_are_distinguishable(self, results):
        from geoai_vlm.vision2slope.pipeline import build_record_table

        records = build_record_table(results, angle_threshold=10.0).set_index("filename")
        assert records.loc["ok_shallow.jpg", "evaluation_status"] == "measured"
        assert records.loc["ok_shallow2.jpg", "evaluation_status"] == "measured"
        assert records.loc["too_steep.jpg", "evaluation_status"] == "angle_filtered"
        assert records.loc["failed.jpg", "evaluation_status"] == "processing_failed"
        assert records.loc["no_angle.jpg", "evaluation_status"] == "angle_missing"

    def test_success_rate_denominator_is_all_processed_images(self, results):
        from geoai_vlm.vision2slope.pipeline import summarise_records

        summary = summarise_records(build_records(results))
        assert summary["n_processed"] == 5, (
            "the denominator must be every image that entered processing"
        )
        assert summary["n_measured"] == 2
        assert summary["n_processing_failed"] == 1
        assert summary["n_angle_filtered"] == 1
        assert summary["n_angle_missing"] == 1
        assert summary["measured_rate"] == pytest.approx(2 / 5)

    def test_measurement_subset_contains_only_measured_rows(self, results):
        from geoai_vlm.vision2slope.pipeline import build_record_table

        records = build_record_table(results, angle_threshold=10.0)
        measured = records[records["evaluation_status"] == "measured"]
        assert set(measured["filename"]) == {"ok_shallow.jpg", "ok_shallow2.jpg"}

    def test_empty_input_is_handled(self):
        from geoai_vlm.vision2slope.pipeline import build_record_table, summarise_records

        empty = pd.DataFrame(columns=["filename", "status", "road_edge_line_angle"])
        records = build_record_table(empty, angle_threshold=10.0)
        assert len(records) == 0
        summary = summarise_records(records)
        assert summary["n_processed"] == 0
        assert summary["measured_rate"] == 0.0


def build_records(results, threshold=10.0):
    from geoai_vlm.vision2slope.pipeline import build_record_table

    return build_record_table(results, angle_threshold=threshold)


# ---------------------------------------------------------------------------
# Panorama metadata
# ---------------------------------------------------------------------------
class TestPanoramaMetadata:
    def test_unknown_heading_is_not_coerced_to_north(self):
        """`pano.heading or 0.0` silently turned an unknown heading into 0 deg."""
        from geoai_vlm.vision2slope.downloader import normalise_heading

        assert normalise_heading(None) is None, (
            "an unknown heading must stay unknown, not become due north"
        )
        assert normalise_heading(0.0) == 0.0
        assert normalise_heading(123.5) == pytest.approx(123.5)

    def test_a_panorama_may_match_several_road_segments(self):
        """Keeping the first row per pano discarded other road directions."""
        from geoai_vlm.vision2slope.pipeline import select_pano_metadata

        metadata = pd.DataFrame(
            {
                "pano_id": ["p1", "p1", "p2"],
                "heading": [10.0, 10.0, 20.0],
                "edge_bearing": [90.0, 270.0, 0.0],
            }
        )
        kept = select_pano_metadata(metadata)
        p1 = kept[kept["pano_id"] == "p1"]
        assert len(p1) == 2, (
            "both road directions for a panorama must be kept, not just the first"
        )
        assert set(p1["edge_bearing"]) == {90.0, 270.0}


# ---------------------------------------------------------------------------
# Visualizer output directories
# ---------------------------------------------------------------------------
class TestVisualizerDirectories:
    """Each output option must create its own directory, independently."""

    def _viz_config(self, **overrides):
        from geoai_vlm.vision2slope.config import VisualizationConfig

        cfg = VisualizationConfig()
        for key in (
            "save_visualizations",
            "save_corrected_images",
            "save_intermediate_results",
            "save_segmentation_masks",
            "save_road_masks",
            "save_edge_images",
            "save_line_images",
            "save_road_edge_fitting",
        ):
            if hasattr(cfg, key):
                setattr(cfg, key, False)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_road_masks_alone_creates_the_mask_directory(self, tmp_path):
        """masks_dir was only created under save_segmentation_masks."""
        from geoai_vlm.vision2slope.visualizers import Visualizer

        viz = Visualizer(self._viz_config(save_road_masks=True), str(tmp_path))
        mask = np.zeros((8, 8), dtype=np.uint8)
        viz.save_road_mask(mask, "img.png")

        written = list(tmp_path.rglob("*road_mask*"))
        assert written, (
            f"road mask was not written; directories present: "
            f"{[p.name for p in tmp_path.iterdir()]}"
        )

    def test_segmentation_masks_alone_still_works(self, tmp_path):
        from geoai_vlm.vision2slope.visualizers import Visualizer

        viz = Visualizer(
            self._viz_config(save_segmentation_masks=True), str(tmp_path)
        )
        viz.save_segmentation_mask(np.zeros((8, 8), dtype=np.int32), "img.png")
        assert list(tmp_path.rglob("*segmentation*"))

    def test_both_mask_options_share_one_directory(self, tmp_path):
        from geoai_vlm.vision2slope.visualizers import Visualizer

        viz = Visualizer(
            self._viz_config(save_segmentation_masks=True, save_road_masks=True),
            str(tmp_path),
        )
        viz.save_segmentation_mask(np.zeros((8, 8), dtype=np.int32), "img.png")
        viz.save_road_mask(np.zeros((8, 8), dtype=np.uint8), "img.png")
        assert list(tmp_path.rglob("*segmentation*"))
        assert list(tmp_path.rglob("*road_mask*"))

    def test_no_mask_options_creates_no_mask_directory(self, tmp_path):
        from geoai_vlm.vision2slope.visualizers import Visualizer

        Visualizer(self._viz_config(), str(tmp_path))
        assert not list(tmp_path.glob("*mask*"))


# ---------------------------------------------------------------------------
# End-to-end: the pipeline must actually use the record table
# ---------------------------------------------------------------------------
class TestPipelineUsesRecordTable:
    """Having the helpers is not enough; the run has to go through them."""

    @pytest.fixture
    def pipeline(self, tmp_path, monkeypatch):
        from PIL import Image

        from geoai_vlm.vision2slope import pipeline as pl
        from geoai_vlm.vision2slope.config import PipelineConfig

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        names = ["shallow", "steep", "broken", "noangle"]
        for name in names:
            Image.new("RGB", (8, 8)).save(in_dir / f"{name}.jpg")

        outcomes = {
            "shallow": {"status": "success", "road_edge_line_angle": 3.0},
            "steep": {"status": "success", "road_edge_line_angle": 60.0},
            "broken": {"status": "failed", "road_edge_line_angle": np.nan},
            "noangle": {"status": "success", "road_edge_line_angle": np.nan},
        }

        class _StubResult:
            def __init__(self, path):
                self.path = path

            def to_dict(self):
                stem = Path(self.path).stem
                # Mirror the real ProcessingResult.to_dict column set.
                row = {
                    "filename": str(self.path),
                    "pano_id": stem,
                    "skew_angle": 1.0,
                    "skew_confidence": 0.9,
                    "num_lines_detected": 3,
                    "correction_applied": False,
                    "corrected_filename": None,
                    "road_edge_line_slope": 0.05,
                    "road_edge_line_intercept": 1.0,
                    "road_area": 100.0,
                    "stage_completed": "analysis",
                    "error_message": None,
                }
                row.update(outcomes[stem])
                return row

        class _StubProcessor:
            def process(self, image_path):
                return _StubResult(image_path)

        # Skip model loading entirely.
        monkeypatch.setattr(
            pl.Vision2SlopePipeline, "_create_processor", lambda self: _StubProcessor()
        )

        config = PipelineConfig(
            input_dir=str(in_dir), output_dir=str(tmp_path / "out")
        )
        config.analysis_config.filter_slope_angle = 10.0
        return pl.Vision2SlopePipeline(config), tmp_path / "out", names

    def test_record_csv_contains_every_input(self, pipeline):
        pipe, out_dir, names = pipeline
        pipe.process_batch()

        written = list(Path(out_dir).glob("vision2slope_records_*.csv"))
        assert written, (
            f"no record table written; files: {[p.name for p in Path(out_dir).iterdir()]}"
        )
        records = pd.read_csv(written[0])
        assert len(records) == len(names), (
            f"record table lost rows: {len(records)} of {len(names)}"
        )
        assert set(Path(f).stem for f in records["filename"]) == set(names)

    def test_record_csv_marks_each_outcome(self, pipeline):
        pipe, out_dir, _ = pipeline
        pipe.process_batch()

        records = pd.read_csv(list(Path(out_dir).glob("vision2slope_records_*.csv"))[0])
        by_stem = {
            Path(row["filename"]).stem: row["evaluation_status"]
            for _, row in records.iterrows()
        }
        assert by_stem["shallow"] == "measured"
        assert by_stem["steep"] == "angle_filtered"
        assert by_stem["broken"] == "processing_failed"
        assert by_stem["noangle"] == "angle_missing"

    def test_pipeline_exposes_the_records(self, pipeline):
        pipe, _, names = pipeline
        pipe.process_batch()
        assert hasattr(pipe, "records_")
        assert len(pipe.records_) == len(names)
