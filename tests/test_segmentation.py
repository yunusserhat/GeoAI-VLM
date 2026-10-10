# -*- coding: utf-8 -*-
"""
Segmentation measurement adapter (P2-G), with fake model outputs.

The 4 x 4 mask below has 12 labelled pixels (four are 255 = ignore):

    0  0  8  8        road (0)        4 / 12
    0  0  8  8        vegetation (8)  4 / 12
   10 10 255 255      sky (10)        2 / 12
    2 99 255 255      building (2)    1 / 12, unmapped id 99: 1 / 12
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from geoai_vlm.segmentation import (
    CITYSCAPES_19,
    LabelMappingError,
    SemanticSegmenter,
    check_label_mapping,
    class_pixel_fractions,
    mapping_from_id2label,
)

MASK = np.array(
    [
        [0, 0, 8, 8],
        [0, 0, 8, 8],
        [10, 10, 255, 255],
        [2, 99, 255, 255],
    ]
)

CITYSCAPES_CONFIG = {
    str(i): name
    for i, name in enumerate(
        ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
         "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
         "motorcycle", "bicycle"]
    )
}
VISTAS_LIKE_CONFIG = {"0": "Bird", "1": "Ground Animal", "2": "Curb", "13": "Road", "15": "Sidewalk"}


class TestFractions:
    def test_hand_computed_fractions(self):
        row = class_pixel_fractions(MASK)
        assert row["road_pixel_fraction"] == pytest.approx(4 / 12)
        assert row["vegetation_pixel_fraction"] == pytest.approx(4 / 12)
        assert row["sky_pixel_fraction"] == pytest.approx(2 / 12)
        assert row["building_pixel_fraction"] == pytest.approx(1 / 12)
        assert row["unmapped_pixel_fraction"] == pytest.approx(1 / 12)
        assert row["labelled_pixels"] == 12
        assert row["label_mapping"] == "cityscapes-19@1"

    def test_every_class_gets_a_named_column(self):
        row = class_pixel_fractions(MASK)
        for name in CITYSCAPES_19.id_to_name.values():
            assert f"{name}_pixel_fraction" in row
        assert row["traffic_light_pixel_fraction"] == 0.0

    def test_fractions_sum_to_one(self):
        row = class_pixel_fractions(MASK)
        total = sum(v for k, v in row.items() if k.endswith("_pixel_fraction"))
        assert total == pytest.approx(1.0)

    def test_fully_ignored_mask_gives_missing_values(self):
        row = class_pixel_fractions(np.full((3, 3), 255))
        assert row["labelled_pixels"] == 0
        assert np.isnan(row["road_pixel_fraction"])

    def test_measurement_names_describe_the_measurement(self):
        names = [k for k in class_pixel_fractions(MASK) if k.endswith("_fraction")]
        assert all(k.endswith("_pixel_fraction") for k in names)
        assert not any("exposure" in k or "green_space" in k for k in names)


class TestLabelMappings:
    def test_cityscapes_model_matches(self):
        check_label_mapping(CITYSCAPES_CONFIG, CITYSCAPES_19)  # no error

    def test_vistas_ids_are_never_taken_for_cityscapes(self):
        with pytest.raises(LabelMappingError, match="do not match cityscapes-19@1"):
            check_label_mapping(VISTAS_LIKE_CONFIG, CITYSCAPES_19)

    def test_a_renamed_class_is_a_mismatch(self):
        renamed = dict(CITYSCAPES_CONFIG, **{"8": "plants"})
        with pytest.raises(LabelMappingError, match="8: model 'plants'"):
            check_label_mapping(renamed, CITYSCAPES_19)

    def test_custom_mapping_from_a_model_config(self):
        mapping = mapping_from_id2label(VISTAS_LIKE_CONFIG, name="vistas-model-x", version="2")
        assert mapping.tag == "vistas-model-x@2"
        assert mapping.id_to_name[13] == "road"
        assert mapping.id_to_name[1] == "ground_animal"
        row = class_pixel_fractions(np.array([13, 13, 15, 0]), mapping)
        assert row["road_pixel_fraction"] == pytest.approx(0.5)
        assert row["label_mapping"] == "vistas-model-x@2"

    def test_mapping_is_versioned_and_named(self):
        assert CITYSCAPES_19.name == "cityscapes-19" and CITYSCAPES_19.version == "1"
        assert CITYSCAPES_19.column(8) == "vegetation_pixel_fraction"


class _FakeProcessor:
    def __call__(self, images=None, return_tensors=None):
        return {"pixel_values": np.zeros((1, 3, 4, 4))}

    def post_process_semantic_segmentation(self, outputs, target_sizes=None):
        return [MASK]


class _FakeModel:
    def __call__(self, **inputs):
        return {"logits": None}


class TestSegmenter:
    def test_measure_with_fake_outputs(self, tmp_path):
        path = tmp_path / "street_01.png"
        Image.new("RGB", (8, 8)).save(path)
        seg = SemanticSegmenter("fake/segformer")
        seg.model, seg.processor, seg.device = _FakeModel(), _FakeProcessor(), "cpu"
        table = seg.measure([path, Image.new("RGB", (4, 4))])
        assert list(table["image_id"]) == ["street_01", "1"]
        assert table.loc[0, "vegetation_pixel_fraction"] == pytest.approx(4 / 12)
        assert set(table["segmentation_model"]) == {"fake/segformer"}
        assert set(table["label_mapping"]) == {"cityscapes-19@1"}

    def test_mismatched_model_is_rejected_on_load(self, monkeypatch):
        pytest.importorskip("torch")
        module = types.ModuleType("transformers")

        class AutoImageProcessor:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return _FakeProcessor()

        class AutoModelForSemanticSegmentation:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return types.SimpleNamespace(config=types.SimpleNamespace(id2label=VISTAS_LIKE_CONFIG))

        module.AutoImageProcessor = AutoImageProcessor
        module.AutoModelForSemanticSegmentation = AutoModelForSemanticSegmentation
        monkeypatch.setitem(sys.modules, "transformers", module)
        with pytest.raises(LabelMappingError):
            SemanticSegmenter("fake/vistas-model").load_model()

    def test_missing_dependencies_name_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match=r"geoai-vlm\[segment\]"):
            SemanticSegmenter().load_model()


@pytest.mark.slow
def test_real_segformer_on_a_synthetic_image():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from geoai_vlm.chat import make_synthetic_street_image

    table = SemanticSegmenter().measure([make_synthetic_street_image()], ids=["synthetic"])
    row = table.iloc[0]
    total = sum(v for k, v in row.items() if k.endswith("_pixel_fraction"))
    assert total == pytest.approx(1.0)
    assert row["labelled_pixels"] == 384 * 256
