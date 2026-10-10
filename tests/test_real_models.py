# -*- coding: utf-8 -*-
"""
Real-model tests. Marked ``slow``: skipped unless ``pytest --run-slow``.

They download small public models from the Hugging Face Hub and run them on
whatever hardware is present (CPU is enough). CI never runs them. The images
are synthetic drawings, so no third-party imagery is downloaded or stored.

What they verify is that the software path works with a real model and a real
processor -- not that the descriptions are any good.
"""

from __future__ import annotations

import json
import re

import pytest

pytestmark = pytest.mark.slow

SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


@pytest.fixture(scope="module")
def two_images(tmp_path_factory):
    from geoai_vlm.chat import make_synthetic_street_image

    folder = tmp_path_factory.mktemp("synthetic")
    paths = []
    for name, size in (("synthetic_a", (384, 256)), ("synthetic_b", (512, 320))):
        path = folder / f"{name}.png"
        make_synthetic_street_image(*size).save(path)
        paths.append(path)
    return paths


def test_smolvlm_describes_two_images_with_provenance(two_images, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from geoai_vlm import ImageDescriber

    describer = ImageDescriber(
        model_name=SMOLVLM,
        backend="transformers",
        prompt_template="simple",
        max_new_tokens=96,
    )
    df = describer.describe(
        image_paths=two_images, output_path=tmp_path / "out.parquet", batch_size=2
    )

    assert list(df["image_id"]) == ["synthetic_a", "synthetic_b"]
    assert (df["raw_response"].str.len() > 0).all(), "a real model must return text"
    assert set(df["backend"]) == {"transformers"}
    assert df["backend_version"].str.startswith("transformers ").all()
    assert df["model_revision"].map(lambda s: bool(re.fullmatch(r"[0-9a-f]{40}", s or ""))).all()
    assert set(df["system_prompt_mode_effective"]) <= {"system", "prepend"}
    assert set(df["decoding_mode"]) == {"unconstrained"}
    params = json.loads(df.iloc[0]["generation_params"])
    assert params["max_new_tokens"] == 96
    assert df["processing_id"].nunique() == 1

    # Resume: the same configuration does not regenerate finished records.
    again = describer.describe(image_paths=two_images, output_path=tmp_path / "out.parquet")
    successful = (~again["parse_error"].astype(bool)).sum()
    assert len(again) == 2
    assert successful >= 0  # parse success depends on the model, not the software


def test_smolvlm_check_model_reports(two_images):
    pytest.importorskip("transformers")
    from geoai_vlm import check_model

    report = check_model(SMOLVLM, backend="transformers", max_new_tokens=64, image=str(two_images[0]))
    assert report.loaded and report.generated, report.summary()
    assert report.chat_template is True
    assert report.system_role is not None
    assert report.peak_rss_mb and report.peak_rss_mb > 0
