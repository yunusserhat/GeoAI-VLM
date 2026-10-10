# -*- coding: utf-8 -*-
"""
The Gradio demo (P2-H). The map helper is tested everywhere; building the
interface needs the optional ``app`` extra and is skipped without Gradio.
Nothing is launched and no network is used.
"""

from __future__ import annotations

import html

import pandas as pd
import pytest

from geoai_vlm.app import scene_map_html


class TestSceneMap:
    def test_markers_and_attribution(self):
        records = pd.DataFrame(
            {"image_id": ["a1", "b2"], "lat": [41.0, 41.001], "lon": [28.9, 28.9],
             "audit_sidewalk_state": ["present", "absent"]}
        )
        page = html.unescape(scene_map_html(records))
        assert "leaflet@1.9.4" in page, "the Leaflet version must be pinned"
        assert "OpenStreetMap contributors" in page
        assert '"lat": 41.0' in page and "sidewalk: present" in page

    def test_text_is_escaped(self):
        records = pd.DataFrame({"image_id": ["<script>alert(1)</script>"], "lat": [0.0], "lon": [0.0]})
        page = scene_map_html(records)
        assert "<script>alert(1)" not in html.unescape(page).split("var pts =")[1]

    def test_empty_records_still_render(self):
        assert "<iframe" in scene_map_html(pd.DataFrame(columns=["image_id", "lat", "lon"]))


def test_interface_builds_without_launching():
    gr = pytest.importorskip("gradio")
    from geoai_vlm.app import build_app
    from geoai_vlm.service import DemoService, SceneIndex
    from tests.conftest import MockImageEmbedder

    records = pd.DataFrame(
        {"image_id": ["a"], "lat": [41.0], "lon": [28.9], "scene_narrative": ["a lane"]}
    )
    service = DemoService(index=SceneIndex.build(records, MockImageEmbedder()))
    demo = build_app(service)
    assert isinstance(demo, gr.Blocks)
