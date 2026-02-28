# -*- coding: utf-8 -*-
"""Tests for geoai_vlm.preparation module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


class TestParseVlmDescriptions:
    """Verify JSON parsing from VLM output stored in a DataFrame column."""

    @pytest.fixture
    def desc_df(self):
        """Create a DataFrame with JSON-encoded VLM descriptions."""
        records = []
        for i in range(5):
            data = {
                "scene_narrative": f"A street scene number {i}.",
                "semantic_tags": "urban, green" if i % 2 == 0 else "historic, modern",
                "land_use_primary": "residential" if i < 3 else "commercial",
            }
            records.append({"image_id": f"img_{i:04d}", "vlm_description": json.dumps(data)})
        return pd.DataFrame(records)

    def test_returns_dataframe(self, desc_df):
        from geoai_vlm.preparation import parse_vlm_descriptions

        df = parse_vlm_descriptions(desc_df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "parsed_description" in df.columns

    def test_extracts_target_field(self, desc_df):
        from geoai_vlm.preparation import parse_vlm_descriptions

        df = parse_vlm_descriptions(desc_df, target_field="scene_narrative")
        assert "parsed_description" in df.columns
        assert "street scene" in df["parsed_description"].iloc[0]

    def test_handles_missing_field_gracefully(self, desc_df):
        from geoai_vlm.preparation import parse_vlm_descriptions

        # request a non-existent field — function falls back to alt_detailed, scene_narrative, or concat
        df = parse_vlm_descriptions(desc_df, target_field="nonexistent")
        assert len(df) == 5
        assert "parsed_description" in df.columns


class TestExtractImageId:
    def test_basic_extraction(self):
        from geoai_vlm.preparation import extract_image_id

        series = pd.Series(["/data/images/0001.jpg", "C:\\photos\\123.png"])
        result = extract_image_id(series)
        assert result.iloc[0] == "0001"
        assert result.iloc[1] == "123"

    def test_custom_pattern(self):
        from geoai_vlm.preparation import extract_image_id

        series = pd.Series(["result_42_desc.json"])
        result = extract_image_id(series, pattern=r"(\d+)")
        assert result.iloc[0] == "42"


class TestMergeDataSources:
    def test_merge_two_dataframes(self):
        from geoai_vlm.preparation import merge_data_sources

        df1 = pd.DataFrame({"image_id": ["1", "2", "3"], "a": ["x", "y", "z"]})
        df2 = pd.DataFrame({"image_id": ["1", "2", "3"], "b": [10, 20, 30]})
        merged = merge_data_sources(df1, df2, merge_on="image_id")
        assert "a" in merged.columns
        assert "b" in merged.columns
        assert len(merged) == 3

    def test_merge_three_dataframes(self):
        from geoai_vlm.preparation import merge_data_sources

        df1 = pd.DataFrame({"image_id": ["1", "2"], "a": [1, 2]})
        df2 = pd.DataFrame({"image_id": ["1", "2"], "b": [3, 4]})
        df3 = pd.DataFrame({"image_id": ["1", "2"], "c": [5, 6]})
        merged = merge_data_sources(df1, df2, classifiers=df3, merge_on="image_id")
        assert "a" in merged.columns
        assert "b" in merged.columns
        assert "c" in merged.columns


class TestBuildEmbeddingText:
    def test_concatenation(self):
        from geoai_vlm.preparation import build_embedding_text

        df = pd.DataFrame(
            {"col_a": ["hello", "foo"], "col_b": ["world", "bar"]}
        )
        result = build_embedding_text(df, columns=["col_a", "col_b"], separator=" | ")
        assert result.iloc[0] == "hello | world"
        assert result.iloc[1] == "foo | bar"

    def test_single_column(self):
        from geoai_vlm.preparation import build_embedding_text

        df = pd.DataFrame({"text": ["alpha", "beta"]})
        result = build_embedding_text(df, columns=["text"])
        assert result.iloc[0] == "alpha"
