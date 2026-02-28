# -*- coding: utf-8 -*-
"""
Data Preparation Module for GeoAI-VLM
=======================================
Utilities for parsing VLM description JSON, merging heterogeneous data
sources, and constructing composite embedding text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd


__all__ = [
    "parse_vlm_descriptions",
    "merge_data_sources",
    "extract_image_id",
    "build_embedding_text",
]


def parse_vlm_descriptions(
    df: pd.DataFrame,
    description_column: str = "vlm_description",
    target_field: Optional[str] = "alt_detailed",
) -> pd.DataFrame:
    """
    Parse JSON-encoded VLM descriptions and extract a human-readable text field.

    The function attempts to decode each cell in *description_column* as JSON.
    When *target_field* is given, it extracts that key from the parsed dict;
    otherwise the whole parsed dict is stored.

    Args:
        df: Input DataFrame.
        description_column: Column with raw JSON strings.
        target_field: JSON key to extract (e.g. ``"alt_detailed"``).
            Set to ``None`` to keep the full parsed dictionary.

    Returns:
        Copy of *df* with an additional ``parsed_description`` column.
    """
    df = df.copy()

    def _parse(text):
        if pd.isna(text) or str(text).strip() == "":
            return None
        try:
            obj = json.loads(str(text))
        except json.JSONDecodeError:
            return str(text)

        if target_field is not None:
            if isinstance(obj, dict):
                # Try the exact key first, then common fallbacks
                for key in (target_field, "alt_detailed", "scene_narrative"):
                    if key in obj:
                        return obj[key]
                # Last resort: concatenate all string values
                parts = [str(v) for v in obj.values() if isinstance(v, str)]
                return " ".join(parts) if parts else str(obj)
            return str(obj)
        return obj

    df["parsed_description"] = df[description_column].apply(_parse)
    return df


def extract_image_id(
    series: pd.Series,
    pattern: str = r"(\d+)\.\w+$",
) -> pd.Series:
    """
    Extract image IDs from file-path strings using a regex pattern.

    The default pattern captures the numeric stem of paths like
    ``/data/images/123456.jpg`` → ``123456``.

    Args:
        series: Series of file-path strings.
        pattern: Regex with one capture group for the ID.

    Returns:
        Series of extracted ID strings.
    """
    return series.astype(str).str.extract(pattern, expand=False)


def merge_data_sources(
    metadata: pd.DataFrame,
    descriptions: pd.DataFrame,
    classifiers: Optional[pd.DataFrame] = None,
    perceptions: Optional[pd.DataFrame] = None,
    merge_on: str = "image_id",
) -> pd.DataFrame:
    """
    Merge multiple data sources on a shared ID column.

    Generalises the manual merging steps from ad-hoc analysis scripts into
    a single parameterised function.

    Args:
        metadata: Primary metadata DataFrame (left frame).
        descriptions: VLM descriptions DataFrame.
        classifiers: Optional classifier predictions DataFrame.
        perceptions: Optional perception scores DataFrame.
        merge_on: Column name to join on.

    Returns:
        Unified DataFrame (left-joined on *metadata*).
    """
    # Ensure string IDs
    metadata = metadata.copy()
    metadata[merge_on] = metadata[merge_on].astype(str)

    descriptions = descriptions.copy()
    descriptions[merge_on] = descriptions[merge_on].astype(str)

    # Columns to bring in (exclude duplicates except merge key)
    desc_cols = [
        c for c in descriptions.columns
        if c not in metadata.columns or c == merge_on
    ]
    result = metadata.merge(descriptions[desc_cols], on=merge_on, how="left")

    if classifiers is not None:
        classifiers = classifiers.copy()
        classifiers[merge_on] = classifiers[merge_on].astype(str)
        cls_cols = [
            c for c in classifiers.columns
            if c not in result.columns or c == merge_on
        ]
        result = result.merge(classifiers[cls_cols], on=merge_on, how="left")

    if perceptions is not None:
        perceptions = perceptions.copy()
        perceptions[merge_on] = perceptions[merge_on].astype(str)
        per_cols = [
            c for c in perceptions.columns
            if c not in result.columns or c == merge_on
        ]
        result = result.merge(perceptions[per_cols], on=merge_on, how="left")

    return result


def build_embedding_text(
    df: pd.DataFrame,
    columns: List[str],
    separator: str = " | ",
) -> pd.Series:
    """
    Construct composite text for embedding from multiple DataFrame columns.

    List-valued cells are joined with commas; ``NaN`` values are skipped.

    Args:
        df: Input DataFrame.
        columns: Column names to concatenate.
        separator: String inserted between column values.

    Returns:
        ``pd.Series`` of concatenated text strings.
    """

    def _join(row):
        parts: List[str] = []
        for col in columns:
            val = row.get(col, "")
            if pd.isna(val):
                continue
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            val = str(val).strip()
            if val:
                parts.append(val)
        return separator.join(parts)

    return df.apply(_join, axis=1)
