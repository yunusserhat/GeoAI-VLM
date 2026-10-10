# -*- coding: utf-8 -*-
"""
Descriptive template schema -> JSON Schema conversion and validation.

``"same options or null"`` must keep the vocabulary of the field before it,
so constrained decoding and validation reject values outside it.
"""

from __future__ import annotations

from geoai_vlm.prompts import GEOAI_JSON_SCHEMA
from geoai_vlm.schemas import descriptive_to_json_schema, validate_json


def test_same_options_keeps_the_sibling_vocabulary():
    land_use = GEOAI_JSON_SCHEMA["properties"]["land_use_character"]["properties"]
    primary, secondary = land_use["primary"], land_use["secondary"]
    assert secondary["type"] == ["string", "null"]
    assert secondary["enum"] == primary["enum"] + [None]


def test_values_outside_the_vocabulary_are_rejected():
    schema = descriptive_to_json_schema(
        {"use": {"primary": "residential|commercial", "secondary": "same options or null"}}
    )
    ok = {"use": {"primary": "residential", "secondary": "commercial"}}
    assert validate_json(ok, schema) == []
    assert validate_json({"use": {"primary": "residential", "secondary": None}}, schema) == []
    issues = validate_json({"use": {"primary": "residential", "secondary": "spaceship"}}, schema)
    assert issues and "spaceship" in issues[0]


def test_without_a_previous_vocabulary_it_stays_a_nullable_string():
    schema = descriptive_to_json_schema({"note": "same options or null"})
    assert schema["properties"]["note"] == {"type": ["string", "null"]}
