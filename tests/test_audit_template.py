# -*- coding: utf-8 -*-
"""
The active_mobility_audit_v1 observation template (P1-C).

Covers: every required topic has an item; the JSON schema accepts a complete
valid response and rejects invalid values; missing fields become
``not_assessed``; invalid values are rejected (``invalid``) and never coerced
into an observation; an unparseable response is ``failed``; and the template
is wired into ImageDescriber's records. No model or network is used.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from geoai_vlm.audit import (
    AUDIT_COLUMNS,
    AUDIT_ITEMS,
    AUDIT_JSON_SCHEMA,
    AUDIT_TEMPLATE_NAME,
    NORMALISED_STATES,
    audit_column,
    audit_table,
    flatten_audit_response,
    normalize_audit_response,
    validate_audit_response,
)
from geoai_vlm.describer import ImageDescriber
from geoai_vlm.prompts import get_prompt_template


def _valid_response():
    items = {}
    for item, spec in AUDIT_ITEMS.items():
        entry = {"state": "absent", "confidence": "medium", "evidence": "street clearly visible, none seen"}
        attribute = spec.get("attribute")
        if attribute:
            entry[attribute[0]] = None
        items[item] = entry
    items["sidewalk"] = {
        "state": "present", "confidence": "high", "evidence": "paved footway on the right",
        "width_class": "medium",
    }
    items["pedestrian_crossing"] = {
        "state": "present", "confidence": "high", "evidence": "zebra stripes ahead",
        "crossing_type": "zebra_marked",
    }
    items["street_trees"] = {"state": "not_visible", "confidence": "low", "evidence": "a bus blocks the left side"}
    items["seating"] = {"state": "uncertain", "confidence": "low", "evidence": "a low object that may be a bench"}
    return {"items": items, "image_quality": {"usable_for_analysis": True, "issues": ["none"]}}


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------
REQUIRED_TOPICS = {
    "sidewalk presence": ["sidewalk"],
    "sidewalk continuity": ["sidewalk_gap"],
    "sidewalk obstructions": ["sidewalk_obstruction"],
    "visual width class": ["sidewalk"],
    "pedestrian crossing and type": ["pedestrian_crossing"],
    "intersection visibility": ["intersection"],
    "traffic calming": ["traffic_calming"],
    "cycle infrastructure": ["cycle_infrastructure"],
    "street trees, greenery and shade": ["street_trees", "other_greenery", "shade_on_walkway"],
    "seating": ["seating"],
    "street light pole presence": ["street_light_pole"],
    "parking on the sidewalk": ["sidewalk_parking"],
    "ramps and accessibility barriers": ["curb_ramp", "accessibility_barrier"],
    "maintenance and litter": ["surface_damage", "litter"],
    "active ground-floor frontages": ["active_frontage"],
    "view obstructions": ["view_obstruction"],
}


class TestProtocol:
    @pytest.mark.parametrize("topic", sorted(REQUIRED_TOPICS))
    def test_every_required_topic_has_an_item(self, topic):
        for item in REQUIRED_TOPICS[topic]:
            assert item in AUDIT_ITEMS, f"{topic}: missing item {item}"

    def test_width_class_and_crossing_type_are_attributes(self):
        assert AUDIT_ITEMS["sidewalk"]["attribute"][0] == "width_class"
        assert AUDIT_ITEMS["pedestrian_crossing"]["attribute"][0] == "crossing_type"
        calming = AUDIT_ITEMS["traffic_calming"]["attribute"][1]
        assert {"speed_bump", "bollards", "narrowing_or_chicane"} <= set(calming)

    def test_template_is_registered_and_versioned(self):
        template = get_prompt_template("active_mobility_audit_v1")
        assert AUDIT_TEMPLATE_NAME.endswith("_v1")
        assert template["json_schema"] is AUDIT_JSON_SCHEMA
        assert template["version"] == "v1"
        assert callable(template["flatten"])

    def test_lighting_is_presence_only(self):
        definition = AUDIT_ITEMS["street_light_pole"]["definition"].lower()
        assert "presence only" in definition
        assert "adequa" not in json.dumps(AUDIT_JSON_SCHEMA).lower()

    def test_no_score_or_composite_field_exists(self):
        keys = json.dumps(AUDIT_JSON_SCHEMA).lower()
        for banned in ("score", "walkability", "health", "safety", "rating", "index"):
            assert banned not in keys, f"schema must not contain a {banned!r} field"

    def test_prompt_asks_for_observation_only(self):
        system = get_prompt_template("active_mobility_audit_v1")["system"]
        assert 'never "absent"' in system
        assert "Do not give scores, ratings or recommendations" in system
        assert "Do not judge lighting adequacy, safety, health or walkability" in system

    def test_state_vocabulary(self):
        assert NORMALISED_STATES == (
            "present", "absent", "not_visible", "uncertain", "not_assessed", "invalid", "failed",
        )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
class TestSchemaValidation:
    def test_complete_valid_response_passes(self):
        assert validate_audit_response(_valid_response()) == []

    @pytest.mark.parametrize(
        "mutate,needle",
        [
            (lambda r: r["items"]["sidewalk"].update(state="yes"), "'yes' is not one of"),
            (lambda r: r["items"]["sidewalk"].update(confidence="very high"), "'very high'"),
            (lambda r: r["items"]["sidewalk"].update(width_class="huge"), "'huge'"),
            (lambda r: r["items"]["litter"].update(evidence="x" * 201), "longer than 200"),
            (lambda r: r["items"].pop("seating"), "missing required field 'seating'"),
            (lambda r: r["items"]["seating"].pop("confidence"), "missing required field 'confidence'"),
            (lambda r: r["items"].update(walkability_score={"state": "present"}), "unexpected field"),
            (lambda r: r["image_quality"].update(usable_for_analysis="yes"), "expected boolean/null"),
        ],
        ids=["state", "confidence", "attribute", "evidence-length", "missing-item",
             "missing-field", "extra-item", "quality-type"],
    )
    def test_invalid_values_are_rejected(self, mutate, needle):
        response = _valid_response()
        mutate(response)
        problems = validate_audit_response(response)
        assert problems, "an invalid response must not validate"
        assert any(needle in p for p in problems), problems

    def test_non_object_is_rejected(self):
        assert validate_audit_response([]) != []


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class TestNormalisation:
    def test_valid_observations_pass_through(self):
        items, issues = normalize_audit_response(_valid_response())
        assert items["sidewalk"] == {
            "state": "present", "confidence": "high",
            "evidence": "paved footway on the right", "width_class": "medium",
        }
        assert items["street_trees"]["state"] == "not_visible"
        assert items["seating"]["state"] == "uncertain"
        assert issues == []

    def test_missing_item_is_not_assessed_not_absent(self):
        response = _valid_response()
        del response["items"]["curb_ramp"]
        items, issues = normalize_audit_response(response)
        assert items["curb_ramp"] == {"state": "not_assessed", "confidence": None, "evidence": None}
        assert "curb_ramp: missing -> not_assessed" in issues

    def test_missing_state_is_not_assessed(self):
        response = _valid_response()
        response["items"]["litter"] = {"confidence": "high", "evidence": "x"}
        items, _ = normalize_audit_response(response)
        assert items["litter"]["state"] == "not_assessed"
        assert items["litter"]["confidence"] is None

    def test_no_items_object_marks_everything_not_assessed(self):
        items, issues = normalize_audit_response({"image_quality": {}})
        assert {e["state"] for e in items.values()} == {"not_assessed"}
        assert issues[0].startswith("no 'items' object")

    @pytest.mark.parametrize("bad", ["yes", "no", 0, 1, "ABSENT", "maybe", True])
    def test_invalid_state_is_rejected_never_coerced(self, bad):
        response = _valid_response()
        response["items"]["seating"]["state"] = bad
        items, issues = normalize_audit_response(response)
        assert items["seating"]["state"] == "invalid"
        assert items["seating"]["confidence"] is None and items["seating"]["evidence"] is None
        assert any("seating: state" in i and "rejected" in i for i in issues)

    def test_invalid_confidence_is_dropped_not_guessed(self):
        response = _valid_response()
        response["items"]["sidewalk"]["confidence"] = "certain"
        items, issues = normalize_audit_response(response)
        assert items["sidewalk"]["state"] == "present"
        assert items["sidewalk"]["confidence"] is None
        assert any("confidence 'certain' rejected" in i for i in issues)

    def test_invalid_attribute_is_dropped(self):
        response = _valid_response()
        response["items"]["pedestrian_crossing"]["crossing_type"] = "pelican"
        items, issues = normalize_audit_response(response)
        assert items["pedestrian_crossing"]["crossing_type"] is None
        assert any("crossing_type 'pelican' rejected" in i for i in issues)

    def test_attribute_is_ignored_unless_present(self):
        response = _valid_response()
        response["items"]["traffic_calming"] = {
            "state": "absent", "confidence": "high", "evidence": "none", "calming_type": "speed_bump",
        }
        items, issues = normalize_audit_response(response)
        assert items["traffic_calming"]["calming_type"] is None
        assert any("ignored because state is absent" in i for i in issues)

    def test_long_evidence_is_truncated_and_noted(self):
        response = _valid_response()
        response["items"]["litter"]["evidence"] = "y" * 500
        items, issues = normalize_audit_response(response)
        assert len(items["litter"]["evidence"]) == 200
        assert any("truncated" in i for i in issues)

    def test_unknown_items_are_ignored_and_noted(self):
        response = _valid_response()
        response["items"]["walkability_score"] = {"state": "present"}
        items, issues = normalize_audit_response(response)
        assert "walkability_score" not in items
        assert "unexpected item 'walkability_score' ignored" in issues

    @pytest.mark.parametrize("parsed", [{"error": "Failed to parse JSON"}, None, "text", []])
    def test_unparseable_response_is_failed(self, parsed):
        items, issues = normalize_audit_response(parsed)
        assert {e["state"] for e in items.values()} == {"failed"}
        assert issues == ["response could not be parsed"]

    def test_flattened_columns_are_complete_and_ordered(self):
        row = flatten_audit_response(_valid_response())
        assert tuple(row) == AUDIT_COLUMNS
        assert row[audit_column("sidewalk", "width_class")] == "medium"
        assert row["audit_template"] == "active_mobility_audit_v1"
        assert json.loads(row["audit_issues"]) == []


# ---------------------------------------------------------------------------
# Wired into ImageDescriber
# ---------------------------------------------------------------------------
class _Canned:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, image_paths, system_prompt, user_prompt):
        assert "walking and cycling environment" in user_prompt
        return [self.responses.pop(0) for _ in image_paths]


def _paths(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), "gray").save(p)
        out.append(p)
    return out


class TestDescriberIntegration:
    def test_records_get_flat_audit_columns(self, tmp_path):
        partial = _valid_response()
        del partial["items"]["intersection"]
        partial["items"]["litter"]["state"] = "maybe"
        d = ImageDescriber(model_name="org/model", prompt_template="active_mobility_audit_v1")
        d._backend = _Canned([json.dumps(_valid_response()), json.dumps(partial), "not json"])
        df = d.describe(image_paths=_paths(tmp_path, 3), batch_size=3)

        for column in AUDIT_COLUMNS:
            assert column in df.columns, column
        first, second, third = (df.iloc[i] for i in range(3))
        assert first[audit_column("sidewalk", "state")] == "present"
        assert first[audit_column("street_trees", "state")] == "not_visible"
        assert second[audit_column("intersection", "state")] == "not_assessed"
        assert second[audit_column("litter", "state")] == "invalid"
        assert third[audit_column("sidewalk", "state")] == "failed"
        assert bool(third["parse_error"]) is True

    def test_schema_issues_are_recorded(self, tmp_path):
        partial = _valid_response()
        partial["items"]["litter"]["state"] = "maybe"
        d = ImageDescriber(model_name="org/model", prompt_template="active_mobility_audit_v1")
        d._backend = _Canned([json.dumps(partial)])
        row = d.describe(image_paths=_paths(tmp_path, 1)).iloc[0]
        assert any("'maybe'" in i for i in json.loads(row["validation_issues"]))
        assert row["quality_status"] == "reported" and bool(row["usable"]) is True

    def test_template_schema_drives_structured_output(self):
        d = ImageDescriber(
            model_name="org/model", prompt_template="active_mobility_audit_v1", structured_output=True
        )
        assert d.json_schema is AUDIT_JSON_SCHEMA

    def test_audit_table_is_long_and_complete(self, tmp_path):
        d = ImageDescriber(model_name="org/model", prompt_template="active_mobility_audit_v1")
        d._backend = _Canned([json.dumps(_valid_response()), "not json"])
        df = d.describe(image_paths=_paths(tmp_path, 2), batch_size=2)
        long = audit_table(df)
        assert len(long) == 2 * len(AUDIT_ITEMS)
        assert set(long["state"]) <= set(NORMALISED_STATES)
        first = long[long["image_id"] == "img0"].set_index("item")
        assert first.loc["sidewalk", "attribute_value"] == "medium"
        assert set(long[long["image_id"] == "img1"]["state"]) == {"failed"}
