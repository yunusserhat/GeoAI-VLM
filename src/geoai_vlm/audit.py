# -*- coding: utf-8 -*-
"""
Active mobility audit for GeoAI-VLM
===================================
A versioned observation protocol for street-level images
(``active_mobility_audit_v1``): for each of a fixed list of features that can
be *seen* in a photograph -- sidewalks, crossings, cycle infrastructure,
trees, seating, light poles, ramps, barriers -- the model records whether it
is present, how confident it is and the visual cue it used.

What this is and is not
-----------------------
Each value is an **image observation**: what one photograph shows from one
viewpoint at one moment. It is not a measurement of the street, of anyone's
exposure, of lighting adequacy, of safety or of health. No item is combined
into a score, and nothing here should be read as a claim that a feature
affects walking, cycling or health.

States
------
The model answers with one of four states per item:

``present``      the feature is visible
``absent``       where it would be is clearly visible, and it is not there
``not_visible``  that part of the street is out of frame, occluded or unclear
``uncertain``    something is visible but it cannot be told apart

After normalisation three more values keep failures distinguishable from
observations -- an unknown is never written as ``absent``:

``not_assessed`` the model returned nothing for the item
``invalid``      the model returned a value outside the vocabulary (rejected)
``failed``       the whole response could not be parsed
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

from .schemas import validate_json


__all__ = [
    "AUDIT_TEMPLATE_NAME",
    "AUDIT_ITEMS",
    "AUDIT_STATES",
    "NORMALISED_STATES",
    "CONFIDENCE_LEVELS",
    "AUDIT_JSON_SCHEMA",
    "AUDIT_COLUMNS",
    "audit_column",
    "validate_audit_response",
    "normalize_audit_response",
    "flatten_audit_response",
    "audit_table",
]

AUDIT_TEMPLATE_NAME = "active_mobility_audit_v1"

#: States the model may answer with.
AUDIT_STATES = ("present", "absent", "not_visible", "uncertain")
#: Every state a normalised record can carry.
NORMALISED_STATES = AUDIT_STATES + ("not_assessed", "invalid", "failed")
CONFIDENCE_LEVELS = ("low", "medium", "high")
EVIDENCE_MAX_CHARS = 200
QUALITY_ISSUES = ("blur", "occlusion", "overexposure", "underexposure", "partial_view", "night", "none")

#: The observable items, in output order. ``attribute`` is an optional
#: categorical detail recorded only when the item is present.
AUDIT_ITEMS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict(
    [
        ("sidewalk", {
            "definition": "A sidewalk or footway runs along the street in view.",
            "attribute": ("width_class", ("narrow", "medium", "wide")),
            "attribute_note": "visual estimate: narrow = room for one person, medium = two people can pass, wide = three or more",
        }),
        ("sidewalk_gap", {
            "definition": "The sidewalk visibly stops, has a gap, or is interrupted without a continuous walking surface.",
        }),
        ("sidewalk_obstruction", {
            "definition": "A non-vehicle object blocks part of the walking path on the sidewalk.",
            "attribute": ("obstruction_type", ("street_furniture", "pole_or_sign", "vegetation", "construction", "waste", "other")),
        }),
        ("pedestrian_crossing", {
            "definition": "A place to cross the road on foot is visible.",
            "attribute": ("crossing_type", ("zebra_marked", "signalized", "raised", "unmarked", "other")),
        }),
        ("intersection", {
            "definition": "An intersection or junction of streets is visible.",
        }),
        ("traffic_calming", {
            "definition": "A physical traffic calming measure is visible.",
            "attribute": ("calming_type", ("speed_bump", "bollards", "narrowing_or_chicane", "raised_table", "other")),
        }),
        ("cycle_infrastructure", {
            "definition": "Infrastructure for cycling is visible.",
            "attribute": ("cycle_type", ("separated_track", "painted_lane", "shared_lane_marking", "other")),
        }),
        ("street_trees", {
            "definition": "Trees stand along the street or sidewalk.",
        }),
        ("other_greenery", {
            "definition": "Other vegetation is visible: planters, hedges, grass verges or front gardens.",
        }),
        ("shade_on_walkway", {
            "definition": "Shade falls on the walking surface in this image (from trees, buildings or awnings).",
        }),
        ("seating", {
            "definition": "A bench or other public seating is visible.",
        }),
        ("street_light_pole", {
            "definition": "A street light pole or lamp is visible. Presence only: a daytime image cannot show whether lighting is adequate.",
        }),
        ("sidewalk_parking", {
            "definition": "A vehicle is parked on the sidewalk.",
        }),
        ("curb_ramp", {
            "definition": "A curb ramp or dropped kerb is visible at a crossing or corner.",
        }),
        ("accessibility_barrier", {
            "definition": "A barrier to wheelchair or stroller access is visible on the walking route.",
            "attribute": ("barrier_type", ("steps", "high_curb_without_ramp", "steep_slope", "narrow_passage", "uneven_surface", "other")),
        }),
        ("surface_damage", {
            "definition": "The walking surface is visibly cracked, broken, uneven or potholed.",
        }),
        ("litter", {
            "definition": "Litter, dumped waste or overflowing bins are visible.",
        }),
        ("active_frontage", {
            "definition": "Ground-floor shops, cafes, entrances or windows face the street.",
        }),
        ("view_obstruction", {
            "definition": "Something in the foreground (a vehicle, wall, vegetation or glare) blocks a large part of the view, limiting what can be assessed.",
        }),
    ]
)


def audit_column(item: str, field: str) -> str:
    """Flat column name for one item field, e.g. ``audit_sidewalk_state``."""
    return f"audit_{item}_{field}"


def _item_fields(item: str) -> List[str]:
    fields = ["state", "confidence", "evidence"]
    attribute = AUDIT_ITEMS[item].get("attribute")
    if attribute:
        fields.append(attribute[0])
    return fields


#: Flat record columns, in order.
AUDIT_COLUMNS: Tuple[str, ...] = tuple(
    [audit_column(item, f) for item in AUDIT_ITEMS for f in _item_fields(item)]
    + ["audit_template", "audit_issues"]
)


def _item_schema(item: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "state": {"type": "string", "enum": list(AUDIT_STATES)},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        "evidence": {"type": "string", "maxLength": EVIDENCE_MAX_CHARS},
    }
    attribute = AUDIT_ITEMS[item].get("attribute")
    if attribute:
        name, values = attribute
        props[name] = {"type": ["string", "null"], "enum": list(values) + [None]}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


#: JSON Schema of a response, for validation and constrained decoding.
AUDIT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "items": {
            "type": "object",
            "properties": {item: _item_schema(item) for item in AUDIT_ITEMS},
            "required": list(AUDIT_ITEMS),
            "additionalProperties": False,
        },
        "image_quality": {
            "type": "object",
            "properties": {
                "usable_for_analysis": {"type": ["boolean", "null"]},
                "issues": {"type": "array", "items": {"type": "string", "enum": list(QUALITY_ISSUES)}},
            },
            "required": ["usable_for_analysis", "issues"],
            "additionalProperties": False,
        },
    },
    "required": ["items", "image_quality"],
    "additionalProperties": False,
}


def validate_audit_response(parsed: Any) -> List[str]:
    """Strict check of a parsed response against :data:`AUDIT_JSON_SCHEMA`.

    Returns the list of problems (empty when valid). Values outside the
    vocabulary are reported here, and :func:`normalize_audit_response` never
    accepts them as observations.
    """
    return validate_json(parsed, AUDIT_JSON_SCHEMA)


def _empty_item(item: str, state: str) -> Dict[str, Any]:
    return {field: (state if field == "state" else None) for field in _item_fields(item)}


def normalize_audit_response(parsed: Any) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Turn a parsed response into one normalised entry per audit item.

    * a missing item, or an item without a state -> ``not_assessed``
    * a state outside the vocabulary -> ``invalid`` (never coerced)
    * an unparseable response (``{"error": ...}``) -> ``failed`` for every item
    * an invalid confidence or attribute -> ``None``, with an issue noted
    * an attribute given for a state other than ``present`` -> dropped
    * evidence longer than 200 characters -> truncated, with an issue noted

    Returns:
        ``(items, issues)`` -- ``items`` maps every item in
        :data:`AUDIT_ITEMS` to its fields; ``issues`` lists what was changed.
    """
    if not isinstance(parsed, Mapping) or "error" in parsed:
        return {item: _empty_item(item, "failed") for item in AUDIT_ITEMS}, ["response could not be parsed"]

    issues: List[str] = []
    raw_items = parsed.get("items")
    if not isinstance(raw_items, Mapping):
        issues.append("no 'items' object; every item not_assessed")
        raw_items = {}
    for extra in sorted(set(raw_items) - set(AUDIT_ITEMS)):
        issues.append(f"unexpected item {extra!r} ignored")

    items: Dict[str, Dict[str, Any]] = {}
    for item, spec in AUDIT_ITEMS.items():
        raw = raw_items.get(item)
        if raw is None:
            items[item] = _empty_item(item, "not_assessed")
            issues.append(f"{item}: missing -> not_assessed")
            continue
        if not isinstance(raw, Mapping):
            items[item] = _empty_item(item, "invalid")
            issues.append(f"{item}: expected an object, got {type(raw).__name__} -> invalid")
            continue

        state = raw.get("state")
        if state is None:
            entry = _empty_item(item, "not_assessed")
            issues.append(f"{item}: no state -> not_assessed")
            items[item] = entry
            continue
        if state not in AUDIT_STATES:
            entry = _empty_item(item, "invalid")
            issues.append(f"{item}: state {state!r} rejected -> invalid")
            items[item] = entry
            continue

        entry = _empty_item(item, state)
        confidence = raw.get("confidence")
        if confidence in CONFIDENCE_LEVELS:
            entry["confidence"] = confidence
        elif confidence is not None:
            issues.append(f"{item}: confidence {confidence!r} rejected")
        else:
            issues.append(f"{item}: no confidence")

        evidence = raw.get("evidence")
        if isinstance(evidence, str) and evidence.strip():
            evidence = evidence.strip()
            if len(evidence) > EVIDENCE_MAX_CHARS:
                evidence = evidence[:EVIDENCE_MAX_CHARS]
                issues.append(f"{item}: evidence truncated to {EVIDENCE_MAX_CHARS} characters")
            entry["evidence"] = evidence
        elif evidence not in (None, ""):
            issues.append(f"{item}: evidence must be text")

        attribute = spec.get("attribute")
        if attribute:
            name, values = attribute
            value = raw.get(name)
            if value is not None:
                if state != "present":
                    issues.append(f"{item}: {name} ignored because state is {state}")
                elif value in values:
                    entry[name] = value
                else:
                    issues.append(f"{item}: {name} {value!r} rejected")
        items[item] = entry
    return items, issues


def flatten_audit_response(parsed: Any) -> Dict[str, Any]:
    """Flat ``audit_*`` record columns for one parsed response.

    Used by :class:`~geoai_vlm.describer.ImageDescriber` when the
    ``active_mobility_audit_v1`` template is selected.
    """
    items, issues = normalize_audit_response(parsed)
    row: Dict[str, Any] = {}
    for item, entry in items.items():
        for field, value in entry.items():
            row[audit_column(item, field)] = value
    row["audit_template"] = AUDIT_TEMPLATE_NAME
    row["audit_issues"] = json.dumps(issues, ensure_ascii=False)
    return row


def audit_table(descriptions: pd.DataFrame, id_column: str = "image_id") -> pd.DataFrame:
    """Long table (one row per image and item) from description records.

    Reads ``parsed_json`` and re-normalises it, so it also works on records
    written before the flat ``audit_*`` columns existed.

    Returns:
        Columns ``image_id``, ``item``, ``state``, ``confidence``,
        ``evidence``, ``attribute`` and ``attribute_value``.
    """
    rows = []
    for _, record in descriptions.iterrows():
        try:
            parsed = json.loads(record["parsed_json"]) if isinstance(record.get("parsed_json"), str) else {}
        except json.JSONDecodeError:
            parsed = {"error": "unreadable parsed_json"}
        failed = record.get("parse_error")
        if failed is not None and pd.notna(failed) and bool(failed):
            parsed = {"error": "parse error"}
        items, _ = normalize_audit_response(parsed)
        for item, entry in items.items():
            attribute = AUDIT_ITEMS[item].get("attribute")
            rows.append(
                {
                    id_column: record[id_column],
                    "item": item,
                    "state": entry["state"],
                    "confidence": entry["confidence"],
                    "evidence": entry["evidence"],
                    "attribute": attribute[0] if attribute else None,
                    "attribute_value": entry.get(attribute[0]) if attribute else None,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[id_column, "item", "state", "confidence", "evidence", "attribute", "attribute_value"],
    )


# ---------------------------------------------------------------------------
# Prompt text
# ---------------------------------------------------------------------------
def _schema_example() -> str:
    lines = ["{", '  "items": {']
    names = list(AUDIT_ITEMS)
    for i, item in enumerate(names):
        attribute = AUDIT_ITEMS[item].get("attribute")
        fields = (
            '"state": "present|absent|not_visible|uncertain", '
            '"confidence": "low|medium|high", '
            '"evidence": "<short visual cue>"'
        )
        if attribute:
            fields += f', "{attribute[0]}": "{"|".join(attribute[1])}|null"'
        comma = "," if i < len(names) - 1 else ""
        lines.append(f'    "{item}": {{{fields}}}{comma}')
    lines.append("  },")
    lines.append(
        '  "image_quality": {"usable_for_analysis": true/false, '
        f'"issues": [{", ".join(json.dumps(q) for q in QUALITY_ISSUES)}]}}'
    )
    lines.append("}")
    return "\n".join(lines)


def _definitions() -> str:
    out = []
    for item, spec in AUDIT_ITEMS.items():
        line = f"- {item}: {spec['definition']}"
        attribute = spec.get("attribute")
        if attribute:
            note = spec.get("attribute_note")
            line += f" {attribute[0]}: {', '.join(attribute[1])}" + (f" ({note})" if note else "") + "."
        out.append(line)
    return "\n".join(out)


AUDIT_SYSTEM_PROMPT = f"""You audit one street-level photograph for observable features of the walking and cycling environment. Record only what is visible in this image.

For every item give:
- "state": "present" if the feature is visible; "absent" only if the part of the street where it would be is clearly visible and the feature is not there; "not_visible" if that part of the street is out of frame, occluded, too far, too dark or too blurred to see; "uncertain" if something is visible but you cannot tell whether it is the feature.
- "confidence": "low", "medium" or "high" for the state you chose.
- "evidence": a short phrase (at most 25 words) naming the visual cue you used, for example "zebra stripes across the road in the foreground". For not_visible, say what blocks the view.
- the item's extra field, when it has one, only if the state is "present"; otherwise null.

RULES:
- Describe this image, not the place. Do not use outside knowledge about the location.
- Never infer what you cannot see. If unsure, use "uncertain" or "not_visible" - never "absent".
- Do not judge lighting adequacy, safety, health or walkability. For street lights, report only whether a light pole or lamp is visible.
- Do not give scores, ratings or recommendations.
- Output only one JSON object that follows the schema exactly, with no markdown or commentary.

ITEMS:
{_definitions()}

SCHEMA:
{_schema_example()}
"""

AUDIT_USER_PROMPT = (
    "Audit this street-level image for the listed walking and cycling environment "
    "features. Return only the JSON object."
)
