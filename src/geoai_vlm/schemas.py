# -*- coding: utf-8 -*-
"""
JSON schemas for GeoAI-VLM prompt templates
===========================================
Machine-readable schemas for the structured responses the prompt templates
ask for, plus a small validator for the subset of JSON Schema they use.

The schemas serve two purposes:

* constrained decoding -- a backend that supports it (vLLM structured
  outputs, an OpenAI-compatible ``response_format``) can force the model to
  emit JSON of this shape;
* validation -- a response is checked against the same schema after parsing,
  whatever the decoding mode was.

Constrained decoding forces *some* value into every required field. So that it
never forces a guess, every closed vocabulary converted here also admits
``"unknown"``, matching the prompts' own rule ("if unclear, use unknown").
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional


__all__ = [
    "descriptive_to_json_schema",
    "validate_json",
    "SIMPLE_JSON_SCHEMA",
]

_ENUM_RE = re.compile(r"^[a-z0-9_]+(\|[a-z0-9_]+)+$")


def _enum_values(text: str) -> Optional[List[str]]:
    candidate = text.strip()
    if _ENUM_RE.match(candidate):
        return candidate.split("|")
    return None


def _convert(value: Any, add_unknown: bool) -> Dict[str, Any]:
    if isinstance(value, dict):
        props: Dict[str, Any] = {}
        options: Optional[List[Any]] = None  # vocabulary of the previous enum field
        for key, item in value.items():
            if isinstance(item, str) and item.lower().startswith("same options or null") and options:
                props[key] = {"type": ["string", "null"], "enum": options + [None]}
                continue
            props[key] = _convert(item, add_unknown)
            if props[key].get("type") == "string" and "enum" in props[key]:
                options = list(props[key]["enum"])
        return {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        items = [str(v) for v in value]
        if add_unknown and "unknown" not in items:
            items = items + ["unknown"]
        return {"type": "array", "items": {"type": "string", "enum": items}}
    text = str(value)
    lowered = text.lower()
    if lowered == "boolean":
        # null is the boolean form of "unknown"; summary extraction already
        # treats a non-boolean verdict as unknown rather than as False.
        return {"type": ["boolean", "null"]} if add_unknown else {"type": "boolean"}
    if lowered.startswith("same options or null"):
        return {"type": ["string", "null"]}
    if lowered.startswith("list of"):
        return {"type": "array", "items": {"type": "string"}}
    enum = _enum_values(text)
    if enum is not None:
        if add_unknown and "unknown" not in enum:
            enum = enum + ["unknown"]
        return {"type": "string", "enum": enum}
    return {"type": "string"}


def descriptive_to_json_schema(
    descriptive: Mapping[str, Any],
    add_unknown: bool = True,
) -> Dict[str, Any]:
    """Convert a descriptive template schema (``GEOAI_SCHEMA`` style) to JSON Schema.

    ``"a|b|c"`` becomes a string enum, ``"boolean"`` a boolean, a list of
    options an array of that enum, ``"same options or null"`` a nullable
    string, any other string a free string. With *add_unknown* every enum also
    admits ``"unknown"`` and every boolean also admits ``null``.
    """
    schema = _convert(dict(descriptive), add_unknown)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


SIMPLE_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "tags"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def validate_json(instance: Any, schema: Mapping[str, Any], path: str = "$") -> List[str]:
    """Validate *instance* against the JSON Schema subset used by the templates.

    Supported keywords: ``type`` (single or list), ``enum``, ``properties``,
    ``required``, ``additionalProperties`` (boolean), ``items``,
    ``minItems``, ``maxItems``, ``minLength`` and ``maxLength``. Anything
    else is ignored.

    Returns:
        A list of human-readable problems; empty when the instance is valid.
    """
    errors: List[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS.get(t, lambda v: True)(instance) for t in types):
            errors.append(f"{path}: expected {'/'.join(types)}, got {type(instance).__name__}")
            return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {list(schema['enum'])}")

    if isinstance(instance, str):
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required field {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{path}: unexpected field {key!r}")
        for key, sub in props.items():
            if key in instance:
                errors.extend(validate_json(instance[key], sub, f"{path}.{key}"))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for i, item in enumerate(instance):
                errors.extend(validate_json(item, item_schema, f"{path}[{i}]"))

    return errors
