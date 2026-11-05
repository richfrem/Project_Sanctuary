#!/usr/bin/env python3
"""Structural validator for mission manifests against mission.schema.json.

Implements the JSON Schema keywords: type, required, properties,
additionalProperties, enum, pattern, minLength, minItems, items, minimum,
maximum and description.

Two keywords outside that list are also honoured because the mission rules
cannot be expressed without them: exclusiveMinimum (cost_weight must be > 0)
and propertyNames (related_missions keys must match ^MISSION-\\d{3}$).

Semantic rules are NOT implemented here: priority-score arithmetic and
projected_impact path safety stay in manage_backlog.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA_RELPATH = os.path.join("LEARNING", "missions", "mission.schema.json")
SCHEMA_FILENAME = "mission.schema.json"


def _kind(value: Any) -> str:
    """Return the JSON Schema type name of a Python value, for messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same(a: Any, b: Any) -> bool:
    """Equality that does not treat True as 1 or 1 as 1.0."""
    return type(a) is type(b) and a == b


def _show(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _join(path: str, key: Any) -> str:
    key = str(key)
    return f"{path}.{key}" if path else key


def _walk(value: Any, schema: Any, path: str, out: list[tuple[str, str]]) -> None:
    """Validate value against schema, appending (path, message) pairs to out."""
    if not isinstance(schema, dict):
        return

    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, t) for t in allowed):
            out.append((path, f"expected {' or '.join(allowed)}, got {_kind(value)} {_show(value)}"))
            return

    if "enum" in schema and not any(_same(value, option) for option in schema["enum"]):
        out.append((path, f"{_show(value)} is not one of {_show(schema['enum'])}"))

    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append((path, f"{_show(value)} does not match pattern {schema['pattern']}"))
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append((path, f"{_show(value)} is shorter than the minimum length {schema['minLength']}"))

    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            out.append((path, f"{_show(value)} is less than the minimum {_show(schema['minimum'])}"))
        if "maximum" in schema and value > schema["maximum"]:
            out.append((path, f"{_show(value)} is greater than the maximum {_show(schema['maximum'])}"))
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            out.append((path, f"{_show(value)} is not greater than the exclusive minimum {_show(schema['exclusiveMinimum'])}"))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append((path, f"has {len(value)} item(s), fewer than the minimum {schema['minItems']}"))
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _walk(item, items, f"{path}[{index}]", out)

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                out.append((_join(path, key), "is required but missing"))

        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in value:
                _walk(value[key], sub_schema, _join(path, key), out)

        if "propertyNames" in schema:
            for key in value:
                _walk(key, schema["propertyNames"], _join(path, key), out)

        extra = [key for key in value if key not in properties]
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in extra:
                out.append((_join(path, key), "is not an allowed property"))
        elif isinstance(additional, dict):
            for key in extra:
                _walk(value[key], additional, _join(path, key), out)


def validate_manifest(manifest: dict, schema: dict) -> list[str]:
    """Validate one manifest. Errors start with '[<id>] ' and name the property path."""
    if not isinstance(manifest, dict):
        return [f"[UNKNOWN] (root): expected object, got {_kind(manifest)}"]
    mid = str(manifest.get("id", "UNKNOWN"))
    out: list[tuple[str, str]] = []
    _walk(manifest, schema, "", out)
    return [f"[{mid}] {path or '(root)'}: {message}" for path, message in out]


def _find_schema() -> str:
    """Search upward from this script for LEARNING/missions/mission.schema.json."""
    here = os.path.dirname(os.path.abspath(__file__))
    current = here
    while True:
        candidate = os.path.join(current, SCHEMA_RELPATH)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    sibling = os.path.join(here, SCHEMA_FILENAME)
    if os.path.isfile(sibling):
        return sibling
    raise FileNotFoundError(f"could not find {SCHEMA_RELPATH} above {here}")


def load_schema(path: str | None = None) -> dict:
    """Load the schema from path, or from the upward search when path is None."""
    if path is None:
        path = _find_schema()
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_missions_dir() -> str:
    """LEARNING/missions found by searching upward from this script (and from the working directory)."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for parent in [start, *start.parents]:
            cand = parent / "LEARNING" / "missions"
            if cand.is_dir():
                return str(cand)
    raise FileNotFoundError("could not find LEARNING/missions above the script or the working directory")


def _manifest_files(missions_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(missions_dir)
        if name.endswith(".json") and name != SCHEMA_FILENAME
    )


def validate_dir(missions_dir: str, schema: dict) -> list[str]:
    """Validate every *.json manifest in missions_dir (the schema file is skipped)."""
    errors: list[str] = []
    files = _manifest_files(missions_dir)
    if not files:  # never report OK when nothing was checked
        return [f"[(none)] (root): no manifests found in {missions_dir}"]
    for name in files:
        stem = os.path.splitext(name)[0]
        try:
            with open(os.path.join(missions_dir, name), encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"[{stem}] (root): invalid JSON: {exc}")
            continue
        errors.extend(validate_manifest(manifest, schema))
        if isinstance(manifest, dict) and manifest.get("id") != stem:
            errors.append(f"[{stem}] (root): filename {name} does not match id {manifest.get('id')!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate mission manifests against mission.schema.json.")
    parser.add_argument(
        "--missions-dir",
        default=None,
        help="Directory containing mission manifests (default: LEARNING/missions, found by upward search).",
    )
    parser.add_argument("--schema", default=None, help="Path to mission.schema.json (default: upward search).")
    args = parser.parse_args(argv)

    schema = load_schema(args.schema)
    missions_dir = args.missions_dir or find_missions_dir()
    errors = validate_dir(missions_dir, schema)
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"OK: {len(_manifest_files(missions_dir))} manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
