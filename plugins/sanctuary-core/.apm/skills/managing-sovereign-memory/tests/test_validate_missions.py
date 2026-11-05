"""pytest tests for validate_missions.validate_manifest."""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))  # tests live beside scripts/, not in it

import validate_missions as vm  # noqa: E402

SCHEMA = copy.deepcopy(vm.load_schema())


def _valid() -> dict:
    return {
        "id": "MISSION-004",
        "title": "Example mission",
        "status": "PROPOSED",
        "execution_surface": "local",
        "target_platform": "Apple Silicon",
        "created_at": "2026-11-04T09:00:00Z",
        "updated_at": "2026-11-05T01:14:00.610727+00:00",
        "confidence_reviewed": "2026-11-04",
        "hypothesis": "A measurable prediction.",
        "falsification_criteria": "An observable result that would refute it.",
        "scores": {
            "strategic_leverage": 4,
            "epistemic_depth": 3,
            "operational_friction": 2,
            "cost_weight": 0.05,
            "priority_score": 120.0,
        },
        "priority_score": 120.0,
        "model_routing": {
            "cheapest_viable_tier": "free-local",
            "recommended_model": "qwen2-7b-mlx-4bit",
            "cli_target": "mlx-lm",
            "cost_weight": 0.05,
            "fallback_models": ["claude-haiku-5.5"],
        },
        "verdict": None,
        "human_gate_required": False,
        "issue_ref": None,
        "tags": ["example"],
        "evidence": [{"citation": "LEARNING/experiments/x/README.md"}, "note"],
        "lessons_learned": [],
        "related_missions": {"MISSION-031": "feeds"},
        "projected_impact": {"paths": ["LEARNING/experiments/x/"]},
    }


def _errors(manifest: dict) -> list[str]:
    return vm.validate_manifest(manifest, SCHEMA)


def _assert_fails(manifest: dict, needle: str) -> list[str]:
    errors = _errors(manifest)
    assert errors, "expected validation errors"
    assert any(needle in e for e in errors), errors
    return errors


def test_valid_manifest_passes():
    assert _errors(_valid()) == []


def test_bad_id_fails():
    m = _valid()
    m["id"] = "MISSION-4"
    errors = _assert_fails(m, "id")
    assert all(e.startswith("[MISSION-4] ") for e in errors)


def test_bad_status_fails():
    m = _valid()
    m["status"] = "DONE"
    _assert_fails(m, "status")


def test_confirmed_verdict_passes_legacy_value():
    m = _valid()
    m["verdict"] = "CONFIRMED"
    assert _errors(m) == []


def test_bad_verdict_fails():
    m = _valid()
    m["verdict"] = "FINISHED"
    _assert_fails(m, "verdict")


def test_missing_required_key_fails():
    m = _valid()
    del m["hypothesis"]
    _assert_fails(m, "hypothesis")


def test_wrong_type_for_tags_fails():
    m = _valid()
    m["tags"] = "example,other"
    _assert_fails(m, "tags")


def test_bad_related_missions_key_fails():
    m = _valid()
    m["related_missions"] = {"mission-004": "feeds"}
    _assert_fails(m, "related_missions")


def test_bad_tier_fails():
    m = _valid()
    m["model_routing"]["cheapest_viable_tier"] = "premium"
    _assert_fails(m, "cheapest_viable_tier")


def test_score_out_of_range_fails():
    m = _valid()
    m["scores"]["strategic_leverage"] = 6
    _assert_fails(m, "strategic_leverage")


def test_cost_weight_above_maximum_message():
    m = _valid()
    m["scores"]["cost_weight"] = 1.5
    errors = _errors(m)
    assert "[MISSION-004] scores.cost_weight: 1.5 is greater than the maximum 1" in errors


def test_cost_weight_must_be_positive():
    m = _valid()
    m["scores"]["cost_weight"] = 0
    _assert_fails(m, "cost_weight")


def test_bool_where_integer_required_fails():
    m = _valid()
    m["scores"]["epistemic_depth"] = True
    errors = _assert_fails(m, "epistemic_depth")
    assert any("boolean" in e for e in errors), errors


def test_bad_created_at_format_fails():
    m = _valid()
    m["created_at"] = "2026-11-04 09:00:00"
    _assert_fails(m, "created_at")
