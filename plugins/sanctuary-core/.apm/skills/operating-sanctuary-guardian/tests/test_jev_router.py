"""test_jev_router.py - Unit tests for JEV Router and OpenRouter Client."""

from __future__ import annotations

import os
import sys
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GUARDIAN_SCRIPTS = os.path.abspath(os.path.join(SCRIPT_DIR, "../scripts"))
sys.path.insert(0, GUARDIAN_SCRIPTS)

from jev_router import JEVRouter, RoutingDecision
from openrouter_client import (
    calculate_cost,
    get_cumulative_spend,
    invoke_micro_tier,
    MONTHLY_HARD_CAP_USD,
)


@pytest.fixture
def router() -> JEVRouter:
    r = JEVRouter()
    r.route_task("warmup intent check")
    return r


def test_jev_router_zero_tokens(router: JEVRouter) -> None:
    decision = router.route_task("Format markdown table columns in README.md")
    assert decision.tokens_consumed == 0
    assert decision.tier == "TIER_1_MICRO"
    assert decision.complexity_class == "MECHANICAL_CHORE"


def test_jev_router_latency(router: JEVRouter) -> None:
    durations = []
    for _ in range(50):
        d = router.route_task("Fix trailing comma in JSON manifest")
        durations.append(d.latency_ms)
    sorted_durations = sorted(durations)
    median_lat = sorted_durations[len(durations) // 2]
    # Trim top/bottom 10% to eliminate OS thread preemption noise
    trimmed = sorted_durations[5:45]
    trimmed_mean = sum(trimmed) / len(trimmed)
    assert median_lat < 3.0, f"Expected <3ms median latency, got {median_lat:.3f}ms"
    assert trimmed_mean < 3.0, f"Expected <3ms trimmed mean latency, got {trimmed_mean:.3f}ms"


def test_jev_router_epistemic_firewall(router: JEVRouter) -> None:
    tasks = [
        "Derive mathematical proof that non-generative JEPAs converge to Platonic manifold",
        "Design orthogonal Procrustes alignment algorithm across embedding spaces",
        "Resolve constitutional conflict between Protocol 128 and Protocol 78",
        "Formulate Popperian research mission for anti-Asch sycophancy resistance",
    ]
    for t in tasks:
        d = router.route_task(t)
        assert d.tier == "TIER_3_FRONTIER", f"Task '{t}' failed epistemic firewall: got {d.tier}"
        assert d.complexity_class == "EPISTEMIC_ARCHITECTURAL"


def test_jev_router_routine_chores(router: JEVRouter) -> None:
    tasks = [
        ("Format markdown table in document", "MECHANICAL_CHORE", "TIER_1_MICRO"),
        ("Write pytest unit test verifying sha256 hash", "CODING_SYNTAX", "TIER_1_MICRO"),
        ("Scan commit messages and categorize feat vs fix", "GENERAL_TRIAGE", "TIER_1_MICRO"),
    ]
    for desc, expected_comp, expected_tier in tasks:
        d = router.route_task(desc)
        assert d.tier == expected_tier, f"Mismatch for '{desc}': got {d.tier}"
        assert d.complexity_class == expected_comp


def test_jev_router_offline_fallback(router: JEVRouter) -> None:
    d = router.route_task("Format markdown table", offline=True)
    assert d.tier == "TIER_2_LOCAL"
    assert d.recommended_model == "qwen2-7b-mlx-4bit"
    assert d.offline_fallback is True


def test_openrouter_prohibited_model() -> None:
    with pytest.raises(ValueError, match="Prohibited model"):
        invoke_micro_tier("hello", model="openrouter/free")


def test_calculate_cost() -> None:
    # 1000 prompt tokens and 100 completion tokens on llama-3.2-1b ($0.027 / $0.201 per 1M)
    cost = calculate_cost("meta-llama/llama-3.2-1b-instruct", 1000, 100)
    expected = (1000 / 1e6) * 0.027 + (100 / 1e6) * 0.201
    assert abs(cost - expected) < 1e-9


def test_monthly_hard_cap_constant() -> None:
    assert MONTHLY_HARD_CAP_USD == 10.00
