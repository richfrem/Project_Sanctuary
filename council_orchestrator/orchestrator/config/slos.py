# council_orchestrator/orchestrator/config/slos.py
# Service Level Objectives for Phase 2 Council Orchestrator

from typing import Dict, Any

# Phase 2 SLOs (Service Level Objectives)
PHASE2_SLOS = {
    # Round-level SLOs
    "round_p95_latency_ms": 300,  # 95th percentile round latency <= 300ms
    "round_p99_latency_ms": 500,  # 99th percentile round latency <= 500ms

    # Stage-level SLOs
    "plan_stage_p95_ms": 50,      # Query planning <= 50ms p95
    "retrieve_stage_p95_ms": 150, # Parent-doc retrieval <= 150ms p95
    "analyze_stage_p95_ms": 100,  # Novelty/conflict analysis <= 100ms p95
    "emit_stage_p95_ms": 20,      # Packet emission <= 20ms p95

    # Quality SLOs
    "evidence_hit_rate_min": 0.85,  # >= 85% of queries find relevant evidence
    "novelty_precision_min": 0.90,   # >= 90% precision on novelty detection
    "citation_overlap_min": 0.95,    # >= 95% citations have token overlap

    # Reliability SLOs
    "round_success_rate_min": 0.99,  # >= 99% rounds complete successfully
    "timeout_rate_max": 0.01,        # <= 1% rounds timeout
}

def validate_round_slo(latency_ms: int, stage_timings: Dict[str, int]) -> Dict[str, Any]:
    """
    Validate a round against SLOs.
    Returns dict with slo_status and violations.
    """
    violations = []

    # Round-level latency
    if latency_ms > PHASE2_SLOS["round_p95_latency_ms"]:
        violations.append(f"round_latency_{latency_ms}ms > {PHASE2_SLOS['round_p95_latency_ms']}ms")

    # Stage-level latencies
    stage_slos = {
        "plan_latency_ms": "plan_stage_p95_ms",
        "retrieval_latency_ms": "retrieve_stage_p95_ms",
        "analyze_latency_ms": "analyze_stage_p95_ms",
        "emit_latency_ms": "emit_stage_p95_ms"
    }

    for stage_key, slo_key in stage_slos.items():
        if stage_key in stage_timings and stage_timings[stage_key] > PHASE2_SLOS[slo_key]:
            violations.append(f"{stage_key}_{stage_timings[stage_key]}ms > {PHASE2_SLOS[slo_key]}ms")

    return {
        "slo_status": "pass" if not violations else "fail",
        "violations": violations,
        "total_latency_ms": latency_ms,
        "stage_timings": stage_timings
    }