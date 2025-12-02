# council_orchestrator/orchestrator/packets/aggregator.py
# Round aggregation and telemetry utilities

import json
import os
from pathlib import Path
from typing import Dict, Any, List

def aggregate_round_events(run_id: str, round_num: int, event_log_path: Path) -> Dict[str, Any]:
    """Aggregate events for a round to determine consensus and early exit conditions."""
    # Read recent events for this round
    round_events = []
    if event_log_path.exists():
        try:
            with open(event_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    event = json.loads(line.strip())
                    if (event.get("run_id") == run_id and
                        event.get("round") == round_num and
                        event.get("event_type") == "member_response"):
                        round_events.append(event)
        except Exception as e:
            print(f"[AGGREGATION ERROR] Failed to read round events: {e}")
            return {}

    if not round_events:
        return {}

    # Calculate round metrics
    total_members = len(round_events)
    successful_responses = [e for e in round_events if e.get("status") == "success"]
    success_rate = len(successful_responses) / total_members if total_members > 0 else 0

    # Consensus detection (simplified - can be enhanced)
    votes = [e.get("vote") for e in successful_responses if e.get("vote")]
    consensus = len(set(votes)) == 1 and len(votes) > 0

    # Novelty distribution for memory placement
    novelty_counts = {}
    for event in successful_responses:
        novelty = event.get("novelty", "medium")
        novelty_counts[novelty] = novelty_counts.get(novelty, 0) + 1

    # Early exit conditions
    early_exit = False
    exit_reason = None
    if success_rate >= 0.8 and consensus:
        early_exit = True
        exit_reason = "consensus_achieved"
    elif success_rate < 0.3:
        early_exit = True
        exit_reason = "low_success_rate"

    return {
        "round": round_num,
        "total_members": total_members,
        "success_rate": success_rate,
        "consensus": consensus,
        "novelty_distribution": novelty_counts,
        "early_exit": early_exit,
        "exit_reason": exit_reason,
        "avg_latency": sum(e.get("latency_ms", 0) for e in successful_responses) / len(successful_responses) if successful_responses else 0,
        "total_tokens_in": sum(e.get("tokens_in", 0) for e in successful_responses),
        "total_tokens_out": sum(e.get("tokens_out", 0) for e in successful_responses)
    }

def calculate_round_telemetry(packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate telemetry metrics across multiple round packets."""
    if not packets:
        return {}

    total_rounds = len(packets)
    total_cost = sum(p.get("cost", {}).get("total", 0) for p in packets)
    total_errors = sum(len(p.get("errors", [])) for p in packets)
    avg_confidence = sum(p.get("confidence", 0) for p in packets) / total_rounds

    # Engine usage distribution
    engine_usage = {}
    for packet in packets:
        engine = packet.get("engine", "unknown")
        engine_usage[engine] = engine_usage.get(engine, 0) + 1

    return {
        "total_rounds": total_rounds,
        "total_cost": total_cost,
        "total_errors": total_errors,
        "avg_confidence": avg_confidence,
        "engine_usage": engine_usage
    }