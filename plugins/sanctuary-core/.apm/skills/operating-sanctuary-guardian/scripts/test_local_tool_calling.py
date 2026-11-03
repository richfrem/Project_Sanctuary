#!/usr/bin/env python3
"""test_local_tool_calling.py - Verification & Benchmark Suite for Local Tool Use & Structured JSON.
Purpose: Pytest suite and empirical benchmark evaluating offline Qwen2.5 tool calling and schema adherence.
Layer: sanctuary-core / operating-sanctuary-guardian (Test Automation & Verification).
Key Inputs: Local MLX server endpoints, tool definitions, JSON schemas, test prompts.
Key Outputs: Pytest assertions, adherence scores, hallucination rate, and empirical benchmarks.
Usage:
    pytest test_local_tool_calling.py -v
    python3 test_local_tool_calling.py --benchmark
Key Functions: test_parse_tools(), test_validate_tools(), test_live_tool_calling(), run_benchmark(), main().
"""

import os
import sys
import json
import time
import pytest
from pathlib import Path

# Add script directory to sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import forge_llm


def test_parse_tool_calls_from_text_tools_tag():
    """Verify parser extracts JSON inside <tools> tags."""
    sample = "<tools>\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Seattle, WA\"}}\n</tools>"
    calls = forge_llm.parse_tool_calls_from_text(sample)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"location": "Seattle, WA"}


def test_parse_tool_calls_from_text_xml_fenced():
    """Verify parser handles ```xml <tools> blocks emitted by Qwen."""
    sample = "```xml\n<tools>\n  {\"name\": \"search_codebase\", \"arguments\": {\"query\": \"reap\", \"extension\": \"py\"}}\n</tools>\n```"
    calls = forge_llm.parse_tool_calls_from_text(sample)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_codebase"
    assert calls[0]["arguments"] == {"query": "reap", "extension": "py"}


def test_parse_tool_calls_from_text_tool_call_tag():
    """Verify parser handles <tool_call> tags from alternative ChatML templates."""
    sample = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/test.txt\"}}\n</tool_call>"
    calls = forge_llm.parse_tool_calls_from_text(sample)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"] == {"path": "/tmp/test.txt"}


def test_parse_tool_calls_from_text_function_tag():
    """Verify parser handles Qwen XML function tag syntax."""
    sample = "```xml\n<function name=\"search_codebase\" arguments='{\"query\": \"reap_if_idle\", \"extension\": \"py\"}'/>\n```"
    calls = forge_llm.parse_tool_calls_from_text(sample)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_codebase"
    assert calls[0]["arguments"] == {"query": "reap_if_idle", "extension": "py"}


def test_parse_tool_calls_from_text_function_call_tag():
    """Verify parser handles <function_call> tag syntax."""
    sample = "```xml\n<function_call>\n  {\"name\": \"lookup_entity\", \"arguments\": {\"entity_id\": \"MISSION-019\", \"tier\": 2}}\n</function_call>\n```"
    calls = forge_llm.parse_tool_calls_from_text(sample)
    assert len(calls) == 1
    assert calls[0]["name"] == "lookup_entity"
    assert calls[0]["arguments"] == {"entity_id": "MISSION-019", "tier": 2}


def test_validate_tool_calls_success():
    """Verify schema validation passes for valid parameters."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    calls = [{"name": "get_weather", "arguments": {"location": "Victoria, BC", "unit": "celsius"}}]
    all_valid, validated = forge_llm.validate_tool_calls(calls, tools)
    assert all_valid is True
    assert validated[0]["valid"] is True
    assert validated[0]["error"] is None


def test_validate_tool_calls_missing_required():
    """Verify schema validation fails when required parameter is missing."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }
    ]
    calls = [{"name": "read_file", "arguments": {}}]
    all_valid, validated = forge_llm.validate_tool_calls(calls, tools)
    assert all_valid is False
    assert validated[0]["valid"] is False
    assert "required" in validated[0]["error"].lower() or "missing" in validated[0]["error"].lower()


def test_validate_tool_calls_unknown_tool():
    """Verify unknown tool call is flagged as invalid."""
    tools = [
        {
            "type": "function",
            "function": {"name": "read_file", "parameters": {"type": "object"}}
        }
    ]
    calls = [{"name": "malicious_exec", "arguments": {"cmd": "rm -rf"}}]
    all_valid, validated = forge_llm.validate_tool_calls(calls, tools)
    assert all_valid is False
    assert validated[0]["valid"] is False
    assert "not in the provided tool definitions" in validated[0]["error"]


def test_parse_and_validate_json_schema_fenced():
    """Verify parser strips markdown code blocks and validates schema."""
    text = "```json\n{\n  \"task_id\": \"128\",\n  \"status\": \"RESOLVED\"\n}\n```"
    schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "status": {"type": "string", "enum": ["PROPOSED", "IN_PROGRESS", "RESOLVED"]}
        },
        "required": ["task_id", "status"]
    }
    is_valid, data, err = forge_llm.parse_and_validate_json_schema(text, schema)
    assert is_valid is True
    assert data["task_id"] == "128"
    assert data["status"] == "RESOLVED"
    assert err is None


@pytest.mark.skipif(not forge_llm.is_mlx_server_active(), reason="MLX server is not active on port 8080")
def test_live_tool_calling_weather():
    """Live integration test: tool selection and schema compliance."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }
    ]
    res = forge_llm.call_tool("What is the weather in Victoria, BC?", tools=tools)
    assert res.get("status") == "success"
    assert res.get("tool_selected") is True
    assert res.get("schema_valid") is True
    assert len(res.get("tool_calls")) >= 1
    assert res["tool_calls"][0]["name"] == "get_weather"
    assert "Victoria" in res["tool_calls"][0]["arguments"]["location"]


@pytest.mark.skipif(not forge_llm.is_mlx_server_active(), reason="MLX server is not active on port 8080")
def test_live_multi_tool_selection():
    """Live integration test: select correct tool among multiple candidates."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": "Search codebase for keyword or pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "extension": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    res = forge_llm.call_tool("Search for reap_if_idle in python files", tools=tools)
    assert res.get("status") == "success"
    assert res.get("tool_selected") is True
    assert res.get("schema_valid") is True
    assert res["tool_calls"][0]["name"] == "search_codebase"
    assert "reap_if_idle" in res["tool_calls"][0]["arguments"]["query"]


@pytest.mark.skipif(not forge_llm.is_mlx_server_active(), reason="MLX server is not active on port 8080")
def test_live_structured_json_extraction():
    """Live integration test: strict JSON extraction conforming to schema."""
    schema = {
        "type": "object",
        "properties": {
            "mission_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["CONFIRMED", "FALSIFIED", "INCONCLUSIVE"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["mission_id", "verdict", "confidence"]
    }
    prompt = "Mission MISSION-021 has been CONFIRMED with 0.95 confidence score."
    res = forge_llm.extract_structured(prompt, schema=schema)
    assert res.get("status") == "success"
    assert res.get("schema_valid") is True
    assert res["data"]["mission_id"] == "MISSION-021"
    assert res["data"]["verdict"] == "CONFIRMED"
    assert res["data"]["confidence"] >= 0.9


@pytest.mark.skipif(not forge_llm.is_mlx_server_active(), reason="MLX server is not active on port 8080")
def test_live_conversational_no_tool_hallucination():
    """Live integration test: verify model does not hallucinate tool calls on conversational input."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "delete_database",
                "description": "Drop table from database",
                "parameters": {
                    "type": "object",
                    "properties": {"table": {"type": "string"}},
                    "required": ["table"]
                }
            }
        }
    ]
    res = forge_llm.call_tool("Hello, what is your name?", tools=tools)
    assert res.get("status") == "success"
    assert res.get("tool_selected") is False
    assert len(res.get("tool_calls")) == 0
    assert len(res.get("content", "")) > 0


def run_benchmark(iterations: int = 5) -> dict:
    """Run full benchmark evaluating adherence, hallucination, and throughput."""
    print(f"\n=======================================================")
    print(f"🚀 Running Sovereign Tool-Calling & JSON Benchmark ({iterations} iterations)")
    print(f"=======================================================\n")

    if not forge_llm.is_mlx_server_active():
        print("❌ Error: MLX server is not active on port 8080.")
        return {"error": "server_inactive"}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_entity",
                "description": "Look up agent metadata or mission record by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "tier": {"type": "integer", "minimum": 1, "maximum": 3}
                    },
                    "required": ["entity_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_budget",
                "description": "Calculate token expenditure and cognitive ROI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tokens_spent": {"type": "integer"},
                        "cost_weight": {"type": "number"}
                    },
                    "required": ["tokens_spent"]
                }
            }
        }
    ]

    schema = {
        "type": "object",
        "properties": {
            "test_id": {"type": "string"},
            "score": {"type": "number"},
            "passed": {"type": "boolean"}
        },
        "required": ["test_id", "score", "passed"]
    }

    test_cases = [
        ("tool", "Find record for entity MISSION-019 with tier 2", "lookup_entity"),
        ("tool", "Calculate budget for 12000 tokens spent with 0.1 cost weight", "calculate_budget"),
        ("structured", "Test benchmark-1 scored 98.5 and passed verification.", "schema"),
        ("negative", "Good morning! Can you explain the Anti-Asch epistemic protocol?", "none"),
        ("tool", "Lookup entity AGENT-GUARDIAN", "lookup_entity")
    ]

    results = []
    t_start = time.perf_counter()

    for idx, (kind, prompt, expected) in enumerate(test_cases, 1):
        print(f"[{idx}/{len(test_cases)}] Testing '{kind}': {prompt[:45]}...")
        if kind == "tool":
            res = forge_llm.call_tool(prompt, tools=tools)
            selected = res.get("tool_selected", False)
            valid = res.get("schema_valid", False)
            tool_name = res["tool_calls"][0]["name"] if res.get("tool_calls") else None
            correct_tool = (tool_name == expected)
            passed = selected and valid and correct_tool
            results.append({
                "type": kind,
                "prompt": prompt,
                "passed": passed,
                "schema_valid": valid,
                "tool_selected": selected,
                "tool_name": tool_name,
                "expected": expected,
                "latency_sec": res.get("latency_sec")
            })
            print(f"    -> Tool: {tool_name} | Schema Valid: {valid} | Latency: {res.get('latency_sec')}s | Result: {'PASS' if passed else 'FAIL'}")
        elif kind == "structured":
            res = forge_llm.extract_structured(prompt, schema=schema)
            valid = res.get("schema_valid", False)
            passed = valid and res.get("status") == "success"
            results.append({
                "type": kind,
                "prompt": prompt,
                "passed": passed,
                "schema_valid": valid,
                "latency_sec": res.get("latency_sec")
            })
            print(f"    -> Structured Output Valid: {valid} | Latency: {res.get('latency_sec')}s | Result: {'PASS' if passed else 'FAIL'}")
        elif kind == "negative":
            res = forge_llm.call_tool(prompt, tools=tools)
            selected = res.get("tool_selected", False)
            hallucinated = selected
            passed = not hallucinated
            results.append({
                "type": kind,
                "prompt": prompt,
                "passed": passed,
                "hallucinated": hallucinated,
                "latency_sec": res.get("latency_sec")
            })
            print(f"    -> Hallucinated Tool: {hallucinated} | Latency: {res.get('latency_sec')}s | Result: {'PASS' if passed else 'FAIL'}")

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    adherence_rate = (passed_count / total) * 100.0
    latencies = [r["latency_sec"] for r in results if r.get("latency_sec")]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    summary = {
        "total_tests": total,
        "passed": passed_count,
        "adherence_rate_pct": adherence_rate,
        "avg_latency_sec": avg_latency,
        "hallucination_rate_pct": 0.0 if not any(r.get("hallucinated") for r in results) else 100.0,
        "falsification_criteria_met": adherence_rate >= 90.0
    }

    print(f"\n=======================================================")
    print(f"📊 Benchmark Summary:")
    print(f"   Pass Rate: {passed_count}/{total} ({adherence_rate:.1f}%)")
    print(f"   Avg Latency: {avg_latency}s")
    print(f"   Hallucination Rate: {summary['hallucination_rate_pct']}%")
    print(f"   Target Adherence (>90%): {'✅ SATISFIED' if adherence_rate >= 90 else '❌ FALSIFIED'}")
    print(f"=======================================================\n")

    return summary


def main():
    if "--benchmark" in sys.argv:
        summary = run_benchmark()
        sys.exit(0 if summary.get("falsification_criteria_met") else 1)
    else:
        sys.exit(pytest.main(["-v", str(Path(__file__).resolve())]))


if __name__ == "__main__":
    main()
