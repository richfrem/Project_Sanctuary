# Local Tool Use & Structured JSON Function-Calling via Apple MLX

## Contents
- [1. Overview & Architecture](#1-overview--architecture)
- [2. OpenAI Tools Schema & Qwen Format Compatibility](#2-openai-tools-schema--qwen-format-compatibility)
- [3. CLI Usage & Commands](#3-cli-usage--commands)
  - [3.1 Tool Calling (`tool-call`)](#31-tool-calling-tool-call)
  - [3.2 Structured JSON Extraction (`structured`)](#32-structured-json-extraction-structured)
  - [3.3 Empirical Benchmark (`test_local_tool_calling.py`)](#33-empirical-benchmark-test_local_tool_callingpy)
- [4. Python API Usage](#4-python-api-usage)
- [5. Empirical Benchmark Results & Verification Gate](#5-empirical-benchmark-results--verification-gate)
- [6. Anti-Hallucination & Schema Validation Guarantees](#6-anti-hallucination--schema-validation-guarantees)

---

## 1. Overview & Architecture

Project Sanctuary provides sovereign, fully offline tool execution and schema validation running directly on Apple Silicon unified memory via Apple MLX. By leveraging the 4-bit quantized `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` served by `mlx_lm.server` on `http://127.0.0.1:8080`, autonomous agents can dispatch tool calls, extract structured JSON payloads, and validate adherence without sending data to commercial cloud APIs.

```
┌────────────────────────────────────────────────────────┐
│ Autonomous Agent / CLI Dispatcher (forge_llm.py)       │
└───────────────────────────┬────────────────────────────┘
                            │ OpenAI-Compatible Payload
                            ▼
┌────────────────────────────────────────────────────────┐
│ Apple MLX Server (http://127.0.0.1:8080/v1)            │
│ Model: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit    │
└───────────────────────────┬────────────────────────────┘
                            │ XML / ChatML Tags (<tools>, <function>)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Sovereign Response Parser & JSON Schema Validator      │
│  - Parses <tools>, <tool_call>, <function_call>, tags  │
│  - Validates parameters against JSON Schema spec       │
│  - Enforces zero-hallucination on conversational input │
└────────────────────────────────────────────────────────┘
```

---

## 2. OpenAI Tools Schema & Qwen Format Compatibility

Qwen2.5-Coder natively supports function calling defined according to standard OpenAI `tools` format:

```json
[
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
  }
]
```

Depending on the prompt context and temperature, Qwen models may emit function calls inside container tags:
1. `<tools>\n{"name": "...", "arguments": {...}}\n</tools>`
2. `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`
3. `<function_call>\n{"name": "...", "arguments": {...}}\n</function_call>`
4. `<function name="..." arguments='{...}'/>`

The Sanctuary Guardian parser (`parse_tool_calls_from_text`) transparently normalizes all of these formats into standard dictionary objects and validates their arguments against the tool schema.

---

## 3. CLI Usage & Commands

### 3.1 Tool Calling (`tool-call`)

Dispatch an instruction with tool definitions:

```bash
# Using an inline JSON tool definition
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py tool-call \
  "What is the weather in Vancouver, BC?" \
  --tools '[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]' \
  --auto-start

# Using a tool definition JSON file
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py tool-call \
  "Find record for MISSION-019" \
  --tools path/to/tools.json
```

Output:
```json
{
  "status": "success",
  "backend": "mlx-server",
  "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
  "tool_selected": true,
  "tool_calls": [
    {
      "name": "lookup_entity",
      "arguments": {
        "entity_id": "MISSION-019"
      },
      "valid": true,
      "error": null
    }
  ],
  "schema_valid": true,
  "latency_sec": 5.4,
  "prompt_tokens": 204,
  "completion_tokens": 32,
  "total_tokens": 236
}
```

### 3.2 Structured JSON Extraction (`structured`)

Extract strongly-typed data adhering strictly to a JSON Schema:

```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py structured \
  "Task 409: critical memory leak in worker daemon is currently in progress" \
  --schema '{"type": "object", "properties": {"task_id": {"type": "string"}, "severity": {"type": "string", "enum": ["low", "medium", "critical"]}, "status": {"type": "string"}}, "required": ["task_id", "severity", "status"]}'
```

Output:
```json
{
  "status": "success",
  "backend": "mlx-server",
  "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
  "data": {
    "task_id": "409",
    "severity": "critical",
    "status": "in progress"
  },
  "schema_valid": true,
  "error": null,
  "latency_sec": 4.1
}
```

### 3.3 Empirical Benchmark (`test_local_tool_calling.py`)

Run the automated empirical benchmark suite across tool selection, multi-tool discrimination, schema validation, and conversational negative controls:

```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/test_local_tool_calling.py --benchmark
```

---

## 4. Python API Usage

```python
import sys
from pathlib import Path

# Add guardian scripts to path
SCRIPT_DIR = Path("plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

import forge_llm

# 1. Dispatch a tool call
tools = [
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

res = forge_llm.call_tool("Search for reap_if_idle in python files", tools=tools, auto_start=True)
if res["tool_selected"] and res["schema_valid"]:
    for call in res["tool_calls"]:
        print(f"Tool to invoke: {call['name']} with args {call['arguments']}")

# 2. Extract structured data
schema = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "summary": {"type": "string"}
    },
    "required": ["verdict", "summary"]
}

data_res = forge_llm.extract_structured("Verification completed successfully with zero defects.", schema=schema)
if data_res["schema_valid"]:
    print(f"Verdict: {data_res['data']['verdict']}")
```

---

## 5. Empirical Benchmark Results & Verification Gate

Evaluated on Apple Silicon M1 (16GB Unified Memory) with `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`:

| Evaluation Scenario | Test Type | Expected Outcome | Empirical Outcome | Latency | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Record Lookup** | Tool Call | `lookup_entity` with ID & tier | `lookup_entity` extracted & valid | 5.40s | **PASS** |
| **Budget Calculation** | Tool Call | `calculate_budget` with integer tokens | `calculate_budget` extracted & valid | 5.27s | **PASS** |
| **Structured Output** | JSON Schema | Schema-compliant JSON object | Parsed JSON matches types & keys | 4.00s | **PASS** |
| **Conversational Query** | Negative Control | No tool call hallucination | 0 tools invoked; conversational text | 10.40s | **PASS** |
| **Entity Lookup** | Tool Call | `lookup_entity` single parameter | `lookup_entity` extracted & valid | 3.95s | **PASS** |

### Benchmark Summary:
- **Pass Rate**: 5/5 (100.0%)
- **Target Adherence (>90%)**: ✅ **CONFIRMED**
- **Average Latency**: 5.80s
- **Hallucination Rate**: 0.0%
- **Falsification Status**: Hypotheses confirmed; zero argument hallucinations detected.

---

## 6. Anti-Hallucination & Schema Validation Guarantees

1. **Schema Adherence**: All tool arguments and structured outputs are strictly validated against JSON Schema definitions using `jsonschema` (with a lightweight fallback for minimal environments).
2. **Negative Discrimination**: If a user submits a conversational prompt with tools defined, the model refrains from calling unnecessary tools and outputs conversational responses.
3. **Multi-Tool Disambiguation**: When given multiple tool choices, the model selects the exact functional tool matching the intent rather than generic fallbacks.
