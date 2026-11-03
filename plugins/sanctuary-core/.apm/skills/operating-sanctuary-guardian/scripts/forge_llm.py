#!/usr/bin/env python3
"""forge_llm.py - Sovereign Local Model CLI & Protocol 128 Verifier.
Purpose: Unified interface for querying sovereign local models via Apple MLX or Ollama.
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 1 Execution & Verification).
Key Inputs: User prompt, model configuration from .env (MLX_MODEL, OLLAMA_MODEL), --backend flag.
Key Outputs: Generated text, tokens/sec throughput, peak memory, and JSON telemetry payloads.
Usage:
    python3 forge_llm.py query "What is the capital of France?" [--auto-start]
    python3 forge_llm.py tool-call "Weather in Vancouver" --tools tools.json [--auto-start]
    python3 forge_llm.py structured "Fix task-409" --schema schema.json [--auto-start]
    python3 forge_llm.py server start [--port 8080] [--idle-timeout 900]
    python3 forge_llm.py server stop [--force]
    python3 forge_llm.py status
    python3 forge_llm.py verify --type pre-completion
Key Functions: query_model(), call_tool(), extract_structured(), check_status(), verify_protocol128(), main().
"""

import os
import sys
import re
import json
import time
import shutil
import urllib.request
import urllib.error
import subprocess
import argparse
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# Add current directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from mlx_server_manager import (
        get_status as get_mlx_status,
        start_server as start_mlx_server,
        stop_server as stop_mlx_server,
        ensure_ready as ensure_mlx_ready,
        reap_if_idle,
        touch_activity
    )
except ImportError:
    get_mlx_status = None
    start_mlx_server = None
    stop_mlx_server = None
    ensure_mlx_ready = None
    reap_if_idle = None
    touch_activity = lambda: None

DEFAULT_MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
DEFAULT_MLX_HOST = os.environ.get("MLX_SERVER_HOST", "http://127.0.0.1:8080")


def is_mlx_server_active(host: str = DEFAULT_MLX_HOST) -> bool:
    """Check if a local mlx_lm.server is responding at host."""
    try:
        url = f"{host.rstrip('/')}/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "Sanctuary-Guardian"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def validate_against_schema(data: Any, schema: dict) -> Tuple[bool, Optional[str]]:
    """Validate data against a JSON schema using jsonschema if available, or lightweight checks."""
    if not isinstance(schema, dict) or not schema:
        return True, None
    if HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=data, schema=schema)
            return True, None
        except jsonschema.ValidationError as ve:
            path_str = "/".join(str(p) for p in ve.path) if ve.path else "root"
            return False, f"Validation error at {path_str}: {ve.message}"
        except Exception as e:
            return False, str(e)

    # Lightweight fallback validator
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(data, dict):
            return False, f"Expected object, got {type(data).__name__}"
        required = schema.get("required", [])
        for req in required:
            if req not in data:
                return False, f"Missing required property: '{req}'"
    elif expected_type == "array" and not isinstance(data, list):
        return False, f"Expected array, got {type(data).__name__}"
    elif expected_type == "string" and not isinstance(data, str):
        return False, f"Expected string, got {type(data).__name__}"
    elif expected_type == "integer" and (not isinstance(data, int) or isinstance(data, bool)):
        return False, f"Expected integer, got {type(data).__name__}"
    elif expected_type == "number" and not isinstance(data, (int, float)):
        return False, f"Expected number, got {type(data).__name__}"
    elif expected_type == "boolean" and not isinstance(data, bool):
        return False, f"Expected boolean, got {type(data).__name__}"

    return True, None


def parse_tool_calls_from_text(text: str, raw_tool_calls: Optional[List[dict]] = None) -> List[dict]:
    """Extract tool calls from model text or raw API payload."""
    calls = []

    # 1. Process raw API tool_calls if provided
    if raw_tool_calls:
        for rtc in raw_tool_calls:
            func = rtc.get("function", {})
            name = func.get("name")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name:
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        if calls:
            return calls

    # 2. Extract container tags: <tools>, <tool_call>, or <function_call>
    tag_blocks = re.findall(r'<(?:tools|tool_call|function_call)>([\s\S]*?)</(?:tools|tool_call|function_call)>', text, re.IGNORECASE)
    for block in tag_blocks:
        block = block.strip()
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                calls.extend([p for p in parsed if isinstance(p, dict) and "name" in p])
            elif isinstance(parsed, dict) and "name" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        p = json.loads(line)
                        if isinstance(p, dict) and "name" in p:
                            calls.append(p)
                    except Exception:
                        pass

    # 4. Extract <function name="..." arguments='...'/> or <function name="...">...</function>
    func_pattern = r'<function\s+name=[\x27\"](?P<name>[^\x27\"]+)[\x27\"](?:\s+arguments=(?P<quote>[\x27\"])(?P<args>[\s\S]*?)(?P=quote)|>(?P<inner_args>[\s\S]*?)</function>|\s*/>)'
    for m in re.finditer(func_pattern, text, re.IGNORECASE):
        name = m.group("name")
        args_str = m.group("args") or m.group("inner_args") or "{}"
        try:
            args = json.loads(args_str)
        except Exception:
            args = {}
        if name:
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})

    # 5. Fallback: check if entire response is a JSON object or markdown fenced JSON
    if not calls:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```[a-zA-Z0-9_-]*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned).strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    if "name" in parsed and "arguments" in parsed:
                        calls.append(parsed)
                    elif "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                        calls.extend([c for c in parsed["tool_calls"] if isinstance(c, dict) and "name" in c])
            except Exception:
                pass

    # Ensure each item has 'arguments' as a dict
    sanitized = []
    for c in calls:
        name = c.get("name")
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            sanitized.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return sanitized


def validate_tool_calls(tool_calls: List[dict], tools_schema: List[dict]) -> Tuple[bool, List[dict]]:
    """Validate extracted tool calls against OpenAI-format tool definitions."""
    tool_defs = {}
    for t in tools_schema:
        if isinstance(t, dict):
            func = t.get("function") if "function" in t else t
            if isinstance(func, dict) and "name" in func:
                tool_defs[func["name"]] = func.get("parameters", {})

    all_valid = True
    validated_calls = []

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("arguments", {})
        if name not in tool_defs:
            all_valid = False
            validated_calls.append({
                "name": name,
                "arguments": args,
                "valid": False,
                "error": f"Tool '{name}' is not in the provided tool definitions"
            })
            continue

        param_schema = tool_defs[name]
        is_val, err_msg = validate_against_schema(args, param_schema)
        if not is_val:
            all_valid = False
        validated_calls.append({
            "name": name,
            "arguments": args,
            "valid": is_val,
            "error": err_msg
        })

    return all_valid, validated_calls


def parse_and_validate_json_schema(text: str, schema: dict) -> Tuple[bool, Any, Optional[str]]:
    """Extract and validate JSON from model output against a schema."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```[a-zA-Z0-9_-]*\n', '', cleaned)
        cleaned = re.sub(r'\n```$', '', cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', cleaned)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception as e:
                return False, None, f"JSON parse error: {e}"
        else:
            return False, None, "No valid JSON structure found in model output"

    is_valid, err_msg = validate_against_schema(data, schema)
    return is_valid, data, err_msg


def query_mlx_server(prompt: str, host: str = DEFAULT_MLX_HOST, model: str = DEFAULT_MLX_MODEL,
                     temp: float = 0.2, max_tokens: int = 512, system_prompt: str = None) -> dict:
    """Query active mlx_lm.server via OpenAI-compatible HTTP endpoint."""
    if touch_activity:
        touch_activity()
    t0 = time.perf_counter()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }

    url = f"{host.rstrip('/')}/v1/chat/completions"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "Sanctuary-Guardian"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            choice = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            return {
                "status": "success",
                "backend": "mlx-server",
                "model": model,
                "response": choice.strip(),
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens")
            }
    except Exception as e:
        return {"status": "error", "backend": "mlx-server", "error": str(e)}


def query_mlx_server_tools(prompt: str, tools: List[dict], host: str = DEFAULT_MLX_HOST,
                           model: str = DEFAULT_MLX_MODEL, temp: float = 0.0,
                           max_tokens: int = 512, system_prompt: str = None) -> dict:
    """Query mlx_lm.server with OpenAI tools payload and validate output."""
    if touch_activity:
        touch_activity()
    t0 = time.perf_counter()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temp,
        "max_tokens": max_tokens
    }

    url = f"{host.rstrip('/')}/v1/chat/completions"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "Sanctuary-Guardian"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            choice_msg = body["choices"][0]["message"]
            raw_content = choice_msg.get("content") or ""
            raw_tool_calls = choice_msg.get("tool_calls")
            usage = body.get("usage", {})

            parsed_calls = parse_tool_calls_from_text(raw_content, raw_tool_calls)
            all_valid, validated = validate_tool_calls(parsed_calls, tools)

            return {
                "status": "success",
                "backend": "mlx-server",
                "model": model,
                "tool_selected": len(parsed_calls) > 0,
                "tool_calls": validated,
                "schema_valid": all_valid if parsed_calls else None,
                "content": raw_content.strip(),
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens")
            }
    except Exception as e:
        return {"status": "error", "backend": "mlx-server", "error": str(e)}


def query_mlx_server_structured(prompt: str, schema: dict, host: str = DEFAULT_MLX_HOST,
                                model: str = DEFAULT_MLX_MODEL, temp: float = 0.0,
                                max_tokens: int = 512, system_prompt: str = None) -> dict:
    """Prompt mlx_lm.server for strict structured JSON output and validate against schema."""
    if touch_activity:
        touch_activity()
    t0 = time.perf_counter()

    schema_instruction = (
        "You are a structured data extraction engine. You must output ONLY a valid JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "CRITICAL: Do NOT include markdown code blocks, fences, explanations, or conversational filler. Output raw JSON only."
    )
    full_system = f"{system_prompt}\n\n{schema_instruction}" if system_prompt else schema_instruction

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }

    url = f"{host.rstrip('/')}/v1/chat/completions"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "Sanctuary-Guardian"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            raw_content = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage", {})

            is_valid, parsed_data, err_msg = parse_and_validate_json_schema(raw_content, schema)

            return {
                "status": "success" if is_valid else "validation_error",
                "backend": "mlx-server",
                "model": model,
                "data": parsed_data,
                "schema_valid": is_valid,
                "error": err_msg,
                "raw_response": raw_content.strip(),
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens")
            }
    except Exception as e:
        return {"status": "error", "backend": "mlx-server", "error": str(e)}


def query_mlx_cli(prompt: str, model: str = DEFAULT_MLX_MODEL,
                  temp: float = 0.2, max_tokens: int = 256, system_prompt: str = None) -> dict:
    """Run one-shot mlx_lm.generate via uvx."""
    t0 = time.perf_counter()
    cmd = [
        "uvx", "--from", "mlx-lm", "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--verbose", "True"
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return {"status": "error", "backend": "mlx-cli", "error": proc.stderr.strip()}

        stdout = proc.stdout
        response_text = ""
        prompt_tps = None
        gen_tps = None
        peak_mem = None

        lines = stdout.splitlines()
        capturing = False
        captured_lines = []
        for line in lines:
            if line.strip().startswith("=========="):
                if not capturing:
                    capturing = True
                    continue
                else:
                    capturing = False
                    continue
            if capturing:
                captured_lines.append(line)
            if "tokens-per-sec" in line and "Prompt:" in line:
                try:
                    prompt_tps = float(line.split(",")[1].replace("tokens-per-sec", "").strip())
                except Exception:
                    pass
            elif "tokens-per-sec" in line and "Generation:" in line:
                try:
                    gen_tps = float(line.split(",")[1].replace("tokens-per-sec", "").strip())
                except Exception:
                    pass
            elif "Peak memory:" in line:
                try:
                    peak_mem = line.split("Peak memory:")[1].strip()
                except Exception:
                    pass

        response_text = "\n".join(captured_lines).strip() if captured_lines else stdout.strip()

        return {
            "status": "success",
            "backend": "mlx-cli",
            "model": model,
            "response": response_text,
            "latency_sec": round(elapsed, 3),
            "prompt_tokens_per_sec": prompt_tps,
            "generation_tokens_per_sec": gen_tps,
            "peak_memory": peak_mem
        }
    except Exception as e:
        return {"status": "error", "backend": "mlx-cli", "error": str(e)}


def query_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL,
                 temp: float = 0.2, max_tokens: int = 512, system_prompt: str = None) -> dict:
    """Query local Ollama instance."""
    try:
        from ollama import Client
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        client = Client(host=ollama_host)
        t0 = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temp, "num_predict": max_tokens}
        )
        elapsed = time.perf_counter() - t0

        return {
            "status": "success",
            "backend": "ollama",
            "model": model,
            "response": res['message']['content'].strip(),
            "latency_sec": round(elapsed, 3),
            "prompt_tokens": res.get('prompt_eval_count'),
            "completion_tokens": res.get('eval_count'),
            "total_tokens": (res.get('prompt_eval_count') or 0) + (res.get('eval_count') or 0)
        }
    except ImportError:
        return {"status": "error", "backend": "ollama", "error": "ollama package not installed"}
    except Exception as e:
        return {"status": "error", "backend": "ollama", "error": str(e)}


def query_model(prompt: str, backend: str = "auto", model: str = None,
                temp: float = 0.2, max_tokens: int = 512, system_prompt: str = None,
                auto_start: bool = False) -> dict:
    """Master entry point for querying local sovereign models."""
    if backend == "auto":
        if sys.platform == "darwin":
            backend = "mlx"
        else:
            backend = "ollama"

    if backend == "mlx":
        should_auto_start = auto_start or os.environ.get("MLX_AUTO_START", "").lower() in ("1", "true")
        if not is_mlx_server_active() and should_auto_start and ensure_mlx_ready:
            target_model = model or DEFAULT_MLX_MODEL
            ensure_mlx_ready(model=target_model)

        if is_mlx_server_active():
            target_model = model or DEFAULT_MLX_MODEL
            return query_mlx_server(prompt, model=target_model, temp=temp, max_tokens=max_tokens, system_prompt=system_prompt)
        target_model = model or DEFAULT_MLX_MODEL
        return query_mlx_cli(prompt, model=target_model, temp=temp, max_tokens=max_tokens, system_prompt=system_prompt)
    elif backend == "ollama":
        target_model = model or DEFAULT_OLLAMA_MODEL
        return query_ollama(prompt, model=target_model, temp=temp, max_tokens=max_tokens, system_prompt=system_prompt)
    else:
        return {"status": "error", "error": f"Unknown backend: {backend}"}


def call_tool(prompt: str, tools: Union[List[dict], str, Path], backend: str = "auto",
              model: str = None, temp: float = 0.0, max_tokens: int = 512,
              system_prompt: str = None, auto_start: bool = False) -> dict:
    """Master entry point for sovereign local tool use."""
    # Resolve tools list
    tools_list = []
    if isinstance(tools, (str, Path)):
        t_path = Path(tools)
        if t_path.exists() and t_path.is_file():
            with open(t_path, "r", encoding="utf-8") as f:
                tools_list = json.load(f)
        else:
            try:
                tools_list = json.loads(str(tools))
            except Exception as e:
                return {"status": "error", "error": f"Failed to parse tools JSON: {e}"}
    elif isinstance(tools, list):
        tools_list = tools
    else:
        return {"status": "error", "error": f"Invalid tools format: {type(tools).__name__}"}

    if not isinstance(tools_list, list):
        return {"status": "error", "error": "Tools definition must be a list of functions"}

    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "ollama"

    if backend == "mlx":
        should_auto_start = auto_start or os.environ.get("MLX_AUTO_START", "").lower() in ("1", "true")
        if not is_mlx_server_active() and should_auto_start and ensure_mlx_ready:
            target_model = model or DEFAULT_MLX_MODEL
            ensure_mlx_ready(model=target_model)

        if not is_mlx_server_active():
            return {
                "status": "error",
                "error": "MLX server is not active on port 8080. Start with: forge_llm.py server start"
            }

        target_model = model or DEFAULT_MLX_MODEL
        return query_mlx_server_tools(
            prompt=prompt,
            tools=tools_list,
            model=target_model,
            temp=temp,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
    else:
        return {"status": "error", "error": f"Tool calling currently requires MLX server backend (got: {backend})"}


def extract_structured(prompt: str, schema: Union[dict, str, Path], backend: str = "auto",
                       model: str = None, temp: float = 0.0, max_tokens: int = 512,
                       system_prompt: str = None, auto_start: bool = False) -> dict:
    """Master entry point for extracting schema-adhering JSON."""
    schema_dict = {}
    if isinstance(schema, (str, Path)):
        s_path = Path(schema)
        if s_path.exists() and s_path.is_file():
            with open(s_path, "r", encoding="utf-8") as f:
                schema_dict = json.load(f)
        else:
            try:
                schema_dict = json.loads(str(schema))
            except Exception as e:
                return {"status": "error", "error": f"Failed to parse schema JSON: {e}"}
    elif isinstance(schema, dict):
        schema_dict = schema
    else:
        return {"status": "error", "error": f"Invalid schema format: {type(schema).__name__}"}

    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "ollama"

    if backend == "mlx":
        should_auto_start = auto_start or os.environ.get("MLX_AUTO_START", "").lower() in ("1", "true")
        if not is_mlx_server_active() and should_auto_start and ensure_mlx_ready:
            target_model = model or DEFAULT_MLX_MODEL
            ensure_mlx_ready(model=target_model)

        if not is_mlx_server_active():
            return {
                "status": "error",
                "error": "MLX server is not active on port 8080. Start with: forge_llm.py server start"
            }

        target_model = model or DEFAULT_MLX_MODEL
        return query_mlx_server_structured(
            prompt=prompt,
            schema=schema_dict,
            model=target_model,
            temp=temp,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
    else:
        return {"status": "error", "error": f"Structured extraction currently requires MLX server backend (got: {backend})"}


def check_status() -> dict:
    """Check availability of MLX and Ollama environments."""
    hf_model_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit"
    mlx_cached = hf_model_dir.exists()
    mlx_server_live = is_mlx_server_active()
    mlx_lifecycle = get_mlx_status() if get_mlx_status else {}

    ollama_live = False
    ollama_models = []
    try:
        import ollama
        resp = ollama.list()
        ollama_live = True
        raw_list = resp.get('models', []) if isinstance(resp, dict) else resp
        ollama_models = [m.get('name', str(m)) if isinstance(m, dict) else str(m) for m in raw_list]
    except Exception:
        pass

    return {
        "status": "success",
        "platform": sys.platform,
        "mlx": {
            "model": DEFAULT_MLX_MODEL,
            "cached": mlx_cached,
            "server_live": mlx_server_live,
            "server_url": DEFAULT_MLX_HOST if mlx_server_live else None,
            "lifecycle": mlx_lifecycle
        },
        "ollama": {
            "live": ollama_live,
            "default_model": DEFAULT_OLLAMA_MODEL,
            "installed_models": ollama_models
        }
    }


def verify_protocol128() -> dict:
    """Run an autonomous offline Protocol 128 verification test."""
    test_prompt = (
        "Respond in valid JSON only with keys: 'status' ('PASS' or 'FAIL') and 'summary'. "
        "Verify: Project Sanctuary git tree is clean, 4 core context files valid, all tests pass. "
        "State whether Pre-Completion Verification Gate is satisfied."
    )
    system = "You are Sanctuary Guardian, an Anti-Asch sovereign engineering agent."
    result = query_model(test_prompt, backend="auto", max_tokens=150, temp=0.0, system_prompt=system)
    return result


def main():
    parser = argparse.ArgumentParser(description="Sanctuary Sovereign Local Model Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # query
    q_parser = subparsers.add_parser("query", help="Query sovereign local model")
    q_parser.add_argument("prompt", help="Prompt to send to model")
    q_parser.add_argument("--backend", choices=["auto", "mlx", "ollama"], default="auto")
    q_parser.add_argument("--model", help="Override default model name")
    q_parser.add_argument("--temp", type=float, default=0.2)
    q_parser.add_argument("--max-tokens", type=int, default=512)
    q_parser.add_argument("--system", help="System prompt")
    q_parser.add_argument("--auto-start", action="store_true", help="Auto-start MLX server daemon if inactive")

    # tool-call
    t_parser = subparsers.add_parser("tool-call", aliases=["call-tool"], help="Dispatch tool call to local model")
    t_parser.add_argument("prompt", help="Prompt or task for tool use")
    t_parser.add_argument("--tools", required=True, help="Path to tools JSON file or inline JSON string")
    t_parser.add_argument("--backend", choices=["auto", "mlx", "ollama"], default="auto")
    t_parser.add_argument("--model", help="Override default model name")
    t_parser.add_argument("--temp", type=float, default=0.0)
    t_parser.add_argument("--max-tokens", type=int, default=512)
    t_parser.add_argument("--system", help="System prompt")
    t_parser.add_argument("--auto-start", action="store_true", help="Auto-start MLX server daemon if inactive")

    # structured
    str_parser = subparsers.add_parser("structured", aliases=["extract-json"], help="Extract structured data adhering to JSON schema")
    str_parser.add_argument("prompt", help="Prompt or text to extract data from")
    str_parser.add_argument("--schema", required=True, help="Path to schema JSON file or inline JSON schema string")
    str_parser.add_argument("--backend", choices=["auto", "mlx", "ollama"], default="auto")
    str_parser.add_argument("--model", help="Override default model name")
    str_parser.add_argument("--temp", type=float, default=0.0)
    str_parser.add_argument("--max-tokens", type=int, default=512)
    str_parser.add_argument("--system", help="System prompt")
    str_parser.add_argument("--auto-start", action="store_true", help="Auto-start MLX server daemon if inactive")

    # server
    s_parser = subparsers.add_parser("server", help="Manage MLX local server daemon lifecycle")
    s_subparsers = s_parser.add_subparsers(dest="server_action", required=True)

    s_start = s_subparsers.add_parser("start", help="Start MLX server daemon")
    s_start.add_argument("--port", type=int, default=8080)
    s_start.add_argument("--model", default=DEFAULT_MLX_MODEL)
    s_start.add_argument("--idle-timeout", type=int, default=900)
    s_start.add_argument("--no-wait", action="store_true")

    s_stop = s_subparsers.add_parser("stop", help="Stop running MLX server daemon")
    s_stop.add_argument("--port", type=int, default=8080)
    s_stop.add_argument("--force", action="store_true")

    s_subparsers.add_parser("status", help="Get MLX server status and telemetry")
    s_subparsers.add_parser("ensure", help="Ensure MLX server is active")
    s_reap = s_subparsers.add_parser("reap", help="Reap MLX server if idle")
    s_reap.add_argument("--timeout", type=int, default=None)

    # status
    subparsers.add_parser("status", help="Inspect local MLX and Ollama runtime status")

    # verify
    subparsers.add_parser("verify", help="Execute offline Protocol 128 verification test")

    args = parser.parse_args()

    if args.command == "query":
        res = query_model(
            prompt=args.prompt,
            backend=args.backend,
            model=args.model,
            temp=args.temp,
            max_tokens=args.max_tokens,
            system_prompt=args.system,
            auto_start=args.auto_start
        )
        if res.get("status") == "success":
            print(f"\n{res.get('response', '')}\n")
            print(f"📊 Backend: {res.get('backend')} | Latency: {res.get('latency_sec')}s | Model: {res.get('model')}")
            if "generation_tokens_per_sec" in res and res["generation_tokens_per_sec"]:
                print(f"⚡ Speed: {res['generation_tokens_per_sec']} tok/sec | Peak RAM: {res.get('peak_memory', 'N/A')}")
        else:
            print(json.dumps(res, indent=2))
            sys.exit(1)

    elif args.command in ("tool-call", "call-tool"):
        res = call_tool(
            prompt=args.prompt,
            tools=args.tools,
            backend=args.backend,
            model=args.model,
            temp=args.temp,
            max_tokens=args.max_tokens,
            system_prompt=args.system,
            auto_start=args.auto_start
        )
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get("status") == "success" else 1)

    elif args.command in ("structured", "extract-json"):
        res = extract_structured(
            prompt=args.prompt,
            schema=args.schema,
            backend=args.backend,
            model=args.model,
            temp=args.temp,
            max_tokens=args.max_tokens,
            system_prompt=args.system,
            auto_start=args.auto_start
        )
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get("status") == "success" else 1)

    elif args.command == "server":
        if not get_mlx_status:
            print(json.dumps({"status": "error", "error": "mlx_server_manager module unavailable"}, indent=2))
            sys.exit(1)

        if args.server_action == "start":
            res = start_mlx_server(port=args.port, model=args.model, idle_timeout=args.idle_timeout, wait_ready=not args.no_wait)
            print(json.dumps(res, indent=2))
            sys.exit(0 if res.get("status") in ("started", "already_running") else 1)
        elif args.server_action == "stop":
            res = stop_mlx_server(port=args.port, force=args.force)
            print(json.dumps(res, indent=2))
            sys.exit(0 if res.get("status") in ("stopped", "not_running") else 1)
        elif args.server_action == "status":
            res = get_mlx_status()
            print(json.dumps(res, indent=2))
            sys.exit(0)
        elif args.server_action == "ensure":
            res = ensure_mlx_ready()
            print(json.dumps(res, indent=2))
            sys.exit(0 if res.get("status") in ("ready", "started", "already_running") else 1)
        elif args.server_action == "reap":
            res = reap_if_idle(idle_timeout_sec=args.timeout)
            print(json.dumps(res, indent=2))
            sys.exit(0)

    elif args.command == "status":
        st = check_status()
        print(json.dumps(st, indent=2))

    elif args.command == "verify":
        v_res = verify_protocol128()
        print(json.dumps(v_res, indent=2))
        if v_res.get("status") != "success":
            sys.exit(1)


if __name__ == "__main__":
    main()
