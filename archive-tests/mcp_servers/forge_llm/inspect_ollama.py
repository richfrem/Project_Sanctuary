#!/usr/bin/env python3
"""
Project Sanctuary - Ollama Inspector
Tests Ollama connectivity from localhost and/or container network.

Usage:
  python tests/mcp_servers/forge_llm/inspect_ollama.py                 # Test localhost only
  python tests/mcp_servers/forge_llm/inspect_ollama.py --url URL       # Test specific URL
  python tests/mcp_servers/forge_llm/inspect_ollama.py --host all       # Test both
"""
import sys
import time
import argparse
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Install with: pip install requests")
    sys.exit(1)

# Add project root based on .git marker
current = Path(__file__).resolve().parent
while not (current / ".git").exists():
    if current == current.parent:
        raise RuntimeError("Could not find Project_Sanctuary root (no .git folder found)")
    current = current.parent
project_root = current
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp_servers.lib.env_helper import get_env_variable, load_env

# ============================================================================
# Configuration (from Environment)
# ============================================================================
load_env()

# Get host from env or use localhost as default
OLLAMA_HOST_ENV = get_env_variable("OLLAMA_HOST", required=False) or "http://127.0.0.1:11434"

# Container network host (for containers, not accessible from host machine)
CONTAINER_HOST = "http://sanctuary_ollama:11434"

# The model we expect to find
TARGET_MODEL = get_env_variable("OLLAMA_MODEL", required=False) or "hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def check_connection(base_url: str, host_name: str) -> bool:
    """Check if Ollama is reachable at the given URL."""
    print(f"\n--- Testing {host_name}: {base_url} ---")
    
    try:
        start_t = time.time()
        resp = requests.get(base_url, timeout=3)
        latency = (time.time() - start_t) * 1000
        
        if resp.status_code == 200:
            print(f"  Status: ✅ ONLINE (HTTP {resp.status_code}, {latency:.2f}ms)")
            return True
        else:
            print(f"  Status: ⚠️ WARNING (HTTP {resp.status_code})")
            return True
            
    except requests.exceptions.ConnectionError:
        print(f"  Status: ❌ FAILED (Connection Refused)")
        if host_name == "container":
            print(f"  Note: Container hostnames only resolve from inside containers")
        return False
    except Exception as e:
        print(f"  Status: ❌ ERROR ({e})")
        return False


def list_models(base_url: str):
    """List models available on Ollama."""
    url = f"{base_url}/api/tags"
    
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            print(f"  Error fetching tags: HTTP {resp.status_code}")
            return None
        
        data = resp.json()
        models = data.get('models', [])
        
        print(f"  Found {len(models)} models:")
        found_target = False
        
        for m in models:
            name = m.get('name', 'unknown')
            size_gb = m.get('size', 0) / (1024**3)
            print(f"    - {name} ({size_gb:.2f} GB)")
            
            if name == TARGET_MODEL:
                found_target = True
        
        if found_target:
            print(f"  [OK] Target model '{TARGET_MODEL}' is available.")
            return TARGET_MODEL
        elif models:
            fallback = models[0]['name']
            print(f"  [WARN] Target model not found. Using: '{fallback}'")
            return fallback
        else:
            print("  [ERR] No models found.")
            return None
            
    except Exception as e:
        print(f"  Error listing models: {e}")
        return None


def test_generation(base_url: str, model_name: str) -> bool:
    """Test model generation."""
    url = f"{base_url}/api/generate"
    
    prompt = "Hello! Respond with just 'OK' if you're working."
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"  Prompt: '{prompt}'")
    print("  Sending request (please wait)...")
    
    try:
        start_t = time.time()
        resp = requests.post(url, json=payload, timeout=60)
        duration = time.time() - start_t
        
        if resp.status_code == 200:
            data = resp.json()
            response_text = data.get("response", "")
            done = data.get("done", False)
            
            print(f"  Response ({duration:.2f}s): {response_text.strip()[:100]}")
            
            if done:
                print("  [SUCCESS] ✅ Generation completed.")
                return True
            else:
                print("  [WARN] Response incomplete (done=False).")
                return True
        else:
            print(f"  [FAIL] API returned HTTP {resp.status_code}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Generation failed: {e}")
        return False


def run_tests(host_filter: str = "localhost") -> dict:
    """Run tests on specified hosts."""
    print("Project Sanctuary - Ollama Inspector")
    print(f"OLLAMA_HOST from .env: {OLLAMA_HOST_ENV}")
    
    # Build host list based on filter
    hosts = {}
    if host_filter in ("all", "localhost"):
        hosts["localhost"] = OLLAMA_HOST_ENV
    if host_filter in ("all", "container"):
        hosts["container"] = CONTAINER_HOST
    
    if not hosts:
        print(f"Unknown host: {host_filter}. Use 'localhost', 'container', or 'all'.")
        return {}
    
    results = {}
    
    # 1. Connectivity Check
    print_header("1. Connectivity Check")
    for name, url in hosts.items():
        results[name] = {"connected": check_connection(url, name)}
    
    # 2. Model Availability (only for connected hosts)
    print_header("2. Model Availability")
    for name, url in hosts.items():
        if results[name]["connected"]:
            results[name]["model"] = list_models(url)
        else:
            print(f"\n--- Skipping {name} (not connected) ---")
            results[name]["model"] = None
    
    # 3. Generation Test (only for hosts with models)
    print_header("3. Generation Test")
    for name, url in hosts.items():
        if results[name].get("model"):
            print(f"\n--- Testing {name} ---")
            results[name]["generation"] = test_generation(url, results[name]["model"])
        else:
            print(f"\n--- Skipping {name} (no model) ---")
            results[name]["generation"] = False
    
    # Summary
    print_header("Summary")
    for name in hosts:
        status = "✅ PASS" if results[name].get("generation") else "❌ FAIL"
        print(f"  {name}: {status}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Inspector - Test connectivity")
    parser.add_argument("--host", default="localhost", 
                        choices=["all", "localhost", "container"],
                        help="Predefined host groups (default: localhost)")
    parser.add_argument("--url", default=None,
                        help="Directly specify the Ollama API URL (e.g., http://127.0.0.1:11434)")
    args = parser.parse_args()
    
    # If explicit URL is provided, override the localhost group
    if args.url:
        OLLAMA_HOST_ENV = args.url
        print(f"Using explicit URL override: {OLLAMA_HOST_ENV}")

    results = run_tests(args.host)
    
    # Exit code: 0 if at least one host passes, 1 otherwise
    if any(r.get("generation") for r in results.values()):
        sys.exit(0)
    else:
        # Special case: if we are only testing connectivity and it passed, but model isn't there yet
        if any(r.get("connected") for r in results.values()):
            print("\n[INFO] Connectivity verified, but generation test skipped or failed.")
            sys.exit(0)
        sys.exit(1)
