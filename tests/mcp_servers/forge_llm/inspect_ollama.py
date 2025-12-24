#!/usr/bin/env python3
import sys
import json
import time
from pathlib import Path

# Try to import requests, fallback to urllib if necessary or fail gracefully
try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Please install it with 'pip install requests'")
    sys.exit(1)

# Add project root based on .git marker (Robust)
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
# Configuration
# ============================================================================
load_env()

OLLAMA_HOST = get_env_variable("OLLAMA_HOST", required=False) or "127.0.0.1"

# Sanitize host: strip protocol and port if present to avoid duplication
if OLLAMA_HOST.startswith("http://"):
    OLLAMA_HOST = OLLAMA_HOST.replace("http://", "")
elif OLLAMA_HOST.startswith("https://"):
    OLLAMA_HOST = OLLAMA_HOST.replace("https://", "")

if ":" in OLLAMA_HOST:
    OLLAMA_HOST = OLLAMA_HOST.split(":")[0]

OLLAMA_PORT = int(get_env_variable("SANCTUARY_OLLAMA_PORT", required=False) or 11434)
BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# The model we expect to find and use for testing
TARGET_MODEL = get_env_variable("OLLAMA_MODEL", required=False) or "hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_connection():
    print_header("1. Connectivity Check")
    url = f"{BASE_URL}/api/version" # usually returns {"version": "..."}
    # If /api/version isn't available in older versions, root / often returns 200 OK "Ollama is running"
    
    print(f"Connecting to: {url}")
    try:
        start_t = time.time()
        # Trying root endpoint first as a simple ping
        resp = requests.get(BASE_URL, timeout=2)
        latency = (time.time() - start_t) * 1000
        
        if resp.status_code == 200:
            print(f"Status: ONLINE (HTTP {resp.status_code})")
            print(f"Latency: {latency:.2f}ms")
            print(f"Message: {resp.text.strip()}")
            return True
        else:
            print(f"Status: WARNING (HTTP {resp.status_code})")
            return True # It responded, so it constitutes a connection
            
    except requests.exceptions.ConnectionError:
        print("Status: FAILED (Connection Refused)")
        print("Tip: Is the Ollama container running? Check 'podman ps'.")
        return False
    except Exception as e:
        print(f"Status: ERROR ({e})")
        return False

def list_models():
    print_header("2. Model Availability")
    url = f"{BASE_URL}/api/tags"
    print(f"Fetching models from: {url}")
    
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Error fetching tags: HTTP {resp.status_code}")
            return None
        
        data = resp.json()
        models = data.get('models', [])
        
        print(f"Found {len(models)} models:")
        found_target = False
        
        for m in models:
            name = m.get('name', 'unknown')
            size_gb = m.get('size', 0) / (1024**3)
            print(f"  - {name} ({size_gb:.2f} GB)")
            
            if name == TARGET_MODEL:
                found_target = True
        
        if found_target:
            print(f"\n[OK] Target model '{TARGET_MODEL}' is available.")
            return TARGET_MODEL
        elif models:
            # If target not found but others exist, pick the first one
            fallback = models[0]['name']
            print(f"\n[WARN] Target model '{TARGET_MODEL}' not found.")
            print(f"       Switching test to available model: '{fallback}'")
            return fallback
        else:
            print("\n[ERR] No models found in Ollama.")
            return None
            
    except Exception as e:
        print(f"Error listing models: {e}")
        return None

def test_generation(model_name):
    print_header(f"3. Generation Test ({model_name})")
    url = f"{BASE_URL}/api/generate"
    
    prompt = "Hello! Are you functioning correctly?"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Prompt: '{prompt}'")
    print("Sending request (please wait)...")
    
    try:
        start_t = time.time()
        resp = requests.post(url, json=payload, timeout=60)
        duration = time.time() - start_t
        
        if resp.status_code == 200:
            data = resp.json()
            response_text = data.get("response", "")
            done = data.get("done", False)
            
            print(f"\nResponse ({duration:.2f}s):")
            print("-" * 40)
            print(response_text.strip())
            print("-" * 40)
            
            if done:
                print("\n[SUCCESS] Generation completed successfully.")
            else:
                print("\n[WARN] Response incomplete (done=False).")
        else:
            print(f"\n[FAIL] API returned HTTP {resp.status_code}")
            print(resp.text)
            
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")

if __name__ == "__main__":
    print("Project Sanctuary - Ollama Inspector")
    if check_connection():
        model_to_use = list_models()
        if model_to_use:
            test_generation(model_to_use)
    else:
        sys.exit(1)
