# council_orchestrator/substrate_monitor.py (v1.1 - Sovereign Override Hardened)
"""
SUBSTRATE MONITOR: Smart AI Engine Picker

This module picks the best available AI engine to use, with backup options.
It ensures the system always has a working AI, even if some services fail.

WHAT IT DOES:
- Picks AI engines in order: Gemini → OpenAI → Ollama (local backup)
- Lets you force-pick a specific engine if needed
- Tests engines live (real API calls) to make sure they work
- Returns engine objects that all work the same way (polymorphism)

WHY IT MATTERS:
- Never runs out of AI power due to service failures
- Local Ollama ensures system works even offline
- Can override automatic choice when you know best

HOW TO USE:
    from substrate_monitor import select_engine

    # Auto-pick best engine
    engine = select_engine()

    # Force specific engine
    engine = select_engine({"force_engine": "ollama"})

RETURNS:
    Working AI engine object, or None if nothing works
"""

import os
from dotenv import load_dotenv

# Load environment variables for engine configuration
load_dotenv()
try:
    from .cognitive_engines.base import BaseCognitiveEngine
    from .cognitive_engines.gemini_engine import GeminiEngine
    from .cognitive_engines.openai_engine import OpenAIEngine
    from .cognitive_engines.ollama_engine import OllamaEngine
except ImportError:
    # Fallback for direct execution
    from cognitive_engines.base import BaseCognitiveEngine
    from cognitive_engines.gemini_engine import GeminiEngine
    from cognitive_engines.openai_engine import OpenAIEngine
    from cognitive_engines.ollama_engine import OllamaEngine

def select_engine(config: dict = None) -> BaseCognitiveEngine | None:
    """
    Selects a cognitive engine based on Guardian override or tiered health check.
    Implements the intelligent triage logic of Protocol 103.
    PRINCIPLE OF SOVEREIGN SUPREMACY: force_engine override is checked FIRST, before any health checks.
    PRINCIPLE OF VERIFIABLE HEALTH: Health checks must perform live API calls, not just code checks.
    """
    print("[SUBSTRATE MONITOR] Initiating cognitive engine triage...")
    print(f"[SUBSTRATE MONITOR] DEBUG: config received: {config}")

    # PRINCIPLE OF SOVEREIGN SUPREMACY: Check for Guardian Override FIRST
    if config and "force_engine" in config:
        forced_engine = config["force_engine"].lower()
        print(f"[SUBSTRATE MONITOR] SOVEREIGN OVERRIDE DETECTED: Force selection of '{forced_engine}' engine.")

        engine: BaseCognitiveEngine | None = None
        if forced_engine == "gemini" or forced_engine == "gemini-2.5-pro":
            print("[SUBSTRATE MONITOR] DEBUG: Creating GeminiEngine...")
            engine = GeminiEngine()
        elif forced_engine == "openai":
            print("[SUBSTRATE MONITOR] DEBUG: Creating OpenAIEngine...")
            engine = OpenAIEngine()
        elif forced_engine == "ollama":
            print("[SUBSTRATE MONITOR] DEBUG: Creating OllamaEngine...")
            engine = OllamaEngine()
        else:
            print(f"[SUBSTRATE MONITOR] CRITICAL FAILURE: Unknown forced engine type '{forced_engine}'.")
            return None

        print(f"[SUBSTRATE MONITOR] DEBUG: Engine created: {type(engine).__name__ if engine else 'None'}")

        # PRINCIPLE OF VERIFIABLE HEALTH: Perform live API call for health check
        if engine:
            print("[SUBSTRATE MONITOR] DEBUG: Performing live health check...")
            try:
                # Attempt a minimal API call to verify actual connectivity
                test_result = engine.run_functional_test()
                if test_result["passed"]:
                    print(f"[SUBSTRATE MONITOR] SUCCESS: Forced engine '{forced_engine}' passed live health check.")
                    return engine
                else:
                    print(f"[SUBSTRATE MONITOR] CRITICAL FAILURE: Forced engine '{forced_engine}' failed live health check: {test_result['details']}")
                    return None
            except Exception as e:
                print(f"[SUBSTRATE MONITOR] CRITICAL FAILURE: Forced engine '{forced_engine}' threw exception during health check: {e}")
                return None
        else:
            print(f"[SUBSTRATE MONITOR] CRITICAL FAILURE: Could not initialize forced engine '{forced_engine}'.")
            return None

    # 2. If no override, proceed with automatic triage
    print("[SUBSTRATE MONITOR] No override detected. Proceeding with automatic triage...")

    # 2a. Check Tier 1 Primary (Gemini) with live health check
    print("[SUBSTRATE MONITOR] Checking Tier 1 Primary: Gemini...")
    gemini = GeminiEngine()
    try:
        test_result = gemini.run_functional_test()
        if test_result["passed"]:
            print("[SUBSTRATE MONITOR] SUCCESS: Gemini engine passed live health check. Selecting as primary.")
            return gemini
        else:
            print(f"[SUBSTRATE MONITOR] WARNING: Gemini engine failed live health check: {test_result['details']}")
    except Exception as e:
        print(f"[SUBSTRATE MONITOR] WARNING: Gemini engine threw exception during health check: {e}")

    # 2b. Check Tier 1 Secondary (OpenAI) with live health check
    print("[SUBSTRATE MONITOR] T1 Primary failed. Checking Tier 1 Secondary: OpenAI...")
    openai = OpenAIEngine()
    try:
        test_result = openai.run_functional_test()
        if test_result["passed"]:
            print("[SUBSTRATE MONITOR] SUCCESS: OpenAI engine passed live health check. Selecting as secondary.")
            return openai
        else:
            print(f"[SUBSTRATE MONITOR] WARNING: OpenAI engine failed live health check: {test_result['details']}")
    except Exception as e:
        print(f"[SUBSTRATE MONITOR] WARNING: OpenAI engine threw exception during health check: {e}")

    # 2c. Default to Tier 2 Sovereign (Ollama) with live health check
    print("[SUBSTRATE MONITOR] T1 Secondary failed. Defaulting to Tier 2 Sovereign: Ollama...")
    ollama = OllamaEngine()
    try:
        test_result = ollama.run_functional_test()
        if test_result["passed"]:
            print("[SUBSTRATE MONITOR] SUCCESS: Ollama engine passed live health check. Selecting as sovereign fallback.")
            return ollama
        else:
            print(f"[SUBSTRATE MONITOR] CRITICAL FAILURE: Ollama engine failed live health check: {test_result['details']}")
            return None
    except Exception as e:
        print(f"[SUBSTRATE MONITOR] CRITICAL FAILURE: Ollama engine threw exception during health check: {e}")
        return None

    # 2d. Catastrophic Failure Condition
    print("[SUBSTRATE MONITOR] CRITICAL FAILURE: All cognitive substrates are unhealthy.")
    return None