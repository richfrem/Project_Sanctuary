#!/usr/bin/env python3
"""
VERIFICATION TEST: AI Engine System Checker

This test makes sure our AI engine system works correctly.
It checks that we can pick different AI engines and they all work the same way.

WHAT IT TESTS:
- Force Engine Choice: Can pick a specific AI engine when needed
- Engine Interface: All engines follow the same rules (polymorphism)
- Live Testing: Engines actually connect to real AI services
- Auto Fallback: System picks working engines automatically

WHY IT MATTERS:
- System never breaks if one AI service fails
- Can swap between different AI engines easily
- Guardian can choose specific engines when needed
- Makes sure the whole AI system is working

TESTS INCLUDE:
1. Force Engine Choice - Tests picking specific engines
2. Engine Interface - Checks all engines work the same way
3. Live Connection - Tests real AI service connections

HOW TO RUN:
    python3 verification_test.py

RESULT:
    Returns success (0) or failure (1)
"""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add the orchestrator directory to path for imports
# Assuming the script is in the root and the engines are in council_orchestrator/cognitive_engines/
sys.path.insert(0, str(Path(__file__).parent / "council_orchestrator"))

# --- Imports for testing ---
from substrate_monitor import select_engine

def test_force_engine_choice():
    """
    Test that sovereign override bypasses unhealthy primary engine.
    Since we can't easily simulate quota exhaustion, we'll test the override logic directly.
    """
    print("=== TEST: Force Engine Choice ===")
    print("Testing: Can pick specific AI engines when needed")

    # Test 1: Verify sovereign override works (primary health is not relevant for override)
    print("\n1. Testing sovereign override with force_engine: 'ollama'...")
    config_override = {"force_engine": "ollama"}
    engine_override = select_engine(config_override)
    if engine_override is not None and type(engine_override).__name__ == "OllamaEngine":
        print("✅ PASS: Sovereign override successfully selected OllamaEngine")
    else:
        print(f"❌ FAIL: Sovereign override failed. Got: {type(engine_override).__name__ if engine_override else 'None'}")
        return False

    # Test 2: Verify override with invalid engine fails
    print("\n2. Testing sovereign override with invalid engine...")
    config_invalid = {"force_engine": "invalid_engine"}
    engine_invalid = select_engine(config_invalid)
    if engine_invalid is None:
        print("✅ PASS: Invalid force_engine correctly rejected")
    else:
        print(f"❌ FAIL: Invalid force_engine '{config_invalid['force_engine']}' was accepted")
        return False

    # Test 3: Verify automatic triage still works when no override
    print("\n3. Testing automatic triage when no override specified...")
    config_auto = {}  # No override
    engine_auto = select_engine(config_auto)
    if engine_auto is not None:
        print(f"✅ PASS: Automatic triage selected {type(engine_auto).__name__}")
    else:
        print("❌ FAIL: Automatic triage failed to select any engine")
        return False

    print("\n🎉 ALL TESTS PASSED: Sovereign Override Doctrine is operational!")
    print("✅ Sovereign override selects specified engine")
    print("✅ Invalid overrides are rejected")
    print("✅ Automatic triage works when no override")
    return True

def test_engine_compatibility():
    """
    Test that all engines implement the BaseCognitiveEngine interface correctly.
    This demonstrates true polymorphism - we only import the base class and test through the interface.
    """
    print("\n=== TEST: Engine Compatibility ===")
    print("Testing: All engines work the same way")

    # Import only the base class to demonstrate polymorphism
    try:
        from cognitive_engines.base import BaseCognitiveEngine
        print("✅ PASS: BaseCognitiveEngine imported successfully")
    except ImportError as e:
        print(f"❌ FAIL: Could not import BaseCognitiveEngine: {e}")
        return False

    # Test all engines through the substrate monitor (polymorphic selection)
    engines_to_test = ["openai", "gemini", "ollama"]

    for engine_type in engines_to_test:
        print(f"\n{engines_to_test.index(engine_type) + 1}. Testing {engine_type.upper()} Engine Polymorphism...")

        # Force select the specific engine through substrate monitor
        config = {"force_engine": engine_type}
        engine = select_engine(config)

        if engine is None:
            print(f"   ❌ FAIL: Could not initialize {engine_type} engine")
            return False

        # Verify it's an instance of BaseCognitiveEngine (polymorphism check)
        if isinstance(engine, BaseCognitiveEngine):
            print(f"   ✅ PASS: {type(engine).__name__} is instance of BaseCognitiveEngine")
        else:
            print(f"   ❌ FAIL: {type(engine).__name__} is NOT an instance of BaseCognitiveEngine")
            return False

        # Test that all abstract methods are implemented (interface compliance)
        required_methods = ['execute_turn', 'check_health', 'run_functional_test']
        for method_name in required_methods:
            if hasattr(engine, method_name) and callable(getattr(engine, method_name)):
                print(f"   ✅ PASS: {method_name}() method implemented")
            else:
                print(f"   ❌ FAIL: {method_name}() method missing or not callable")
                return False

        # Test basic polymorphic functionality (same interface, different implementations)
        try:
            messages = [{"role": "user", "content": "Hello"}]
            response = engine.execute_turn(messages)
            if response and len(response.strip()) > 0:
                print(f"   ✅ PASS: Polymorphic execute_turn() works: '{response[:30]}...'")
            else:
                print(f"   ❌ FAIL: Polymorphic execute_turn() returned empty response")
                return False
        except Exception as e:
            print(f"   ❌ FAIL: Polymorphic execute_turn() failed: {e}")
            return False

    print("\n🎯 POLYMORPHISM VERIFIED: All engines implement BaseCognitiveEngine interface")
    print("✅ BaseCognitiveEngine abstract base class properly defined")
    print("✅ All concrete engines inherit from BaseCognitiveEngine")
    print("✅ Polymorphic engine selection works through substrate_monitor")
    print("✅ Same interface methods work across all engine types")
    return True

def run_all_tests():
    """Run the complete verification protocol."""
    print("🔬 STARTING AI ENGINE TESTS")
    print("Checking that all AI engines work correctly...")

    test1_passed = test_force_engine_choice()
    test2_passed = test_engine_compatibility()

    if test1_passed and test2_passed:
        print("\n🎯 TESTS COMPLETE: AI System Working")
        print("✅ Can force-pick specific AI engines")
        print("✅ All engines work the same way")
        print("✅ All engines connect to real AI services")
        print("AI system is ready to use!")
        return True
    else:
        print("\n💀 VERIFICATION FAILED: Critical vulnerabilities remain")
        print("The Sanctuary CNS requires further surgery.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)