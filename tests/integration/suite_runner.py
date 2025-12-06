#!/usr/bin/env python3
"""
Integration Suite Runner
Executes comprehensive Python-level integration tests for the Sanctuary ecosystem.
These tests verify logical chains (Council -> Agent -> Forge) bypassing the MCP transport layer.
"""
import sys
import unittest
import argparse
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def run_suite(scenarios=None):
    """Run specified test scenarios."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Define available scenarios
    all_scenarios = {
        "forge": "tests.integration.test_chain_forge_ollama",
        "agent": "tests.integration.test_chain_agent_forge",
        "council": "tests.integration.test_chain_council_agent",
    }
    
    # Filter scenarios
    if not scenarios:
        scenarios = all_scenarios.keys()
        
    print(f"🚀 Running Integration Scenarios: {', '.join(scenarios)}")
    print("="*60)
    
    for key in scenarios:
        if key in all_scenarios:
            try:
                module_name = all_scenarios[key]
                print(f"\n🔍 Loading Scenario: {key} ({module_name})")
                tests = loader.loadTestsFromName(module_name)
                suite.addTests(tests)
            except Exception as e:
                print(f"❌ Failed to load {key}: {e}")
        else:
            print(f"⚠️ Unknown scenario: {key}")
            
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanctuary Integration Suite Runner")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--forge", action="store_true", help="Run Forge -> Ollama tests")
    parser.add_argument("--agent", action="store_true", help="Run Agent -> Forge tests")
    parser.add_argument("--council", action="store_true", help="Run Council -> Agent tests")
    
    args = parser.parse_args()
    
    selected = []
    if args.all:
        selected = None # Run all
    else:
        if args.forge: selected.append("forge")
        if args.agent: selected.append("agent")
        if args.council: selected.append("council")
        
    if not selected and not args.all:
        print("Please specify a scenario: --all, --forge, --agent, --council")
        sys.exit(1)
        
    success = run_suite(selected)
    sys.exit(0 if success else 1)
