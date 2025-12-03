#!/usr/bin/env python3
"""
Test script for Forge MCP server.
Verifies that the query_sanctuary_model tool works correctly.
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from mcp_servers.system.forge.operations import ForgeOperations

def test_model_availability():
    """Test if the Sanctuary model is available."""
    print("=" * 60)
    print("Testing Sanctuary Model Availability")
    print("=" * 60)
    
    forge_ops = ForgeOperations(PROJECT_ROOT)
    result = forge_ops.check_model_availability()
    
    print(f"\nStatus: {result['status']}")
    if result['status'] == 'success':
        print(f"Model: {result['model']}")
        print(f"Available: {result['available']}")
        print(f"\nAll models in Ollama:")
        for model in result.get('all_models', []):
            print(f"  - {model}")
    else:
        print(f"Error: {result.get('error')}")
    
    return result['status'] == 'success' and result.get('available', False)

def test_model_query():
    """Test querying the Sanctuary model."""
    print("\n" + "=" * 60)
    print("Testing Sanctuary Model Query")
    print("=" * 60)
    
    forge_ops = ForgeOperations(PROJECT_ROOT)
    
    prompt = "What is Protocol 101?"
    print(f"\nPrompt: {prompt}")
    print("\nQuerying model...")
    
    response = forge_ops.query_sanctuary_model(
        prompt=prompt,
        temperature=0.7,
        max_tokens=500
    )
    
    print(f"\nStatus: {response.status}")
    if response.status == "success":
        print(f"Model: {response.model}")
        print(f"Temperature: {response.temperature}")
        print(f"Tokens: {response.total_tokens}")
        print(f"\nResponse:\n{response.response}")
    else:
        print(f"Error: {response.error}")
    
    return response.status == "success"

def main():
    """Run all tests."""
    print("\n🔥 Forge MCP Server Test Suite 🔥\n")
    
    # Test 1: Model availability
    availability_ok = test_model_availability()
    
    if not availability_ok:
        print("\n❌ Model not available. Please ensure:")
        print("   1. Ollama is installed and running")
        print("   2. Sanctuary model is loaded: ollama list")
        print("   3. Python ollama package is installed: pip install ollama")
        return 1
    
    # Test 2: Model query
    query_ok = test_model_query()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Model Availability: {'✅ PASS' if availability_ok else '❌ FAIL'}")
    print(f"Model Query: {'✅ PASS' if query_ok else '❌ FAIL'}")
    
    if availability_ok and query_ok:
        print("\n🎉 All tests passed! Forge MCP is ready to use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
