#!/usr/bin/env python3
"""
Simple test to debug Ollama Python client API
"""

import ollama
import json

try:
    print("Testing Ollama Python client...")
    
    # Test 1: List models
    print("\n1. Testing ollama.list()...")
    models = ollama.list()
    print(f"Response type: {type(models)}")
    print(f"Response: {models}")
    
    # Test 2: Simple chat
    print("\n2. Testing ollama.chat()...")
    response = ollama.chat(
        model='nemotron-nano:latest',
        messages=[
            {'role': 'user', 'content': 'Hello, say hi back in one word.'}
        ]
    )
    print(f"Chat response: {response}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
