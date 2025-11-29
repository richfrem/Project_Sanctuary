#!/usr/bin/env python3
"""
Mnemonic Cortex Master Verification Harness
Run this script to verify all subsystems: RAG, Cache, Guardian, and Training.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_step(name: str, command: List[str] = None, func=None) -> bool:
    """Run a verification step."""
    print(f"\n--- STEP: {name} ---")
    try:
        if command:
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command, 
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"❌ FAILED with exit code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                print(f"Stdout: {result.stdout}")
                return False
            print("✅ PASSED")
            return True
            
        if func:
            print("Running internal check...")
            func()
            print("✅ PASSED")
            return True
            
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")
        return False
    return False

def check_cache_ops():
    """Verify Cache Get/Set directly."""
    from mnemonic_cortex.core.cache import MnemonicCache
    cache = MnemonicCache()
    
    # Test Set
    test_key = "verify_harness_test"
    test_val = {"status": "verified", "timestamp": "now"}
    cache.set(test_key, test_val)
    
    # Test Get
    retrieved = cache.get(test_key)
    if retrieved != test_val:
        raise ValueError(f"Cache retrieval mismatch. Expected {test_val}, got {retrieved}")
    print(f"Cache verified: {retrieved}")

def check_guardian_wakeup():
    """Verify Guardian Wakeup."""
    from mcp_servers.cognitive.cortex.operations import CortexOperations
    ops = CortexOperations(str(PROJECT_ROOT))
    result = ops.guardian_wakeup()
    print(f"Guardian Wakeup Result: {result}")
    if not result.digest_path:
        raise ValueError("No digest path returned")

def check_adaptation_packet():
    """Verify Adaptation Packet Generation."""
    from mnemonic_cortex.app.synthesis.generator import SynthesisGenerator
    gen = SynthesisGenerator(str(PROJECT_ROOT))
    packet = gen.generate_packet(days=1)
    print(f"Generated packet with {len(packet.examples)} examples")

def main():
    print("============================================================")
    print("   MNEMONIC CORTEX - MASTER VERIFICATION HARNESS")
    print("============================================================")
    
    steps = [
        ("1. Database Health Check", ["python3", "mnemonic_cortex/scripts/inspect_db.py"], None),
        ("2. RAG Query Test", ["python3", "mnemonic_cortex/app/main.py", "What is Protocol 101?"], None),
        ("3. Cache Warmup", ["python3", "mnemonic_cortex/scripts/cache_warmup.py", "--queries", "Protocol 101"], None),
        ("4. Cache Operations (Get/Set)", None, check_cache_ops),
        ("5. Guardian Wakeup", None, check_guardian_wakeup),
        ("6. Adaptation Packet Gen", None, check_adaptation_packet),
        ("7. LoRA Training Dry-Run", [
            "python3", "mnemonic_cortex/scripts/train_lora.py", 
            "--data", "test_data.jsonl", 
            "--output", "adapters/verify_run", 
            "--dry-run"
        ], None)
    ]
    
    # Create dummy data for LoRA test
    with open(PROJECT_ROOT / "test_data.jsonl", "w") as f:
        f.write('{"instruction": "Test", "input": "", "output": "Test"}\n')

    passed = 0
    failed = 0
    
    for name, cmd, func in steps:
        if run_step(name, cmd, func):
            passed += 1
        else:
            failed += 1
            
    print("\n============================================================")
    print(f"VERIFICATION COMPLETE: {passed} PASSED, {failed} FAILED")
    print("============================================================")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
