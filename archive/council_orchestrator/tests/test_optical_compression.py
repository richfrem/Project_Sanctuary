#!/usr/bin/env python3
"""
Verification test for Optical Compression functionality in orchestrator.py v4.1
Tests the Optical Decompression Chamber integration per DIRECTIVE_FORGE_ORCHESTRATOR_V4_1.md
"""

import json
import time
from pathlib import Path

def test_optical_compression_enabled():
    """
    Test that optical compression is triggered when enabled in config.
    This validates the Optical Decompression Chamber integration.
    """
    print("\n" + "="*80)
    print("TEST: Optical Compression Enabled")
    print("="*80)
    
    # Create a command with optical compression enabled
    command = {
        "task_description": "This is a test task with a very long context that should trigger optical compression. " * 100,  # Large payload
        "output_artifact_path": "WORK_IN_PROGRESS/TEST_OPTICAL_COMPRESSION/",
        "config": {
            "max_rounds": 1,
            "force_engine": "openai",
            "enable_optical_compression": True,
            "optical_compression_threshold": 1000,  # Low threshold to ensure trigger
            "vlm_engine": "mock"
        }
    }
    
    # Write command file
    command_path = Path(__file__).parent / "command.json"
    with open(command_path, 'w') as f:
        json.dump(command, f, indent=2)
    
    print(f"✓ Command file created: {command_path}")
    print(f"✓ Optical compression: ENABLED")
    print(f"✓ Threshold: {command['config']['optical_compression_threshold']} tokens")
    print(f"✓ Expected behavior: Should see '[OPTICAL] Compressing payload...' in logs")
    print("\nWaiting for orchestrator to process command...")
    print("Monitor the orchestrator logs for optical compression messages.")
    
    return True

def test_optical_compression_disabled():
    """
    Test that system falls back to v4.0 distillation when optical compression is disabled.
    This validates backward compatibility.
    """
    print("\n" + "="*80)
    print("TEST: Optical Compression Disabled (Backward Compatibility)")
    print("="*80)
    
    # Create a command with optical compression disabled
    command = {
        "task_description": "This is a test task that should use standard v4.0 distillation logic.",
        "output_artifact_path": "WORK_IN_PROGRESS/TEST_STANDARD_DISTILLATION/",
        "config": {
            "max_rounds": 1,
            "force_engine": "openai",
            "enable_optical_compression": False  # Explicitly disabled
        }
    }
    
    # Write command file
    command_path = Path(__file__).parent / "command.json"
    with open(command_path, 'w') as f:
        json.dump(command, f, indent=2)
    
    print(f"✓ Command file created: {command_path}")
    print(f"✓ Optical compression: DISABLED")
    print(f"✓ Expected behavior: Should use standard v4.0 distillation path")
    print("\nWaiting for orchestrator to process command...")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPTICAL COMPRESSION VERIFICATION TEST SUITE")
    print("orchestrator.py v4.1 - Operation: Optical Anvil")
    print("="*80)
    
    print("\nThis test suite validates:")
    print("1. Optical Decompression Chamber initialization")
    print("2. Optical compression decision logic")
    print("3. Backward compatibility with v4.0 distillation")
    
    print("\n" + "-"*80)
    print("INSTRUCTIONS:")
    print("-"*80)
    print("1. Ensure orchestrator.py v4.1 is running")
    print("2. Run this script to generate test commands")
    print("3. Monitor orchestrator logs for optical compression messages")
    print("4. Verify task_log.md artifacts are generated successfully")
    
    choice = input("\nSelect test:\n1. Optical Compression Enabled\n2. Optical Compression Disabled\n3. Both\n\nChoice (1/2/3): ")
    
    if choice == "1":
        test_optical_compression_enabled()
    elif choice == "2":
        test_optical_compression_disabled()
    elif choice == "3":
        print("\nRunning Test 1...")
        test_optical_compression_enabled()
        time.sleep(2)
        print("\n\nRunning Test 2...")
        test_optical_compression_disabled()
    else:
        print("Invalid choice. Exiting.")
    
    print("\n" + "="*80)
    print("Test command(s) generated. Monitor orchestrator logs for results.")
    print("="*80 + "\n")