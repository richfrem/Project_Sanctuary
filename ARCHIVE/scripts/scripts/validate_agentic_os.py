#!/usr/bin/env python3
"""
E2E Validation for Agentic OS Integration (Phase 8)
Confirms:
1. Kernel can acquire/release locks
2. Events can be emitted to events.jsonl
3. update_memory.py hook logic works (simulated)
4. State updates persist
"""
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = "/Users/richardfremmerlid/Projects/Project_Sanctuary"
KERNEL = os.path.join(PROJECT_ROOT, "context/kernel.py")
UPDATE_MEMORY = os.path.join(PROJECT_ROOT, ".agents/hooks/update_memory.py")

def run_kernel(args):
    cmd = ["python3", KERNEL] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def test_kernel_locks():
    print("\n--- Testing Kernel Locks ---")
    # Acquire
    run_kernel(["acquire_lock", "test_lock"])
    lock_file = os.path.join(PROJECT_ROOT, "context/.locks/test_lock.lock")
    if os.path.exists(lock_file):
        print("PASS: Lock file exists")
    else:
        print("FAIL: Lock file missing")
    
    # Release
    run_kernel(["release_lock", "test_lock"])
    if not os.path.exists(lock_file):
        print("PASS: Lock file removed")
    else:
        print("FAIL: Lock file still exists")

def test_events():
    print("\n--- Testing Event Bus ---")
    run_kernel(["emit_event", "--agent", "sanctuary-guardian", "--type", "intent", "--action", "validation_test", "--summary", "Smoke test for E2E flow"])
    
    events_file = os.path.join(PROJECT_ROOT, "context/events.jsonl")
    if os.path.exists(events_file):
        with open(events_file, "r") as f:
            lines = f.readlines()
            if any("validation_test" in line for line in lines):
                print("PASS: Event found in events.jsonl")
            else:
                print("FAIL: Event missing from events.jsonl")
    else:
        print("FAIL: events.jsonl missing")

def test_hook_simulation():
    print("\n--- Testing Hook Simulation ---")
    # Simulate a file write
    dummy_file = os.path.join(PROJECT_ROOT, "temp/smoke_test.txt")
    with open(dummy_file, "w") as f:
        f.write("Validation test content")
    
    # Run the hook manually
    # The hook script uses CLAUDE_PROJECT_DIR and reads tool events if available.
    # In this simulation, we'll just check if it runs without error.
    os.environ["CLAUDE_PROJECT_DIR"] = PROJECT_ROOT
    result = subprocess.run(["python3", UPDATE_MEMORY], capture_output=True, text=True)
    print(f"Hook exit code: {result.returncode}")
    if result.returncode == 0:
        print("PASS: update_memory.py ran successfully")
    else:
        print(f"FAIL: Hook error: {result.stderr}")

def test_state():
    print("\n--- Testing State Updates ---")
    run_kernel(["state_update", "validation_status", "passed"])
    state_file = os.path.join(PROJECT_ROOT, "context/os-state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            import json
            state = json.load(f)
            if state.get("validation_status") == "passed":
                print("PASS: State update persisted")
            else:
                print("FAIL: State mismatch")
    else:
        print("FAIL: os-state.json missing")

if __name__ == "__main__":
    test_kernel_locks()
    test_events()
    test_hook_simulation()
    test_state()
    print("\nE2E Smoke Test Complete")
