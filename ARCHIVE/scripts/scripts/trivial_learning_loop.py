#!/usr/bin/env python3
"""
Trivial Learning Loop Validation Script
Simulates an agent discovering a fact and verifies the autonomous promotion via hooks.
"""
import os
import sys
import subprocess
import time

PROJECT_ROOT = os.getcwd()
KERNEL_PATH = os.path.join(PROJECT_ROOT, "context", "kernel.py")
HOOK_PATH = os.path.join(PROJECT_ROOT, ".agents", "hooks", "update_memory.py")
MEMORY_DIR = os.path.join(PROJECT_ROOT, "context", "memory")
LEDGER_FILE = os.path.join(PROJECT_ROOT, "context", "memory.md")

def run_learning_loop():
    print("--- Starting Trivial Learning Loop Validation ---")
    
    # 1. Simulate Agent discovering a fact
    session_id = f"test_{int(time.time())}"
    journal_path = os.path.join(MEMORY_DIR, f"{time.strftime('%Y-%m-%d')}_{session_id}.md")
    
    fact_content = f"""# Learning Session {session_id}

## [Fact]
The Agentic OS kernel uses atomic file locks in `context/.locks/` for concurrency safety.
"""
    
    print(f"1. Simulating fact discovery in {journal_path}...")
    with open(journal_path, "w") as f:
        f.write(fact_content)
    
    # 2. Run the update_memory hook (Simulating PostToolUse or SessionEnd)
    print("2. Invoking update_memory hook...")
    try:
        # We simulate the event bus trigger
        subprocess.run([sys.executable, HOOK_PATH], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Hook failed: {e.stderr}")
        sys.exit(1)
        
    # 3. Verify promotion to context/memory.md
    print("3. Verifying fact promotion to memory.md...")
    if not os.path.exists(LEDGER_FILE):
        print("Error: memory.md does not exist.")
        sys.exit(1)
        
    with open(LEDGER_FILE, "r") as f:
        content = f.read()
        if "uses atomic file locks" in content:
            print("SUCCESS: Fact was promoted to the ledger!")
            # Find the ID
            import re
            match = re.search(r"\[#(\d+)\]", content)
            if match:
                print(f"Verified Fact ID: [# {match.group(1)}]")
        else:
            print("FAILURE: Fact not found in memory.md.")
            sys.exit(1)

    print("--- Validation Complete ---")

if __name__ == "__main__":
    run_learning_loop()
