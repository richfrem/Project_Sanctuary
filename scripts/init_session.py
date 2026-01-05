#!/usr/bin/env python3
"""
Agent Session Initialization Script (Protocol 118)

This script automates the mandatory session initialization sequence for all agents:
1. Verifies Git status (clean working directory).
2. Executes Cortex Guardian Wakeup (HOLISTIC mode).
3. Hydrates the Context Strategy (Guardian Briefing).

Usage:
    python scripts/init_session.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp_servers.rag_cortex.operations import CortexOperations
    from mcp_servers.git.git_ops import GitOperations
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print("Ensure you are running from the project root and requirements are installed.")
    sys.exit(1)

def main():
    print("\n🛡️  PROTOCOL 118: AGENT SESSION INITIALIZATION 🛡️\n")
    
    # 1. Initialize Operations
    try:
        git_ops = GitOperations(str(project_root))
        cortex_ops = CortexOperations(str(project_root))
    except Exception as e:
        print(f"❌ Failed to initialize operations: {e}")
        if "connect to a Chroma server" in str(e) or "Connection refused" in str(e):
             print("\n💡 HINT: ChromaDB does not appear to be running.")
             print("   Ensure the 'sanctuary_vector_db' container is active.")
             print("   Run: podman run -d --name sanctuary_vector_db -p 8000:8000 chromadb/chroma  (Example)")
             print("   See docs/architecture/mcp/servers/rag_cortex/SETUP.md for full instructions.")
        sys.exit(1)

    # 2. Git Safety Check
    print("1️⃣  Verifying Git State...")
    try:
        status = git_ops.status()
        print(f"   Current Branch: {status['branch']}")
        
        if status['modified'] or status['staged']:
            print("   ⚠️  WARNING: You have uncommitted changes.")
            # We don't block, but we warn heavily per Protocol 118 (Context Integrity)
        else:
            print("   ✅ Working directory clean.")
            
    except Exception as e:
        print(f"   ⚠️  Git check skipped/failed: {e}")

    # 3. Guardian Wakeup (HOLISTIC)
    print("\n2️⃣  Executing Guardian Wakeup (HOLISTIC Mode)...")
    try:
        response = cortex_ops.guardian_wakeup(mode="HOLISTIC")
        
        if response.status == "success":
            print(f"   ✅ Context Synthesized.")
            print(f"   📂 Briefing: {response.digest_path}")
            print(f"   ⚡ Latency: {response.total_time_ms}ms")
            
            # Read and display the briefing summary (Strategic Signal)
            briefing_path = Path(response.digest_path)
            if briefing_path.exists():
                print("\n   --- STRATEGIC SIGNAL (Preview) ---")
                with open(briefing_path, 'r') as f:
                    lines = f.readlines()
                    # Print first 15 lines for immediate context
                    for line in lines[:15]:
                        print(f"   {line.strip()}")
                print("   ----------------------------------")
        else:
            print(f"   ❌ Guardian Wakeup Failed: {response.error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Critical Error during Wakeup: {e}")
        sys.exit(1)

    print("\n✅ SESSION INITIALIZED.")
    print("👉 ACTION REQUIRED: Read the full 'guardian_boot_digest.md' before acting.\n")

if __name__ == "__main__":
    main()
