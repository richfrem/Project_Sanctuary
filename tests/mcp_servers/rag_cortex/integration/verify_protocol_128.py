#!/usr/bin/env python3
import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = "/Users/richardfremmerlid/Projects/Project_Sanctuary"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.rag_cortex.server import cortex_capture_snapshot, cortex_learning_debrief

async def run_tests():
    print("="*60)
    print("PROTOCOL 128: HARDENED LEARNING LOOP - INTEGRATION VERIFICATION")
    print("="*60)
    
    ops = CortexOperations(PROJECT_ROOT)
    
    # --- TEST 1: Direct Operations Seal ---
    print("\n[TEST 1] Direct Operations: capture_snapshot (type='seal')")
    try:
        # Empty manifest should trigger default manifest loading
        resp = ops.capture_snapshot(manifest_files=[], snapshot_type="seal")
        # Check attributes that now exist on CaptureSnapshotResponse
        print(f"✅ Success: Generated {resp.total_files} files, {resp.total_bytes} bytes")
        print(f"   Artifact: {resp.snapshot_path}")
        print(f"   Manifest Verified: {resp.manifest_verified}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # --- TEST 2: Direct Operations Debrief ---
    print("\n[TEST 2] Direct Operations: learning_debrief")
    try:
        debrief = ops.learning_debrief(hours=1)
        if "Loaded Learning Package Snapshot" in debrief:
            print("✅ Success: Debrief correctly detected the recent seal.")
        else:
            print("⚠️ Warning: Debrief did not find the recent seal (check window/mtime).")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # --- TEST 3: Server Wrapper Seal ---
    print("\n[TEST 3] Server Wrapper: cortex_capture_snapshot (seal)")
    try:
        res_json = await cortex_capture_snapshot(manifest_files=[], snapshot_type="seal")
        res = json.loads(res_json)
        if res.get("status") == "success":
            print(f"✅ Success: Server wrapper executed and returned success status.")
        else:
            print(f"❌ Failed: {res.get('error')}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # --- TEST 4: Server Wrapper Debrief ---
    print("\n[TEST 4] Server Wrapper: cortex_learning_debrief")
    try:
        res_json = await cortex_learning_debrief(hours=1)
        res = json.loads(res_json)
        if res.get("status") == "success" and "debrief" in res:
            print(f"✅ Success: Server wrapper returned valid debrief JSON.")
            # Verify recency check in JSON
            if "✅ Loaded Learning Package Snapshot" in res["debrief"]:
                 print("   Verify: Recency marker found in debrief text.")
        else:
            print(f"❌ Failed: {res.get('error')}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_tests())
