import asyncio
import json
import shutil
import sys
from pathlib import Path
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.types import CallToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / ".agent" / "learning" / "learning_manifest.json"
TARGET_OUTPUT = PROJECT_ROOT / ".agent" / "learning" / "learning_package_snapshot.md"
TEMP_OUTPUT = PROJECT_ROOT / ".agent" / "learning" / "red_team" / "red_team_audit_packet.md"

async def run_snapshot():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest_files = json.load(f)
    except Exception as e:
        print(f"Error loading manifest: {e}")
        return

    print("Connecting to Sanctuary Cortex (port 8104)...")
    url = "http://localhost:8104/sse"
    
    async with sse_client(url) as streams:
        async with ClientSession(streams.read, streams.write) as session:
            await session.initialize()
            
            print("Calling cortex_capture_snapshot...")
            # Using 'audit' mode to generate a packet we can move
            result = await session.call_tool(
                "cortex_capture_snapshot",
                arguments={
                    "manifest_files": manifest_files,
                    "snapshot_type": "audit",
                    "strategic_context": "Automated Learning Package Update"
                }
            )
            
            if isinstance(result, CallToolResult):
                print("Tool Result:")
                for content in result.content:
                    print(content.text)
                
                # Check if file was created
                if TEMP_OUTPUT.exists():
                    print(f"Snapshot generated at {TEMP_OUTPUT}")
                    print(f"Updating {TARGET_OUTPUT}...")
                    shutil.copy(TEMP_OUTPUT, TARGET_OUTPUT)
                    print("✅ Learning Package Snapshot Updated Successfully.")
                else:
                    print(f"❌ Error: Expected output file {TEMP_OUTPUT} not found.")
            else:
                 print(f"Unexpected result type: {type(result)}")

if __name__ == "__main__":
    try:
        asyncio.run(run_snapshot())
    except Exception as e:
        print(f"Execution Failed: {e}")
        sys.exit(1)
