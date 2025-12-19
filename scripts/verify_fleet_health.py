#!/usr/bin/env python3
"""
Fleet Connectivity Pulse Check (Phase 1 of Protocol 125 Validation)
Verifies health endpoints for all 6 active containers.
"""
import httpx
import asyncio
import sys
from typing import Dict, Tuple

# Container Health Endpoints (Internal Docker Network DNS would be used inside, 
# but for verification script on host we use localhost ports)
CONTAINERS = {
    "sanctuary-utils": "http://localhost:8100/health",
    "sanctuary-filesystem": "http://localhost:8101/health",
    "sanctuary-network": "http://localhost:8102/health",
    "sanctuary-git": "http://localhost:8103/health",
    "sanctuary-cortex": "http://localhost:8104/health",
    "sanctuary-domain": "http://localhost:8105/health", # Might be /sse depending on FastMCP
    # "sanctuary-vector-db": "http://localhost:8000/api/v1/heartbeat", # External to Fleet
    # "sanctuary-ollama-mcp": "http://localhost:11434/" # External to Fleet
}

async def check_endpoint(name: str, url: str) -> Tuple[str, bool, str]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return name, True, f"OK ({response.status_code})"
            else:
                return name, False, f"Failed ({response.status_code})"
        except Exception as e:
            return name, False, f"Error: {str(e)}"

async def main():
    print("🏥 Starting Fleet Pulse Check...")
    print(f"Targeting {len(CONTAINERS)} containers...\n")
    
    results = await asyncio.gather(*(check_endpoint(name, url) for name, url in CONTAINERS.items()))
    
    success_count = 0
    for name, success, msg in results:
        icon = "✅" if success else "🔴"
        print(f"{icon} {name:<25} {msg}")
        if success:
            success_count += 1
            
    print(f"\n📊 System Status: {success_count}/{len(CONTAINERS)} Operational")
    
    if success_count == len(CONTAINERS):
        print("🚀 Fleet is FULLY OPERATIONAL. Proceed to Phase 2 (RAG Qualification).")
        sys.exit(0)
    else:
        print("⚠️  Fleet shows DEGRADED performance. Check container logs.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
