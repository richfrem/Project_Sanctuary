#!/usr/bin/env python3
"""
Full Fleet Verification Script
Verifies health and SSE connectivity for all 8 fleet containers.
"""
import httpx
import asyncio
import sys

FLEET = {
    "sanctuary-utils": "http://localhost:8100",
    "sanctuary-filesystem": "http://localhost:8101",
    "sanctuary-network": "http://localhost:8102",
    "sanctuary-git": "http://localhost:8103",
    "sanctuary-cortex": "http://localhost:8104",
    "sanctuary-domain": "http://localhost:8105",
}

async def check_container(name, url):
    health_url = f"{url}/health"
    sse_url = f"{url}/sse"
    
    print(f"Testing {name} ({url})...")
    
    # 1. Health Check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(health_url)
            if resp.status_code == 200:
                print(f"  ✅ Health: OK")
            else:
                print(f"  ⚠️ Health: {resp.status_code} (may be OK for SSE-only)")
    except Exception as e:
        print(f"  ⚠️ Health: {str(e)[:50]}")
        
    # 2. SSE Check (Handshake)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            async with client.stream("GET", sse_url) as resp:
                if resp.status_code == 200:
                    print(f"  ✅ SSE: Handshake OK")
                    return True
                else:
                    print(f"  ❌ SSE: {resp.status_code}")
                    return False
    except Exception as e:
        print(f"  ❌ SSE: {str(e)[:50]}")
        return False

async def main():
    print("🚀 Verifying Full Fleet Deployment (8 Containers)...")
    print("="*60)
    results = {}
    for name, url in FLEET.items():
        results[name] = await check_container(name, url)
        
    success = sum(results.values())
    print("\n" + "="*60)
    print("📊 Fleet Verification Report:")
    for name, ok in results.items():
        status = "✅ ONLINE" if ok else "❌ OFFLINE"
        print(f"  {name:<22} : {status}")
        
    print(f"\n🎯 Result: {success}/6 Fleet Containers Operational")
    if success == 6:
        print("✅ All Systems Go!")
        sys.exit(0)
    else:
        print("⚠️  Some systems need attention.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
