import os
import httpx
import json

# Test /admin/tools endpoint and pagination
token = ""
dotenv_path = os.path.expanduser("~/Projects/Project_Sanctuary/.env")
if os.path.exists(dotenv_path):
    with open(dotenv_path, "r") as f:
        for line in f:
            if "MCPGATEWAY_BEARER_TOKEN" in line:
                token = line.strip().partition('=')[2].strip('"').strip("'")

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

try:
    with httpx.Client(verify=False, http2=False) as client:
        # Test 1: Standard /tools with explicit limit
        print("=== Standard /tools ===")
        r = client.get("https://localhost:4444/tools", params={"limit": 100}, headers=headers)
        print(f"Status: {r.status_code}, Count: {len(r.json()) if r.status_code == 200 else 'N/A'}")
        
        # Test 2: Admin endpoint
        print("\n=== Admin /admin/tools ===")
        r = client.get("https://localhost:4444/admin/tools", params={"limit": 100}, headers=headers)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "data" in data:
                print(f"Count: {len(data['data'])}")
                print(f"Pagination: {data.get('pagination', 'N/A')}")
            else:
                print(f"Count: {len(data)}")
        else:
            print(f"Response: {r.text[:200]}")
            
        # Test 3: Try /api/tools
        print("\n=== /api/tools ===")
        r = client.get("https://localhost:4444/api/tools", params={"limit": 100}, headers=headers)
        print(f"Status: {r.status_code}, Response: {r.text[:100] if r.status_code != 200 else len(r.json())}")
        
except Exception as e:
    print(f"Error: {e}")
