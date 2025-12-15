"""
Forge LLM MCP E2E Tests - Protocol Verification
===============================================

Verifies all tools via JSON-RPC protocol against the real Forge LLM server.
Requires Ollama Service properly configured.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/forge_llm/e2e/test_forge_llm_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                          | Type  | Description                    |
|-------------------------------|-------|--------------------------------|
| check_sanctuary_model_status  | READ  | Check model availability       |
| query_sanctuary_model         | WRITE | Query LLM                      |

"""
import pytest
import json
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestForgeLLME2E(BaseE2ETest):
    SERVER_NAME = "forge_llm"
    SERVER_MODULE = "mcp_servers.forge_llm.server"

    def test_forge_llm_lifecycle(self, mcp_client):
        """Test cycle: Check Status -> Query Model -> Validate Response"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "check_sanctuary_model_status" in names
        assert "query_sanctuary_model" in names

        # 2. Check Model Status
        status_res = mcp_client.call_tool("check_sanctuary_model_status", {})
        status_text = status_res.get("content", [])[0]["text"]
        print(f"📡 Status: {status_text}")
        
        # Validate status content
        assert "status" in status_text.lower() or "ready" in status_text.lower() or "available" in status_text.lower()

        # 3. Query Model (Simple Query)
        prompt = "Hello, confirm you are operational."
        
        print("\n🧠 Querying Sanctuary Model...")
        try:
            query_res = mcp_client.call_tool("query_sanctuary_model", {
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.1
            })
            query_text = query_res.get("content", [])[0]["text"]
            print(f"🤖 Response: {query_text}")
            
            assert len(query_text) > 0
            
            # Check for JSON structure if applicable, or just length
            # Some implementations return raw text, others JSON.
            if query_text.strip().startswith("{"):
                data = json.loads(query_text)
                assert "response" in data or "content" in data or "text" in data
                print("   ✅ Valid JSON response")
            else:
                 print("   ✅ Text response received")
                 
        except Exception as e:
            # If Ollama is down, this might fail. We should report it but maybe not fail format check?
            # E2E implies environment IS ready. So failure is expected if backend is down.
            pytest.fail(f"Query failed (Ollama likely down): {e}")
