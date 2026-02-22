"""
RAG Cortex MCP E2E Tests - Protocol Verification
================================================

Verifies all tools via JSON-RPC protocol against the real RAG Cortex server.
Requires live Vector DB (Chroma) and Ollama.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/rag_cortex/e2e/test_cortex_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                      | Type  | Description              |
|---------------------------|-------|--------------------------|
| cortex_get_stats          | READ  | DB stats                 |
| cortex_ingest_incremental | WRITE | Ingest file              |
| cortex_query              | READ  | Semantic search          |
| cortex_cache_set          | WRITE | Cache operation          |
| cortex_cache_get          | READ  | Cache retrieval          |

"""
import pytest
import os
import json
import time
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

from mcp_servers.lib.path_utils import find_project_root
PROJECT_ROOT = Path(find_project_root())

@pytest.mark.e2e
class TestRAGCortexE2E(BaseE2ETest):
    SERVER_NAME = "rag_cortex"
    SERVER_MODULE = "mcp_servers.rag_cortex.server"

    def test_cortex_lifecycle(self, mcp_client):
        """Test full cycle: Stats -> Ingest -> Query -> Cache"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "cortex_get_stats" in names
        assert "cortex_ingest_incremental" in names

        # 2. Get Stats
        stats_res = mcp_client.call_tool("cortex_get_stats", {})
        stats_text = stats_res.get("content", [])[0]["text"]
        print(f"📊 cortex_get_stats: {stats_text}")
        # Validate output format (JSON string or text)
        # Usually JSON string
        assert "documents" in stats_text.lower() or "stats" in stats_text.lower()

        # 3. Ingest File
        # Create temp file in valid ingestion directory
        test_file = PROJECT_ROOT / "00_CHRONICLE" / "e2e_cortex_test.md"
        # Ensure dir exists
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_file, "w") as f:
            f.write("# E2E Cortex Test\n\nThis is a unique string: CortexE2E verification sequence.")
            
        try:
            ingest_res = mcp_client.call_tool("cortex_ingest_incremental", {
                "request": {
                    "file_paths": [str(test_file)],
                    "skip_duplicates": False
                }
            })
            ingest_text = ingest_res.get("content", [])[0]["text"]
            print(f"📥 cortex_ingest_incremental: {ingest_text}")
            assert "success" in ingest_text.lower() or "processed" in ingest_text.lower()

            # 4. Query
            # Small delay for persistence if needed
            # time.sleep(1)
            
            query_res = mcp_client.call_tool("cortex_query", {
                "request": {
                    "query": "CortexE2E verification sequence",
                    "max_results": 1
                }
            })
            query_text = query_res.get("content", [])[0]["text"]
            print(f"🔍 cortex_query: {query_text}")
            
            # Check if found
            assert "CortexE2E" in query_text

            # 5. Cache Operations
            cache_res = mcp_client.call_tool("cortex_cache_set", {
                "request": {
                    "query": "E2E Cache Key",
                    "answer": "Cached Answer 123"
                }
            })
            print(f"💾 cortex_cache_set: {cache_res.get('content', [])[0]['text']}")
            
            get_cache = mcp_client.call_tool("cortex_cache_get", {"request": {"query": "E2E Cache Key"}})
            get_text = get_cache.get("content", [])[0]["text"]
            print(f"📂 cortex_cache_get: {get_text}")
            assert "Cached Answer" in get_text

        finally:
            if test_file.exists():
                os.remove(test_file)
                print(f"🧹 Cleaned up {test_file}")
