"""
RAG Cortex MCP Integration Tests - Operations Testing
=====================================================

Comprehensive integration tests for all RAG Cortex operations.
Uses BaseIntegrationTest to ensure robust setup/teardown of real services.

MCP OPERATIONS:
---------------
| Operation            | Type | Description                          |
|----------------------|------|--------------------------------------|
| cortex_ingest_full   | WRITE| Full ingest of documents             |
| cortex_query         | READ | Query the vector database            |
| cortex_cache_set     | WRITE| Set memory cache                     |
| cortex_cache_get     | READ | Get memory cache                     |
| cortex_cache_warmup  | WRITE| Warmup cache from ChromaDB           |
| cortex_guardian_wakeup| READ| Generate context briefing            |
| cortex_get_stats     | READ | Get database statistics              |

"""
import pytest
import os
import time
import chromadb
from langchain_chroma import Chroma
from pathlib import Path

from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.lib.utils.env_helper import get_env_variable


class TestCortexOperations(BaseIntegrationTest):
    """
    Integration tests for all RAG Cortex operations.
    Connects to REAL ChromaDB and Ollama services.
    """
    
    def get_required_services(self):
        chroma_host = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
        chroma_port = int(get_env_variable("CHROMA_PORT", required=False) or "8110")
        
        # Ollama is NOT required for RAG operations (Local Nomic used)
        # But it is required for Forge/Reasoning tools in the same server.
        # For this specific test suite, we only focus on Cortex (RAG).
        
        return [
            (chroma_host, chroma_port, "ChromaDB")
        ]

    @pytest.fixture
    def cortex_ops(self, tmp_path):
        """Initialize real CortexOperations connected to real ChromaDB with ISOLATED collections."""
        # Use a temporary directory for file storage to avoid polluting real data
        project_root = tmp_path / "project_root"
        project_root.mkdir()
        
        # Create necessary subdirectories
        chroma_host = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
        chroma_port = get_env_variable("CHROMA_PORT", required=False) or "8110"
        
        env_content = f"CHROMA_HOST={chroma_host}\nCHROMA_PORT={chroma_port}\n"
        (project_root / ".env").write_text(env_content)
        (project_root / "00_CHRONICLE").mkdir()
        (project_root / "01_PROTOCOLS").mkdir()
        (project_root / "TASKS").mkdir()
        
        # Connect to REAL ChromaDB (local server)
        host = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
        port = int(get_env_variable("CHROMA_PORT", required=False) or "8110")
        client = chromadb.HttpClient(host=host, port=port)
        
        ops = CortexOperations(str(project_root), client=client)
        
        # OVERRIDE collections to test-specific ones to avoid wrecking production data
        timestamp = int(time.time())
        ops.child_collection_name = f"test_child_{timestamp}"
        ops.parent_collection_name = f"test_parent_{timestamp}"
        
        # Re-init vectorstore with new collection name
        ops.vectorstore = Chroma(
            client=client,
            collection_name=ops.child_collection_name,
            embedding_function=ops.embedding_model
        )
        
        yield ops
        
        # TEARDOWN: Delete test collections
        try:
            client.delete_collection(ops.child_collection_name)
            # Parent documents are in the FileStore (folder), not a Chroma collection
            # The cleanup of the tmp_path/project_root is handled by pytest
        except Exception as e:
            print(f"Warning: Failed to cleanup test collections: {e}")

    def test_chroma_connectivity(self, cortex_ops):
        """Validate we can talk to ChromaDB."""
        heartbeat = cortex_ops.chroma_client.heartbeat()
        assert heartbeat is not None

    def test_nomic_embedding_generation(self, cortex_ops):
        """Validate Nomic (Local) is generating real embeddings."""
        text = "The quick brown fox jumps over the lazy dog."
        embedding = cortex_ops.embedding_model.embed_query(text)
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert isinstance(embedding[0], float)

    def test_ingest_and_query(self, cortex_ops):
        """
        Validate full Ingest -> Store -> Retrieve cycle with REAL components.
        """
        # 1. Create content
        source_dir = cortex_ops.project_root / "00_CHRONICLE"
        (source_dir / "test_doc.md").write_text(
            "# Live Test Document\n\nThis is a live integration test for Protocol 101."
        )
        
        # 2. Ingest
        print("\nRunning cortex_ingest_full...")
        result = cortex_ops.ingest_full(purge_existing=True, source_directories=["00_CHRONICLE"])
        assert result.status == "success"
        assert result.documents_processed == 1
        
        # 3. Query
        print("\nRunning cortex_query...")
        q_result = cortex_ops.query("Protocol 101", max_results=1)
        assert q_result.status == "success"
        assert len(q_result.results) > 0
        assert "Live Test Document" in q_result.results[0].content
        
    def test_cache_operations(self, cortex_ops):
        """Test Memory Cache (CAG) operations."""
        print("\nTesting Cache Operations...")
        
        # 1. Cache Set
        ops = cortex_ops
        query = "What is the meaning of life?"
        answer = "42 - Test Answer"
        
        res = ops.cache_set(query, answer)
        assert res.stored is True
        
        # 2. Cache Get
        cached = ops.cache_get(query)
        assert cached.cache_hit is True
        assert cached.answer == answer
        
        # 3. Cache Miss
        miss = ops.cache_get("Unknown query")
        assert miss.cache_hit is False

    @pytest.mark.skip(reason="Requires pre-existing extensive database content for meaningful warmup")
    def test_cache_warmup(self, cortex_ops):
        """Test Cache Warmup (queries Chroma)."""
        # Populate DB first
        (cortex_ops.project_root / "00_CHRONICLE" / "test.md").write_text("# Test\nContent")
        cortex_ops.ingest_full(source_directories=["00_CHRONICLE"])
        
        # Warmup
        res = cortex_ops.cache_warmup()
        assert res.status == "success"

    def test_guardian_wakeup_basic(self, cortex_ops):
        """Test Guardian Wakeup generation."""
        # Create some basic files to be found
        (cortex_ops.project_root / "WORK_IN_PROGRESS").mkdir(exist_ok=True)
        (cortex_ops.project_root / "TASKS" / "test_task.md").write_text("- [ ] Task 1")
        
        print("\nRunning guardian_wakeup...")
        res = cortex_ops.guardian_wakeup(mode="HOLISTIC")
        
        assert res.status == "success"
        assert res.digest_path is not None
        assert Path(res.digest_path).exists()
        
    def test_get_stats(self, cortex_ops):
        """Test get_stats operation."""
        # 1. Verify empty = error
        res_empty = cortex_ops.get_stats()
        assert res_empty.health_status == "error"

        # 2. Ingest data
        (cortex_ops.project_root / "00_CHRONICLE" / "stats_test.md").write_text("# Stats Test\nContent")
        cortex_ops.ingest_full(source_directories=["00_CHRONICLE"])

        # 3. Verify populated = healthy
        res = cortex_ops.get_stats()
        assert res.health_status == "healthy"
        assert res.collections is not None
        assert res.total_documents > 0


    def test_polyglot_code_ingestion(self, cortex_ops):
        """
        Validate Polyglot (Python + JS) code ingestion and retrieval.
        Tests the shim integration within the full pipeline.
        """
        print("\nTesting Polyglot Code Ingestion...")
        
        # 1. Create Source Files
        src_dir = cortex_ops.project_root / "src"
        src_dir.mkdir()
        
        # Python File (AST Test)
        py_content = '''
class MnemonicValidator:
    """
    A test class to verify AST ingestion.
    """
    def validate_quantum_state(self, coherence: float) -> bool:
        """
        Verifies if the quantum state matches the threshold.
        Target Threshold: 0.99
        """
        if coherence > 0.99:
            return True
        return False
'''
        (src_dir / "physics.py").write_text(py_content, encoding="utf-8")
        
        # JavaScript File (Regex Test)
        js_content = '''
/**
 * Renders the dashboard UI.
 */
function renderDashboard(userId) {
    if (userId === "admin") return true;
    return false;
}
'''
        (src_dir / "ui.js").write_text(js_content, encoding="utf-8")
        
        # 2. Run Full Ingestion
        # Note: We must explicitly pointing to 'src' if we want to limit scope, 
        # or rely on ingest_full scanning everything.
        # ingest_full default excludes nothing? logic scans specific dirs usually.
        # Let's pass source_directories to be precise.
        print("Running ingest_full on src directory...")
        result = cortex_ops.ingest_full(purge_existing=True, source_directories=["src"])
        
        assert result.status == "success"
        # We expect at least 2 code files processed
        assert result.documents_processed >= 2
        
        # 3. Verify Artifacts (Side Effect Check)
        # Shim should produce .py.md and .js.md
        assert (src_dir / "physics.py.md").exists()
        assert (src_dir / "ui.js.md").exists()
        
        # 4. Verify Retrieval (Python)
        print("Querying Python Structure...")
        # Query for class/function concepts
        py_res = cortex_ops.query("How does MnemonicValidator work?", max_results=1)
        assert len(py_res.results) > 0
        content = py_res.results[0].content
        
        # Check for structural headers generated by shim
        has_header = "Class: `MnemonicValidator`" in content or "## Class: MnemonicValidator" in content
        has_docstring = "Target Threshold: 0.99" in content
        
        if not (has_header and has_docstring):
             print(f"DEBUG (Py): Content retrieved:\n{content}")
             
        assert has_header or has_docstring, "Failed to retrieve Python structure/content"
        
        # 5. Verify Retrieval (JS)
        print("Querying JS Structure...")
        js_res = cortex_ops.query("renderDashboard", max_results=1)
        assert len(js_res.results) > 0
        content = js_res.results[0].content
        
        has_js_header = "Function: `renderDashboard`" in content or "## Function: renderDashboard" in content
        has_snippet = "userId === \"admin\"" in content
        
        if not (has_js_header or has_snippet):
             print(f"DEBUG (JS): Content retrieved:\n{content}")
             
        assert has_js_header or has_snippet, "Failed to retrieve JS structure/content"

    def test_python_structural_search(self, cortex_ops):
        """
        Verify that we can search for specific Python structural elements 
        (signatures and docstrings) extracted by the ingest_code_shim.
        """
        print("\nTesting Python Structural Search...")
        
        # 1. Create a specialized Python file
        logic_dir = cortex_ops.project_root / "logic"
        logic_dir.mkdir()
        
        py_code = '''
def handle_mnemonic_cascade(sequence_id: str, intensity_threshold: float = 0.85):
    """
    Handles a high-intensity mnemonic cascade event.
    
    This function specifically looks for the intensity_threshold 
    to trigger Protocol 121.
    """
    if intensity_threshold > 0.85:
        return f"Cascade {sequence_id} active"
    return "Normal"
'''
        (logic_dir / "cascade.py").write_text(py_code, encoding="utf-8")
        
        # 2. Ingest
        cortex_ops.ingest_full(purge_existing=True, source_directories=["logic"])
        
        # 3. Query for specific signature concepts
        print("Searching for specialized signature...")
        # We query for terms present in the signature and docstring
        q_res = cortex_ops.query("function handle_mnemonic_cascade intensity_threshold 0.85", max_results=1)
        
        assert len(q_res.results) > 0
        content = q_res.results[0].content
        
        # 4. Verify the shim worked as expected
        # It should have extracted the signature into a clear header
        assert "handle_mnemonic_cascade" in content
        assert "sequence_id: str" in content
        assert "intensity_threshold: float = 0.85" in content
        assert "Protocol 121" in content
        print_success = lambda m: print(f"✓ {m}")
        print_success("Successfully retrieved Python structural content via shim.")



