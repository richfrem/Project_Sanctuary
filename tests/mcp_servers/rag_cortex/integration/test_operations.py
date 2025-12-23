"""
RAG Cortex MCP Integration Tests - Operations Testing
=====================================================

Comprehensive integration tests for all RAG Cortex operations.
Uses BaseIntegrationTest to ensure robust setup/teardown of real services.

MCP OPERATIONS:
---------------
| Operation                 | Type | Description                          |
|---------------------------|------|--------------------------------------|
| cortex_ingest_full        | WRITE| Full ingest of documents             |
| cortex_ingest_incremental | WRITE| Incremental ingest of specific files |
| cortex_query              | READ | Query the vector database (RAG)      |
| cortex_cache_set          | WRITE| Set memory cache (CAG)               |
| cortex_cache_get          | READ | Get memory cache (CAG)               |
| cortex_cache_warmup       | WRITE| Warmup cache from ChromaDB           |
| cortex_guardian_wakeup    | READ | Generate context briefing            |
| cortex_get_stats          | READ | Get database statistics              |
| cortex_learning_debrief   | WRITE| Record session debrief (Protocol 128)|

"""
import pytest
import os
import time
import chromadb
import textwrap
from langchain_chroma import Chroma
from pathlib import Path

from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.lib.utils.env_helper import get_env_variable


# ============================================================================
# INTEGRATION TESTING STRATEGY (Layer 2)
# ============================================================================
# These tests validate the interaction between CortexOperations, ChromaDB, 
# and the local filesystem. 
#
# SYNTHETIC TEST FILES:
# We create temporary .py and .js files with known content during the test 
# rather than pointing to real project files. This ensures:
# 1. ISOLATION: Tests don't break when production code changes.
# 2. CONTRACTS: We can verify character-perfect retrieval of specific 
#    signatures, docstrings, and syntax patterns.
# 3. SPEED: We use ingest_incremental() to verify specific language-parsing 
#    logic without rebuilding the entire project index.
# ============================================================================


# Remove the old constants block from the previous step
class TestCortexOperations(BaseIntegrationTest):
    """
    Integration tests for all RAG Cortex operations.
    Connects to REAL ChromaDB and Ollama services.
    """
    
    #===========================================================================
    # INTERNAL: Service Discovery
    # Purpose: Identify mandatory components (ChromaDB) for the test to run.
    #===========================================================================
    def get_required_services(self):
        chroma_host = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
        chroma_port = int(get_env_variable("CHROMA_PORT", required=False) or "8110")
        
        # Ollama is NOT required for RAG operations (Local HuggingFace used)
        # But it is required for Forge/Reasoning tools in the same server.
        # For this specific test suite, we only focus on Cortex (RAG).
        
        return [
            (chroma_host, chroma_port, "ChromaDB")
        ]

    #===========================================================================
    # FIXTURE: cortex_ops
    # Purpose: Provision an isolated, ephemeral RAG environment for each test.
    # Scenarios:
    #   - Generates unique Chroma collection names (test_*)
    #   - Creates temporary workspace directory structure
    #   - Handles cleanup of collections on teardown
    #===========================================================================
    @pytest.fixture
    def cortex_ops(self, tmp_path):
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

    #===========================================================================
    # MCP OPERATIONS: Internal (Chroma Connectivity)
    # Purpose: Verify basic network connectivity to the vector database.
    # Inputs: 
    #   - Chroma HttpClient
    # Scenarios tested:
    #   - Heartbeat check
    #===========================================================================
    def test_chroma_connectivity(self, cortex_ops):
        heartbeat = cortex_ops.chroma_client.heartbeat()
        assert heartbeat is not None

    #===========================================================================
    # MCP OPERATIONS: Internal (Embeddings)
    # Purpose: Verify Local HuggingFace embedding generation.
    # Inputs: 
    #   - Nomic-embed-text-v1.5 model
    # Scenarios tested:
    #   - Vector generation from string
    #   - Embedding dimension validation
    #===========================================================================
    def test_embedding_generation(self, cortex_ops):
        text = "The quick brown fox jumps over the lazy dog."
        embedding = cortex_ops.embedding_model.embed_query(text)
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert isinstance(embedding[0], float)

    #===========================================================================
    # MCP OPERATIONS: cortex_ingest_full & cortex_query
    # Purpose: Verify basic end-to-end RAG pipeline (Layer 2).
    # Inputs: 
    #   - Real Markdown files on disk
    # Scenarios tested:
    #   - Full purge and rebuild
    #   - Semantic retrieval of parent documents
    # Operations tested:
    #   - ingest_full
    #   - query
    #===========================================================================
    def test_query_and_ingest_flow(self, cortex_ops):
        print("\nTesting Full Ingest and Query...")
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
        
    #===========================================================================
    # MCP OPERATIONS: cortex_cache_set & cortex_cache_get
    # Purpose: Verify the Mnemonic Cache (CAG) two-tier system.
    # Inputs: 
    #   - Query string, Answer string
    # Scenarios tested:
    #   - Cache hit after storage
    #   - Cache miss for unknown queries
    #===========================================================================
    def test_cache_operations(self, cortex_ops):
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

    #===========================================================================
    # MCP OPERATIONS: cortex_cache_warmup
    # Purpose: Verify pre-population of cache from Chroma content.
    # Inputs: 
    #   - Existing vector store documents
    # Scenarios tested:
    #   - Warmup execution status
    #===========================================================================
    @pytest.mark.skip(reason="Requires pre-existing extensive database content for meaningful warmup")
    def test_cache_warmup(self, cortex_ops):
        # Populate DB first
        (cortex_ops.project_root / "00_CHRONICLE" / "test.md").write_text("# Test\nContent")
        cortex_ops.ingest_full(source_directories=["00_CHRONICLE"])
        
        # Warmup
        res = cortex_ops.cache_warmup()
        assert res.status == "success"

    #===========================================================================
    # MCP OPERATIONS: cortex_guardian_wakeup
    # Purpose: Verify generation of context-aware digests.
    # Inputs: 
    #   - Workspace tasks and documents
    # Scenarios tested:
    #   - Holistic digest generation
    #   - Markdown digest file creation
    #===========================================================================
    def test_guardian_wakeup_basic(self, cortex_ops):
        # Create some basic files to be found
        (cortex_ops.project_root / "WORK_IN_PROGRESS").mkdir(exist_ok=True)
        (cortex_ops.project_root / "TASKS" / "test_task.md").write_text("- [ ] Task 1")
        
        print("\nRunning guardian_wakeup...")
        res = cortex_ops.guardian_wakeup(mode="HOLISTIC")
        
        assert res.status == "success"
        assert res.digest_path is not None
        assert Path(res.digest_path).exists()
        
    #===========================================================================
    # MCP OPERATIONS: cortex_get_stats
    # Purpose: Verify database health and statistics reporting.
    # Inputs: 
    #   - Chroma collections
    # Scenarios tested:
    #   - Error status on empty DB
    #   - Healthy status and document counts after ingestion
    #===========================================================================
    def test_get_stats(self, cortex_ops):
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

    #===========================================================================
    # MCP OPERATIONS: cortex_ingest_incremental
    # Purpose: Verify that knowledge can be appended without rebuilding DB.
    # Inputs: 
    #   - Single Markdown file
    # Scenarios tested:
    #   - Adding knowledge to an existing collection
    #   - Retrieval of incrementally added content
    #===========================================================================
    def test_ingest_incremental(self, cortex_ops):
        print("\nTesting Incremental Ingestion...")
        
        # 1. Create a new file
        inc_dir = cortex_ops.project_root / "incremental"
        inc_dir.mkdir()
        file_path = inc_dir / "new_knowledge.md"
        file_path.write_text("# Incremental Secret\nThe password for the vault is 'Mnemonic-2025'.")
        
        # 2. Run Incremental Ingest
        res = cortex_ops.ingest_incremental(file_paths=[str(file_path)])
        assert res.status == "success"
        assert res.documents_added == 1
        
        # 3. Verify it's searchable
        q_res = cortex_ops.query("What is the password for the vault?", max_results=1)
        assert len(q_res.results) > 0
        assert "Mnemonic-2025" in q_res.results[0].content
        print("✓ Incremental knowledge successfully retrieved.")

    #===========================================================================
    # MCP OPERATIONS: Internal Code-Ingestion Pipeline (Language Shims)
    # Purpose: Verify that RAG Cortex can handle multiple programming languages.
    # Inputs: 
    #   - Real Python and JS syntax
    # Scenarios tested:
    #   - Python class with method (AST)
    #   - JS function (Regex)
    #===========================================================================
    def test_polyglot_code_ingestion(self, cortex_ops):
        print("\nTesting Polyglot Code Ingestion (Incremental)...")
        
        # 1. Provide Real Code Input
        src_dir = cortex_ops.project_root / "src"
        src_dir.mkdir()
        
        py_path = src_dir / "physics.py"
        py_path.write_text(textwrap.dedent('''
            class MnemonicValidator:
                """A test class for AST ingestion."""
                def validate_quantum_state(self, coherence: float) -> bool:
                    """Verifies coherence. Threshold: 0.99"""
                    return coherence > 0.99
        '''))

        js_path = src_dir / "ui.js"
        js_path.write_text(textwrap.dedent('''
            /** Renders UI. */
            function renderDashboard(userId) {
                return userId === "admin";
            }
        '''))
        
        # 2. Add to ChromaDB via Incremental Ingest
        cortex_ops.ingest_incremental(file_paths=[str(py_path), str(js_path)])
        
        # 3. Verify Python Retrieval (Parsing AST headers)
        py_res = cortex_ops.query("MnemonicValidator validate_quantum_state", max_results=1)
        assert len(py_res.results) > 0
        content = py_res.results[0].content
        
        # We check for structural components (Headers and content)
        assert "MnemonicValidator" in content
        assert "validate_quantum_state" in content
        assert "0.99" in content
        
        # 4. Verify JS Retrieval (Parsing Regex headers)
        js_res = cortex_ops.query("renderDashboard", max_results=1)
        assert len(js_res.results) > 0
        content = js_res.results[0].content
        
        # The JS shim uses "Function: `renderDashboard`"
        assert "renderDashboard" in content
        print("✓ Polyglot actual DB operations verified.")

    #===========================================================================
    # MCP OPERATIONS: cortex_query (Structural Search)
    # Purpose: Verify search targeting of specific signatures/docstrings.
    # Inputs: 
    #   - Python function with type hints
    # Scenarios tested:
    #   - Retrieval of full parent context for signature-based queries
    #===========================================================================
    def test_python_structural_search(self, cortex_ops):
        print("\nTesting Python Structural Search (Incremental)...")
        
        # 1. Create a file with a very specific signature
        logic_dir = cortex_ops.project_root / "logic"
        logic_dir.mkdir()
        py_path = logic_dir / "cascade.py"
        py_path.write_text(textwrap.dedent('''
            def handle_mnemonic_cascade(sequence_id: str, intensity_threshold: float = 0.85):
                """Processes alpha-level cascades. Protocol 121 active."""
                return f"Cascade {sequence_id} active"
        '''))
        
        # 2. Ingest
        cortex_ops.ingest_incremental(file_paths=[str(py_path)])
        
        # 3. Query for specific types/signatures
        print("Searching for specialized signature...")
        q_res = cortex_ops.query("handle_mnemonic_cascade intensity_threshold", max_results=1)
        
        assert len(q_res.results) > 0
        content = q_res.results[0].content
        
        # 4. Verify full parent content is retrieved
        assert "handle_mnemonic_cascade" in content
        assert "intensity_threshold" in content
        assert "Protocol 121" in content
        print("✓ Structural search for Python signatures verified.")



    def test_learning_debrief(self, cortex_ops):
        """
        Test the learning_debrief operation (Protocol 128).
        """
        # The new learning_debrief (v3.5) is autonomous and returns "Liquid Information" (markdown string)
        # It takes 'hours' as an argument, not raw content.
        res = cortex_ops.learning_debrief(hours=1)
        
        assert "# [DRAFT] Learning Package Snapshot v3.5" in res
        assert "**Scan Time:**" in res
        assert "## 🧬 I. Tactical Evidence" in res
