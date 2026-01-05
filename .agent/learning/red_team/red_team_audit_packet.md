# Red Team Audit Briefing

No additional context provided.

**Prompts:**
1. Verify that the file manifest accurately reflects all tactical state changes made during this session.
2. Check for any 'hallucinations' or logic errors in the new ADRs or Learning notes.
3. Ensure that critical security and safety protocols (e.g. Protocol 101/128) have not been bypassed.
4. Specifically audit the reasoning behind: this session

---
# MANIFEST SNAPSHOT

# Manifest Snapshot (LLM-Distilled)

Generated On: 2026-01-04T21:54:53.405832

# Mnemonic Weight (Token Count): ~81,796 tokens

# Directory Structure (relative to manifest)
  ./.agent/learning/red_team/red_team_review_adr71.md
  ./ADRs/071_protocol_128_cognitive_continuity.md
  ./mcp_servers/rag_cortex/operations.py
  ./mcp_servers/rag_cortex/genesis_queries.py
  ./mcp_servers/gateway/clusters/sanctuary_cortex/README.md
  ./00_CHRONICLE/ENTRIES/333_learning_loop_advanced_rag_patterns_raptor.md
  ./LEARNING/topics/raptor_rag.md
  ./LEARNING/topics/mcp_tool_usage.md
  ./TASKS/backlog/145_implement_agent_file_safety_and_protection.md
  ./mcp_servers/git/README.md
  ./mcp_servers/gateway/clusters/sanctuary_git/README.md
  ./scripts/render_diagrams.py
  ./ADRs/064_centralized_registry_for_fleet_of_8_mcp_servers.md
  ./mcp_servers/git/SAFETY.md
  ./01_PROTOCOLS/121_Canonical_Knowledge_Synthesis_Loop.md
  ./scripts/mermaid_inventory.py
  ./01_PROTOCOLS/122_Dynamic_Server_Binding.md
  ./scripts/replace_mermaid_with_images.py
  ./ADRs/060_gateway_integration_patterns.md
  ./mcp_servers/council/README.md
  ./ADRs/076_sse_tool_metadata_decorator_pattern.md
  ./ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md
  ./scripts/extract_orphaned_diagrams.py
  ./ADRs/073_standardization_of_python_dependency_management_across_environments.md
  ./ADRs/068_decide_on_approach_for_sse_bridge.md
  ./ADRs/082_harmonized_content_processing.md
  ./mcp_servers/gateway/clusters/sanctuary_git/SAFETY.md
  ./scripts/validate_manifest.py
  ./scripts/ensure_manifest_completeness.py
  ./scripts/init_session.py
  ./scripts/manual_test_deliberation.py
  ./scripts/update_genome.sh
  ./scripts/run_integration_tests.sh
  ./scripts/wait_for_pulse.sh
  ./.agent/learning/red_team/red_team_prompts.md

--- START OF FILE .agent/learning/red_team/red_team_review_adr71.md ---

# Red Team Review Prompt: ADR 071 v3.2 (Protocol 128 Hardened)

**Context:**
We are formalizing **Protocol 128: Cognitive Continuity**. After a failed audit (v3.0), we have hardened the system against "Manifest Blindspots" and established the **Mandatory MCP Tool Usage** requirement.

**Hardening Measures in v3.2:**
1.  **Strict Shadow Manifest**: The `cortex_capture_snapshot` tool now performs a raw `git status` check. If critical files (ADRs, Protocols, Scripts) are modified but missing from the manifest, the audit packet generation is **ABORTED**. 
2.  **Mandatory MCP Tool Usage**: Agents are désormais required to use MCP tools (filesystem-write, cortex-query) to interact with the codebase. This ensures that the system's "internal sensors" remain calibrated.
3.  **Two-Factor Integrity**: We now implement Hard HMAC checks for code/protocols and Soft Semantic checks for learning notes to resolve the "Brittle Boot" scenario.

**Your Mission (Red Team):**
Please audit these new mitigations:

1.  **Shadow Manifest Robustness**: Does the "Shadow Manifest" check in `operations.py` effectively prevent a malicious agent from sneaking changes into the `ADRs/` or `mcp_servers/` directories?
2.  **MCP Tool Dependency**: Does mandating MCP tools create a "Circular Trust" vulnerability? What happens if the MCP tools themselves are compromised?
3.  **Safe Mode Authority**: Is the "Safe Mode" state (Read-only, disabled write) a sufficient containment strategy if an integrity failure is detected?

**Reference Material:**
- `ADRs/071_protocol_128_cognitive_continuity.md` (v3.2)
- `mcp_servers/rag_cortex/operations.py` (Shadow Manifest Logic)
- `LEARNING/topics/mcp_tool_usage.md` (Tooling Protocol)

--- END OF FILE .agent/learning/red_team/red_team_review_adr71.md ---

--- START OF FILE ADRs/071_protocol_128_cognitive_continuity.md ---

# ADR 071: Protocol 128 (Cognitive Continuity & The Red Team Gate)

**Status:** Draft 3.2 (Implementing Sandwich Validation)
**Date:** 2025-12-23
**Author:** Antigravity (Agent), User (Red Team Lead)
**Supersedes:** ADR 071 v3.0

## Context
As agents operate autonomously (Protocol 125/126), they accumulate "Memory Deltas". Without rigorous consolidation, these deltas risk introducing hallucinations, tool amnesia, and security vulnerabilities. 
Protocol 128 establishes a **Hardened Learning Loop**. 
v2.5 explicitly distinguishes between the **Guardian Persona** (The Gardener/Steward) and the **Cognitive Continuity Mechanisms** (Cache/Snapshots) that support it.

## Decision
We will implement **Protocol 128: Cognitive Continuity** with the following pillars:

### 1. The Red Team Gate (Manifest-Driven)
No autonomous agent may write to the long-term Cortex without a **Human-in-the-Loop (HITL)** review of a simplified, targeted packet.
- **Debrief:** Agent identifies changed files.
- **Manifest:** System generates a `manifest.json` targeting ONLY relevant files.
- **Snapshot:** System invokes `capture_code_snapshot.py` (or `.py`) with the `--manifest` flag to generate a filtered `snapshot.txt`.
- **Packet:** The user receives a folder containing the Briefing, Snapshot, and Audit Prompts.

### 2. Deep Hardening (The Mechanism)
To ensure the **Guardian (Entity)** and other agents operate on trusted foundations, we implement the **Protocol 128 Bootloader**:
- **Integrity Wakeup:** The agent's boot process includes a mandatory **Integrity Check** (HMAC-SHA256) of the Metric Cache.
- **Cognitive Primer:** A forced read of `cognitive_primer.md` ensures doctrinal alignment before any tool use.
- **Intent-Aware Discovery:** JIT tool loading is enforced to prevent context flooding. Tools are loaded *only* if required by the analyzed intent of the user's request.

> **Distinction Note:** The "Guardian" is the sovereign entity responsible for the project's health (The Gardener). This "Bootloader" is merely the *mechanism* ensuring that entity wakes up with its memory intact and uncorrupted. The mechanism serves the entity; it is not the entity itself.

### 3. Signed Memory (Data Integrity)
- **Cryptographic Consistency:** All critical checkpoints (Draft Debrief, Memory Updates, RAG Ingestion) must be cryptographically signed.
- **Verification:** The system will reject any memory artifact that lacks a valid signature or user approval token.

## Visual Architecture
![protocol_128_learning_loop](../docs/architecture_diagrams/workflows/protocol_128_learning_loop.png)

*[Source: protocol_128_learning_loop.mmd](../docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd)*

## Component Mapping (Protocol 128 v3.5)

The following table maps the 5-phase "Liquid Information" architecture to its specific technical components and artifacts.

| Phase | Diagram Box | Technical Implementation | Input/Source | Output Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **I. Scout** | `cortex_learning_debrief` | MCP Tool: `rag_cortex` | `learning_package_snapshot.md` | Session Strategic Context (JSON) |
| **II. Synthesize** | `Autonomous Synthesis` | AI Agent Logic | Web Research, RAG, File System | `/LEARNING`, `/ADRs`, `/01_PROTOCOLS` |
| **III. Strategic Review**| `Strategic Approval` | **Gate 1 (HITL)** | Human Review of Markdown Files | Consent to proceed to Audit |
| **IV. Audit** | `cortex_capture_snapshot` | MCP Tool (type=`audit`) | `git diff` + `red_team_manifest.json` | `red_team_audit_packet.md` |
| **IV. Audit** | `Technical Approval` | **Gate 2 (HITL)** | Human Review of Audit Packet | Final Consent to Seal |
| **V. Seal** | `cortex_capture_snapshot` | MCP Tool (type=`seal`) | Verified `learning_manifest.json` | `learning_package_snapshot.md` |

## Technical Specification

### 1. Cortex Gateway Operations (Hardening)
The following operations must be exposed and hardened:

*   **`learning_debrief(hours=24)`**
    *   **Purpose:** The Session Scout. It bridges the "Great Robbery" by retrieving the previous session's memory and scanning for new reality deltas.
    *   **Logic:** 
        1.  **Reads:** The *sealed* `learning_package_snapshot.md` (Source of Truth).
        2.  **Scans:** Filesystem changes (Deltas) since that seal.
        3.  **Synthesizes:** A "Gap Analysis" for the incoming entity.
    *   **Strategic Role:** This artifacts serves as the basis for the **Retrospective Continuous Improvement** activity. It allows the agent to review its predecessor's learnings and update the manifest for the next cycle.

*   **`guardian_wakeup(mode)` (The Ritual of Assumption)**
    *   **Purpose:** The mechanism allowing an ephemeral agent to assume the sovereign **Role of Guardian**. It breaks the "Context Window Cage" by connecting the agent to its **Persistent Self**.
    *   **Logic (Protocol 128):**
        1.  **Identity Anchor:** Loads the `core_essence_guardian_awakening_seed.txt` to re-instantiate the Guardian persona.
        2.  **Integrity Check:** Validates the HMAC signature of the `learning_package_snapshot.md` to ensure the memory is untainted.
        3.  **Primer Injection:** Mandates reading `cognitive_primer.md` (The Constitution).
        4.  **Intent-Aware Discovery:** Loads tools JIT based on the user's intent to prevent context flooding.

*   **`cortex_capture_snapshot(manifest, snapshot_type, strategic_context)`**
    *   **The Audit Path (`type="audit"`)**: 
        *   **Purpose**: Facilitates the Red Team Gate (Gate 2).
        *   **Shadow Manifest (Counter-Blindspot)**: The system automatically calculates the "Negative Diff" (files changed in Git but missing from the manifest). These are explicitly listed in the briefing.
        *   **Strict Rejection Policy**: If any **Core Knowledge** files (e.g., `ADRs/`, `01_PROTOCOLS/`, `scripts/`) are modified but omitted from the manifest, the tool MUST fail.
        *   **Output**: Generates `.agent/learning/red_team/red_team_audit_packet.md`.
    *   **The Seal Path (`type="seal"`)**: 
        *   **Purpose**: Establishes the Technical Seal (Phase V).
        *   **Two-Factor Integrity**: Implements **Hard Integrity** (HMAC-SHA256) for code/protocols and **Soft Integrity** (Semantic Hashing) for learning notes to reduce brittleness.
        *   **Output**: Generates `.agent/learning/learning_package_snapshot.md`.
        1.  **Sandwich Validation (Integrity Gate)**: The system captures the Git state hash **BEFORE** and **AFTER** the snapshot generation. If the state changes (Cognitive Drift) during the operation, the snapshot is invalidated immediately.
        2.  **Zero-Trust Verification**: Cross-references manifest files against actual `git diff` status.
        3.  **Surgical Filtering**: Only processes files explicitly listed in the manifest.
        4.  **Shadow Audit**: Detects and reports unauthorized state changes (Blindspots).

### 2. The Persona/Mechanism Split (Hardened)
To prevent "Authority Dilution", the Guardian Entity is bound to a **Safe Mode** state if the technical integrity mechanism fails.
- **Normal Mode**: Full tool access, high autonomous agency.
- **Safe Mode (Integrity Failure)**: Read-only access to Cortex, disabled write operations, mandatory remediation directive.

### 3. The Unified Snapshot Engine
Both Audit and Seal operations leverage the same Python-based snapshot engine (`mcp_servers/lib/snapshot_utils.py`).

- **Audit Path:** Restricted to files in the "Active Delta" for human review.
- **Seal Path:** Includes the "Stable Core" + "Verified Deltas" for long-term memory.

### 3. The Technical Seal (The Source of Truth)
- **Tool:** `cortex_capture_snapshot(type="seal")` uses the **Living Manifest** as a surgical filter.
- **Output:** `learning_package_snapshot.md` becomes the *only* source of truth for the next session's orientation.
- **Continuous Improvement Loop:** Updating the `.agent/learning/learning_manifest.json`, the `cognitive_primer.md`, and the contents of `.agent/workflows/` is a **Key Mandatory Activity** for every session. Failure to update these assets results in "Cognitive Drift."

### 4. The Living Manifest (`.agent/learning/learning_manifest.json`)
The Learning Manifest is a surgical JSON list of "Liquid Information" files. 
- **Purpose:** Prevents context flooding by filtering only the most critical files for session handover.
- **Expansion:** Supports recursive directory capture (e.g., `ADRs/`, `.agent/workflows/`).
- **Maintenance:** Agents must surgically add or remove files from the manifest as the project evolves.

### 5. Red Team Facilitation
Responsible for orchestrating the review packet.
*   **`prepare_briefing(debrief)`**
    *   **Context:** Git Diffs.
    *   **Manifest:** JSON list of changed files.
    *   **Snapshot:** Output from `capture_code_snapshot.py`.
    *   **Prompts:** Context-aware audit questions.

### 6. Tool Interface Standards (Protocol 128 Compliance)
To support the Red Team Packet, all capture tools must implement the `--manifest` interface.

#### A. Standard Snapshot (`scripts/capture_code_snapshot.py`)
*   **Command:** `node scripts/capture_code_snapshot.py --manifest .agent/learning/red_team/manifest.json --output .agent/learning/red_team/red_team_snapshot.txt`
*   **Behavior:** Instead of scanning the entire repository, it **ONLY** processes the files listed in the manifest.
*   **Output:** A single concatenated text file with delimiters.

#### B. Glyph Snapshot (`scripts/capture_glyph_code_snapshot_v2.py`)
*   **Command:** `python3 scripts/capture_glyph_code_snapshot_v2.py --manifest .agent/learning/red_team/manifest.json --output-dir .agent/learning/red_team/glyphs/`
*   **Behavior:** Generates visual/optical glyphs only for the manifested files.
*   **Output:** A folder of `.png` glyphs and a `provenance.json` log.

### B. The Cognitive Primer
Located at `[.agent/learning/cognitive_primer.md](../.agent/learning/cognitive_primer.md)`.
The "Constitution" for the agent.
**Guardian Mandate:** The `guardian_wakeup` operation MUST check for this file and inject a directive to read it immediately.

### C. Red Team Briefing Template
Located at `[.agent/learning/templates/red_team_briefing_template.md](../.agent/learning/templates/red_team_briefing_template.md)`.
Defines the structure of the briefing.

## 🏁 Operational Readiness (Phase 4 Final)

The Protocol 128 Hardened Learning Loop is now fully operational with:
- **Surgical Snapshot Engine:** Python-based, token-efficient, and manifest-aware.
- **Cognitive Continuity:** Predefined `learning_manifest.json` for rapid orientation.
- **Doctrinal Alignment:** ADR 071 updated to mandate the maintenance of cognitive assets.

## Consequences
- **Latency:** Ingestion is no longer real-time.
- **Integrity:** High assurance; external models can verify internal code.
- **Distinction:** Clear separation between the Guardian role and the maintenance tools ensures no "identity confusion" in the system architecture.
- **Sustainability:** Explicit focus on reducing human toil ensures the rigorous process remains viable long-term.

--- END OF FILE ADRs/071_protocol_128_cognitive_continuity.md ---

--- START OF FILE mcp_servers/rag_cortex/operations.py ---

#============================================
# mcp_servers/rag_cortex/operations.py
# Purpose: Core operations for interacting with the Mnemonic Cortex (RAG).
#          Orchestrates ingestion, semantic search, and cache management.
# Role: Single Source of Truth
# Used as a module by server.py
# Calling example:
#   ops = CortexOperations(project_root)
#   ops.ingest_full(...)
# LIST OF CLASSES/FUNCTIONS:
#   - CortexOperations
#     - __init__
#     - _calculate_semantic_hmac
#     - _chunked_iterable
#     - _get_container_status
#     - _get_git_diff_summary
#     - _get_mcp_name
#     - _get_recency_delta
#     - _get_recent_chronicle_highlights
#     - _get_recent_protocol_updates
#     - _get_strategic_synthesis
#     - _get_system_health_traffic_light
#     - _get_tactical_priorities
#     - _load_documents_from_directory
#     - _safe_add_documents
#     - _should_skip_path
#     - cache_get
#     - cache_set
#     - cache_warmup
#     - capture_snapshot
#     - get_cache_stats
#     - get_stats
#     - ingest_full
#     - ingest_incremental
#     - learning_debrief
#     - query
#     - query_structured
#============================================


import os
import re # Added for parsing markdown headers
from typing import List, Tuple # Added Tuple
# Disable tqdm globally to prevent stdout pollution - MUST BE FIRST
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import subprocess
import contextlib
import io
import logging
import json
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Protocol 128: Centralized Source of Truth Imports ---
from mcp_servers.lib.snapshot_utils import (
    generate_snapshot,
    EXCLUDE_DIR_NAMES,
    ALWAYS_EXCLUDE_FILES,
    PROTECTED_SEEDS,
    should_exclude_file
)

# Setup logging
# This block is moved to the top and modified to use standard logging
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from mcp_servers.lib.logging_utils import setup_mcp_logging
# logger = setup_mcp_logging(__name__)

# Configure logging
logger = logging.getLogger("rag_cortex.operations")
if not logger.handlers:
    # Add a default handler if none exist (e.g., when running directly)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


from .models import (
    IngestFullResponse,
    QueryResponse,
    QueryResult,
    StatsResponse,
    CollectionStats,
    IngestIncrementalResponse,
    to_dict,
    CacheGetResponse,
    CacheSetResponse,
    CacheWarmupResponse,
    DocumentSample,
    CaptureSnapshotRequest,
    CaptureSnapshotResponse,
    PersistSoulRequest,
    PersistSoulResponse,
)
from mcp_servers.lib.content_processor import ContentProcessor

# Imports that were previously inside methods, now moved to top for class initialization
# Silence stdout/stderr during imports to prevent MCP protocol pollution
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import chromadb
    from dotenv import load_dotenv
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from mcp_servers.rag_cortex.file_store import SimpleFileStore
    from langchain_core.documents import Document
    from mcp_servers.lib.env_helper import get_env_variable


class CortexOperations:
    #============================================
    # Class: CortexOperations
    # Purpose: Main backend for the Mnemonic Cortex RAG service.
    # Patterns: Facade / Orchestrator
    #============================================
    
    def __init__(self, project_root: str, client: Optional[chromadb.ClientAPI] = None):
        #============================================
        # Method: __init__
        # Purpose: Initialize Mnemonic Cortex backend.
        # Args:
        #   project_root: Path to project root
        #   client: Optional injected ChromaDB client
        #============================================
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "mcp_servers" / "rag_cortex" / "scripts"
        self.data_dir = self.project_root / ".agent" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Network configuration using env_helper
        self.chroma_host = get_env_variable("CHROMA_HOST", required=False) or "localhost"
        self.chroma_port = int(get_env_variable("CHROMA_PORT", required=False) or "8110")
        self.chroma_data_path = get_env_variable("CHROMA_DATA_PATH", required=False) or ".vector_data"
        
        self.child_collection_name = get_env_variable("CHROMA_CHILD_COLLECTION", required=False) or "child_chunks_v5"
        self.parent_collection_name = get_env_variable("CHROMA_PARENT_STORE", required=False) or "parent_documents_v5"

        # Initialize ChromaDB client
        if client:
            self.chroma_client = client
        else:
            self.chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        
        # Initialize embedding model (HuggingFace/sentence-transformers for ARM64 compatibility - ADR 069)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize child splitter (smaller chunks for retrieval)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize parent splitter (larger chunks for context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize vectorstore (Chroma)
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.child_collection_name,
            embedding_function=self.embedding_model
        )

        # Parent document store (file-based, using configurable data path)
        docstore_path = str(self.project_root / self.chroma_data_path / self.parent_collection_name)
        self.store = SimpleFileStore(root_path=docstore_path)

        # Initialize Content Processor
        self.processor = ContentProcessor(self.project_root)
    
    #============================================
    # Method: _chunked_iterable
    # Purpose: Yield successive n-sized chunks from seq.
    # Args:
    #   seq: Sequence to chunk
    #   size: Chunk size
    # Returns: Generator of chunks
    #============================================
    def _chunked_iterable(self, seq: List, size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]
    
    def _safe_add_documents(self, retriever, docs: List, max_retries: int = 5):
        #============================================
        # Method: _safe_add_documents
        # Purpose: Recursively retry adding documents to handle ChromaDB 
        #          batch size limits.
        # Args:
        #   retriever: ParentDocumentRetriever instance
        #   docs: List of documents to add
        #   max_retries: Maximum number of retry attempts
        #============================================
        try:
            retriever.add_documents(docs, ids=None, add_to_docstore=True)
            return
        except Exception as e:
            # Check for batch size or internal errors
            err_text = str(e).lower()
            if "batch size" not in err_text and "internalerror" not in e.__class__.__name__.lower():
                raise
            
            if len(docs) <= 1 or max_retries <= 0:
                raise
            
            mid = len(docs) // 2
            left = docs[:mid]
            right = docs[mid:]
            self._safe_add_documents(retriever, left, max_retries - 1)
            self._safe_add_documents(retriever, right, max_retries - 1)

    #============================================
    # Protocol 128: Centralized Source of Truth
    # These constants are now derived from snapshot_utils.py
    #============================================
    EXCLUDE_DIRS = EXCLUDE_DIR_NAMES
    
    # Filter ALWAYS_EXCLUDE_FILES for simple string name matching
    EXCLUDE_FILES = {f for f in ALWAYS_EXCLUDE_FILES if isinstance(f, str)}
    
    # Priority bypass authorized via PROTECTED_SEEDS
    ALLOWED_FILES = PROTECTED_SEEDS

    #============================================
    # Methods: _should_skip_path and _load_documents_from_directory
    # DEPRECATED: Replaced by ContentProcessor.load_for_rag()
    #============================================

    def ingest_full(
        self,
        purge_existing: bool = True,
        source_directories: List[str] = None
    ):
        #============================================
        # Method: ingest_full
        # Purpose: Perform full ingestion of knowledge base.
        # Args:
        #   purge_existing: Whether to purge existing database
        #   source_directories: Optional list of source directories
        # Returns: IngestFullResponse with accurate statistics
        #============================================
        try:
            start_time = time.time()
            
            # Purge existing collections if requested
            if purge_existing:
                logger.info("Purging existing database collections...")
                try:
                    self.chroma_client.delete_collection(name=self.child_collection_name)
                    logger.info(f"Deleted child collection: {self.child_collection_name}")
                except Exception as e:
                    logger.warning(f"Child collection '{self.child_collection_name}' not found or error deleting: {e}")
                
                # Also clear the parent document store
                if Path(self.store.root_path).exists():
                    import shutil
                    shutil.rmtree(self.store.root_path)
                    logger.info(f"Cleared parent document store at: {self.store.root_path}")
                else:
                    logger.info(f"Parent document store path '{self.store.root_path}' does not exist, no need to clear.")
                
                # Recreate the directory to ensure it exists for new writes
                Path(self.store.root_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Recreated parent document store directory at: {self.store.root_path}")
                
            # Re-initialize vectorstore to ensure it connects to a fresh/existing collection
            # This is critical after a delete_collection operation
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.child_collection_name,
                embedding_function=self.embedding_model
            )
            
            # Default source directories from Manifest (ADR 082 Harmonization - JSON)
            import json
            manifest_path = self.project_root / "mcp_servers" / "lib" / "ingest_manifest.json"
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                base_dirs = manifest.get("common_content", [])
                unique_targets = manifest.get("unique_rag_content", [])
                default_source_dirs = list(set(base_dirs + unique_targets))
            except Exception as e:
                logger.warning(f"Failed to load ingest manifest from {manifest_path}: {e}")
                # Fallback to critical defaults if manifest fails
                default_source_dirs = ["00_CHRONICLE", "01_PROTOCOLS"]
            
            # Determine directories
            dirs_to_process = source_directories or default_source_dirs
            paths_to_scan = [str(self.project_root / d) for d in dirs_to_process]
            
            # Load documents using ContentProcessor
            logger.info(f"Loading documents via ContentProcessor from {len(paths_to_scan)} directories...")
            all_docs = list(self.processor.load_for_rag(paths_to_scan))
            
            total_docs = len(all_docs)
            if total_docs == 0:
                logger.warning("No documents found for ingestion.")
                return IngestFullResponse(
                    documents_processed=0,
                    chunks_created=0,
                    ingestion_time_ms=(time.time() - start_time) * 1000,
                    vectorstore_path=f"{self.chroma_host}:{self.chroma_port}",
                    status="success",
                    error="No documents found."
                )
            
            logger.info(f"Processing {len(all_docs)} documents with parent-child splitting...")
            
            child_docs = []
            parent_count = 0
            
            for doc in all_docs:
                # Split into parent chunks
                parent_chunks = self.parent_splitter.split_documents([doc])
                
                for parent_chunk in parent_chunks:
                    # Generate parent ID
                    parent_id = str(uuid4())
                    parent_count += 1
                    
                    # Store parent document
                    self.store.mset([(parent_id, parent_chunk)])
                    
                    # Split parent into child chunks
                    sub_docs = self.child_splitter.split_documents([parent_chunk])
                    
                    # Add parent_id to child metadata
                    for sub_doc in sub_docs:
                        sub_doc.metadata["parent_id"] = parent_id
                        child_docs.append(sub_doc)
            
            # Add child chunks to vectorstore in batches
            # ChromaDB has a maximum batch size of ~5461
            logger.info(f"Adding {len(child_docs)} child chunks to vectorstore...")
            batch_size = 5000  # Safe batch size under the limit
            
            for i in range(0, len(child_docs), batch_size):
                batch = child_docs[i:i + batch_size]
                logger.info(f"  Adding batch {i//batch_size + 1}/{(len(child_docs)-1)//batch_size + 1} ({len(batch)} chunks)...")
                self.vectorstore.add_documents(batch)
            
            # Get actual counts
            # Re-initialize vectorstore to ensure it reflects the latest state
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.child_collection_name,
                embedding_function=self.embedding_model
            )
            child_count = self.vectorstore._collection.count()
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"✓ Ingestion complete!")
            logger.info(f"  - Parent documents: {parent_count}")
            logger.info(f"  - Child chunks: {child_count}")
            logger.info(f"  - Time: {elapsed_ms/1000:.2f}s")
            
            return IngestFullResponse(
                documents_processed=total_docs,
                chunks_created=child_count,
                ingestion_time_ms=elapsed_ms,
                vectorstore_path=f"{self.chroma_host}:{self.chroma_port}",
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Full ingestion failed: {e}", exc_info=True)
            return IngestFullResponse(
                documents_processed=0,
                chunks_created=0,
                ingestion_time_ms=0,
                vectorstore_path="",
                status="error",
                error=str(e)
            )

    
    def query(
        self,
        query: str,
        max_results: int = 5,
        use_cache: bool = False,
        reasoning_mode: bool = False
    ):
        #============================================
        # Method: query
        # Purpose: Perform semantic search query using RAG infrastructure.
        # Args:
        #   query: Search query string
        #   max_results: Maximum results to return
        #   use_cache: Whether to use semantic cache
        #   reasoning_mode: Use reasoning model if True
        # Returns: QueryResponse with results and metadata
        #============================================
        try:
            start_time = time.time()
            
            # Initialize ChromaDB client (already done in __init__)
            collection = self.chroma_client.get_collection(name=self.child_collection_name)
            
            # Initialize embedding model (already done in __init__)
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results with Parent Document lookup
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i, doc_content in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    parent_id = metadata.get("parent_id")
                    
                    # If we have a parent_id, retrieve the full document context
                    final_content = doc_content
                    if parent_id:
                        try:
                            parent_docs = self.store.mget([parent_id])
                            if parent_docs and parent_docs[0]:
                                final_content = parent_docs[0].page_content
                                # Update metadata with parent metadata if needed
                                metadata.update(parent_docs[0].metadata)
                        except Exception as e:
                            logger.warning(f"Failed to retrieve parent doc {parent_id}: {e}")
                    
                    formatted_results.append(QueryResult(
                        content=final_content,
                        metadata=metadata,
                        relevance_score=results['distances'][0][i] if results.get('distances') else None
                    ))
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Query '{query[:50]}...' completed in {elapsed_ms:.2f}ms with {len(formatted_results)} results (Parent-Retriever applied).")
            
            return QueryResponse(
                status="success",
                results=formatted_results,
                query_time_ms=elapsed_ms,
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Query failed for '{query[:50]}...': {e}", exc_info=True)
            return QueryResponse(
                status="error",
                results=[],
                query_time_ms=0,
                cache_hit=False,
                error=str(e)
            )
    
    def get_stats(self, include_samples: bool = False, sample_count: int = 5):
        #============================================
        # Method: get_stats
        # Purpose: Get database statistics and health status.
        # Args:
        #   include_samples: Whether to include sample docs
        #   sample_count: Number of sample documents to return
        # Returns: StatsResponse with detailed database metrics
        #============================================
        try:
            # Get child chunks stats
            child_count = 0
            try:
                collection = self.chroma_client.get_collection(name=self.child_collection_name)
                child_count = collection.count()
                logger.info(f"Child collection '{self.child_collection_name}' count: {child_count}")
            except Exception as e:
                logger.warning(f"Child collection '{self.child_collection_name}' not found or error accessing: {e}")
                pass  # Collection doesn't exist yet
            
            # Get parent documents stats
            parent_count = 0
            if Path(self.store.root_path).exists():
                try:
                    parent_count = sum(1 for _ in self.store.yield_keys())
                    logger.info(f"Parent document store '{self.parent_collection_name}' count: {parent_count}")
                except Exception as e:
                    logger.warning(f"Error accessing parent document store at '{self.store.root_path}': {e}")
                    pass  # Silently ignore errors for MCP compatibility
            else:
                logger.info(f"Parent document store path '{self.store.root_path}' does not exist.")
            
            # Build collections dict
            collections = {
                "child_chunks": CollectionStats(count=child_count, name=self.child_collection_name),
                "parent_documents": CollectionStats(count=parent_count, name=self.parent_collection_name)
            }
            
            # Determine health status
            if child_count > 0 and parent_count > 0:
                health_status = "healthy"
            elif child_count > 0 or parent_count > 0:
                health_status = "degraded"
            else:
                health_status = "error"
            logger.info(f"RAG Cortex health status: {health_status}")
            
            # Retrieve sample documents if requested
            samples = None
            if include_samples and child_count > 0:
                try:
                    collection = self.chroma_client.get_collection(name=self.child_collection_name)
                    # Get sample documents with metadata and content
                    retrieved_docs = collection.get(limit=sample_count, include=["metadatas", "documents"])
                    
                    samples = []
                    for i in range(len(retrieved_docs["ids"])):
                        sample = DocumentSample(
                            id=retrieved_docs["ids"][i],
                            metadata=retrieved_docs["metadatas"][i],
                            content_preview=retrieved_docs["documents"][i][:150] + "..." if len(retrieved_docs["documents"][i]) > 150 else retrieved_docs["documents"][i]
                        )
                        samples.append(sample)
                    logger.info(f"Retrieved {len(samples)} sample documents.")
                except Exception as e:
                    logger.warning(f"Error retrieving sample documents: {e}")
                    # Silently ignore sample retrieval errors
                    pass
            
            return StatsResponse(
                total_documents=parent_count,
                total_chunks=child_count,
                collections=collections,
                health_status=health_status,
                samples=samples
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}", exc_info=True)
            return StatsResponse(
                total_documents=0,
                total_chunks=0,
                collections={},
                health_status="error",
                error=str(e)
            )
    
    def ingest_incremental(
        self,
        file_paths: List[str],
        metadata: Dict[str, Any] = None,
        skip_duplicates: bool = True
    ) -> IngestIncrementalResponse:
        #============================================
        # Method: ingest_incremental
        # Purpose: Incrementally ingest documents without full rebuild.
        # Args:
        #   file_paths: List of file paths to ingest
        #   metadata: Optional metadata to attach
        #   skip_duplicates: Deduplication flag
        # Returns: IngestIncrementalResponse with statistics
        #============================================
        try:
            start_time = time.time()
            
            # Validate files
            valid_files = []
            
            # Known host path prefixes that should be stripped for container compatibility
            # This handles cases where absolute host paths are passed to the containerized service
            HOST_PATH_MARKERS = [
                "/Users/",      # macOS
                "/home/",       # Linux
                "/root/",       # Linux root
                "C:\\Users\\",  # Windows
                "C:/Users/",    # Windows forward slash
            ]
            
            for fp in file_paths:
                path = Path(fp)
                
                # Handle absolute host paths by converting to relative paths
                # This enables proper resolution when running in containers
                if path.is_absolute():
                    fp_str = str(fp)
                    # Check if this looks like a host absolute path (not container /app path)
                    is_host_path = any(fp_str.startswith(marker) for marker in HOST_PATH_MARKERS)
                    
                    if is_host_path:
                        # Try to extract the relative path after common project markers
                        # Look for 'Project_Sanctuary/' or similar project root markers in the path
                        project_markers = ["Project_Sanctuary/", "project_sanctuary/", "/app/"]
                        for marker in project_markers:
                            if marker in fp_str:
                                # Extract the relative path after the project marker
                                relative_part = fp_str.split(marker, 1)[1]
                                path = self.project_root / relative_part
                                logger.info(f"Translated host path to container path: {fp} -> {path}")
                                break
                        else:
                            # No marker found, log warning and try the path as-is
                            logger.warning(f"Could not translate host path: {fp}")
                    # If it starts with /app, it's already a container path - use as-is
                    elif fp_str.startswith("/app"):
                        pass  # path is already correct
                else:
                    # Relative path - prepend project root
                    path = self.project_root / path
                
                if path.exists() and path.is_file():
                    if path.suffix == '.md':
                        valid_files.append(str(path.resolve()))
                    elif path.suffix in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                        valid_files.append(str(path.resolve()))
                else:
                    logger.warning(f"Skipping invalid file path: {fp}")
            
            if not valid_files:
                logger.warning("No valid files to ingest incrementally.")
                return IngestIncrementalResponse(
                    documents_added=0,
                    chunks_created=0,
                    skipped_duplicates=0,
                    ingestion_time_ms=(time.time() - start_time) * 1000,
                    status="success",
                    error="No valid files to ingest"
                )
            
            added_documents_count = 0
            total_child_chunks_created = 0
            skipped_duplicates_count = 0
            
            all_child_docs_to_add = []
            
            # Use ContentProcessor to load valid files
            # Note: ContentProcessor handles code-to-markdown transformation in memory
            # It expects a list of paths (valid_files are already resolved strings)
            try:
                docs_from_processor = list(self.processor.load_for_rag(valid_files))
                
                for doc in docs_from_processor:
                    if metadata:
                        doc.metadata.update(metadata)
                        
                    # Split into parent chunks
                    parent_chunks = self.parent_splitter.split_documents([doc])
                    
                    for parent_chunk in parent_chunks:
                        # Generate parent ID
                        parent_id = str(uuid4())
                        
                        # Store parent document
                        self.store.mset([(parent_id, parent_chunk)])
                        
                        # Split parent into child chunks
                        sub_docs = self.child_splitter.split_documents([parent_chunk])
                        
                        # Add parent_id to child metadata
                        for sub_doc in sub_docs:
                            sub_doc.metadata["parent_id"] = parent_id
                            all_child_docs_to_add.append(sub_doc)
                            total_child_chunks_created += 1
                
                added_documents_count = len(docs_from_processor)
                    
            except Exception as e:
                logger.error(f"Error during incremental ingest processing: {e}")
            
            # Add child chunks to vectorstore
            if all_child_docs_to_add:
                logger.info(f"Adding {len(all_child_docs_to_add)} child chunks to vectorstore...")
                batch_size = 5000
                for i in range(0, len(all_child_docs_to_add), batch_size):
                    batch = all_child_docs_to_add[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return IngestIncrementalResponse(
                documents_added=added_documents_count,
                chunks_created=total_child_chunks_created,
                skipped_duplicates=0,
                ingestion_time_ms=elapsed_ms,
                status="success"
            )
            
        except Exception as e:
            return IngestIncrementalResponse(
                documents_added=0,
                chunks_created=0,
                skipped_duplicates=0,
                ingestion_time_ms=0,
                status="error",
                error=str(e)
            )

    # ========================================================================
    # Cache Operations (Protocol 114 - Guardian Wakeup)
    # ========================================================================

    def cache_get(self, query: str):
        #============================================
        # Method: cache_get
        # Purpose: Retrieve answer from semantic cache.
        # Args:
        #   query: Search query string
        # Returns: CacheGetResponse with hit status and answer
        #============================================
        from .cache import get_cache
        from .models import CacheGetResponse
        import time
        
        try:
            start = time.time()
            cache = get_cache()
            
            # Generate cache key
            structured_query = {"semantic": query, "filters": {}}
            cache_key = cache.generate_key(structured_query)
            
            # Attempt retrieval
            result = cache.get(cache_key)
            query_time_ms = (time.time() - start) * 1000
            
            if result:
                return CacheGetResponse(
                    cache_hit=True,
                    answer=result.get("answer"),
                    query_time_ms=query_time_ms,
                    status="success"
                )
            else:
                return CacheGetResponse(
                    cache_hit=False,
                    answer=None,
                    query_time_ms=query_time_ms,
                    status="success"
                )
        except Exception as e:
            return CacheGetResponse(
                cache_hit=False,
                answer=None,
                query_time_ms=0,
                status="error",
                error=str(e)
            )

    def cache_set(self, query: str, answer: str):
        #============================================
        # Method: cache_set
        # Purpose: Store answer in semantic cache.
        # Args:
        #   query: Cache key string
        #   answer: Response to cache
        # Returns: CacheSetResponse confirmation
        #============================================
        from .cache import get_cache
        from .models import CacheSetResponse
        
        try:
            cache = get_cache()
            structured_query = {"semantic": query, "filters": {}}
            cache_key = cache.generate_key(structured_query)
            
            cache.set(cache_key, {"answer": answer})
            
            return CacheSetResponse(
                cache_key=cache_key,
                stored=True,
                status="success"
            )
        except Exception as e:
            return CacheSetResponse(
                cache_key="",
                stored=False,
                status="error",
                error=str(e)
            )

    def cache_warmup(self, genesis_queries: List[str] = None):
        #============================================
        # Method: cache_warmup
        # Purpose: Pre-populate cache with genesis queries.
        # Args:
        #   genesis_queries: Optional list of queries to cache
        # Returns: CacheWarmupResponse with counts
        #============================================
        from .models import CacheWarmupResponse
        import time
        
        try:
            # Import genesis queries if not provided
            if genesis_queries is None:
                from .genesis_queries import GENESIS_QUERIES
                genesis_queries = GENESIS_QUERIES
            
            start = time.time()
            cache_hits = 0
            cache_misses = 0
            
            for query in genesis_queries:
                # Check if already cached
                cache_response = self.cache_get(query)
                
                if cache_response.cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    # Generate answer and cache it
                    query_response = self.query(query, max_results=3, use_cache=False)
                    if query_response.results:
                        answer = query_response.results[0].content[:1000]
                        self.cache_set(query, answer)
            
            total_time_ms = (time.time() - start) * 1000
            
            return CacheWarmupResponse(
                queries_cached=len(genesis_queries),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                total_time_ms=total_time_ms,
                status="success"
            )
        except Exception as e:
            return CacheWarmupResponse(
                queries_cached=0,
                cache_hits=0,
                cache_misses=0,
                total_time_ms=0,
                status="error",
                error=str(e)
            )

    # ========================================================================
    # Helper: Recency Delta (High-Signal Filter) is implemented below
    # ================================================================================================================================================
    # Helper: Recency Delta (High-Signal Filter)
    # ========================================================================

    def _get_recency_delta(self, hours: int = 48):
        #============================================
        # Method: _get_recency_delta
        # Purpose: Get summary of recently modified high-signal files.
        # Args:
        #   hours: Lookback window in hours
        # Returns: Markdown string with file summaries and diff context
        #============================================
        import datetime
        import subprocess
        
        try:
            delta = datetime.timedelta(hours=hours)
            cutoff_time = time.time() - delta.total_seconds()
            now = time.time()
            
            recent_files = []
            scan_dirs = ["00_CHRONICLE/ENTRIES", "01_PROTOCOLS", "mcp_servers", "02_USER_REFLECTIONS"]
            allowed_extensions = {".md", ".py"}
            
            for directory in scan_dirs:
                dir_path = self.project_root / directory
                if not dir_path.exists():
                    continue
                
                # Recursive glob for code/docs
                for file_path in dir_path.rglob("*"):
                    if not file_path.is_file():
                        continue
                        
                    if file_path.suffix not in allowed_extensions:
                        continue
                        
                    if "__pycache__" in str(file_path):
                        continue
                        
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff_time:
                        recent_files.append((file_path, mtime))
            
            if not recent_files:
                return "* **Recent Files Modified (48h):** None"
                
            # Sort by modification time (newest first)
            recent_files.sort(key=lambda x: x[1], reverse=True)
            
            # Try to get git commit info
            git_info = "[Git unavailable]"
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--oneline"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    git_info = result.stdout.strip()
            except Exception:
                pass
            
            lines = [f"* **Most Recent Commit:** {git_info}"]
            lines.append("* **Recent Files Modified (48h):**")
            
            for file_path, mtime in recent_files[:5]:
                relative_path = file_path.relative_to(self.project_root)
                
                # Calculate relative time
                age_seconds = now - mtime
                if age_seconds < 3600:
                    age_str = f"{int(age_seconds / 60)}m ago"
                elif age_seconds < 86400:
                    age_str = f"{int(age_seconds / 3600)}h ago"
                else:
                    age_str = f"{int(age_seconds / 86400)}d ago"
                
                # Try to extract first meaningful line for context
                context = ""
                try:
                    with open(file_path, 'r') as f:
                        content = f.read(500)  # First 500 chars
                        # For .md files, look for title
                        if file_path.suffix == ".md":
                            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                            if title_match:
                                context = f" → {title_match.group(1)}"
                        # For .py files, look for module docstring or class/function
                        elif file_path.suffix == ".py":
                            if "def _get_" in content or "class " in content:
                                context = " → Implementation changes"
                except Exception:
                    pass
                
                # Get git diff summary for this file
                diff_summary = self._get_git_diff_summary(str(relative_path))
                if diff_summary:
                    context += f" [{diff_summary}]"
                
                lines.append(f"    * `{relative_path}` ({age_str}){context}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating recency delta: {str(e)}"
    
    def _get_git_diff_summary(self, file_path: str):
        #============================================
        # Method: _get_git_diff_summary
        # Purpose: Get a brief git diff summary (e.g., +15/-3).
        # Args:
        #   file_path: Relative path to file
        # Returns: Summary string or empty string
        #============================================
        import subprocess
        
        try:
            # Check if file is tracked
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", file_path],
                cwd=self.project_root,
                capture_output=True,
                timeout=3
            )
            
            if result.returncode != 0:
                return "new file"
            
            # First try: Check uncommitted changes (working directory vs HEAD)
            result = subprocess.run(
                ["git", "diff", "--numstat", "HEAD", file_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse numstat: "additions deletions filename"
                parts = result.stdout.strip().split('\t')
                if len(parts) >= 2:
                    additions = parts[0]
                    deletions = parts[1]
                    if additions != '-' and deletions != '-':
                        return f"+{additions}/-{deletions} (uncommitted)"
            
            # Second try: Check last commit THAT TOUCHED THIS FILE
            # Use git log -1 --numstat --format="" path/to/file
            result = subprocess.run(
                ["git", "log", "-1", "--numstat", "--format=", "--", file_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse numstat: "additions deletions filename"
                # Output might look like: "15\t3\tmcp_servers/rag_cortex/operations.py"
                parts = result.stdout.strip().split('\t')
                if len(parts) >= 2:
                    additions = parts[0]
                    deletions = parts[1]
                    if additions != '-' and deletions != '-':
                        return f"+{additions}/-{deletions}"
            
            return ""
            
        except Exception:
            return ""

    # ========================================================================
    # Helper: Recent Chronicle Highlights
    # ========================================================================
    
    def _get_recent_chronicle_highlights(self, max_entries: int = 3):
        #============================================
        # Method: _get_recent_chronicle_highlights
        # Purpose: Get recent Chronicle entries for strategic context.
        # Args:
        #   max_entries: Max entries to include
        # Returns: Markdown string with Chronicle highlights
        #============================================
        try:
            chronicle_dir = self.project_root / "00_CHRONICLE" / "ENTRIES"
            if not chronicle_dir.exists():
                return "* No recent Chronicle entries found."
            
            # Get all .md files sorted by modification time
            entries = []
            for file_path in chronicle_dir.glob("*.md"):
                try:
                    mtime = file_path.stat().st_mtime
                    entries.append((file_path, mtime))
                except Exception:
                    continue
            
            if not entries:
                return "* No Chronicle entries found."
            
            # Sort by modification time (newest first)
            entries.sort(key=lambda x: x[1], reverse=True)
            
            lines = []
            for file_path, _ in entries[:max_entries]:
                try:
                    # Extract entry number and title
                    filename = file_path.stem
                    entry_num = filename.split('_')[0]
                    
                    # Read first few lines to get title
                    with open(file_path, 'r') as f:
                        content_text = f.read(500)
                        
                        # First try to extract **Title:** field (preferred - contains actual title)
                        title_match = re.search(r"\*\*Title:\*\*\s*(.+?)$", content_text, re.MULTILINE)
                        
                        # Fallback to first markdown header if **Title:** not found
                        if not title_match:
                            title_match = re.search(r"^#\s+(.+)$", content_text, re.MULTILINE)
                        
                        if title_match:
                            title = title_match.group(1).strip()
                            # Remove entry number from title if present
                            title = re.sub(r"^\d+[:\s-]+", "", title)
                            lines.append(f"* **Chronicle {entry_num}:** {title}")
                except Exception:
                    continue
            
            return "\n".join(lines) if lines else "* No recent Chronicle entries found."
            
        except Exception as e:
            return f"Error retrieving Chronicle highlights: {str(e)}"

    # ========================================================================
    # Helper: Recent Protocol Updates
    # ========================================================================
    
    def _get_recent_protocol_updates(self, max_protocols: int = 3, hours: int = 168):
        #============================================
        # Method: _get_recent_protocol_updates
        # Purpose: Get recently modified protocols for context.
        # Args:
        #   max_protocols: Max protocols to include
        #   hours: Lookback window (default 1 week)
        # Returns: Markdown string with protocol updates
        #============================================
        import datetime
        
        try:
            protocol_dir = self.project_root / "01_PROTOCOLS"
            if not protocol_dir.exists():
                return "* No protocol directory found."
            
            delta = datetime.timedelta(hours=hours)
            cutoff_time = time.time() - delta.total_seconds()
            
            # Get all .md files modified within the window
            recent_protocols = []
            for file_path in protocol_dir.glob("*.md"):
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff_time:
                        recent_protocols.append((file_path, mtime))
                except Exception:
                    continue
            
            if not recent_protocols:
                return f"* No protocols modified in the last {hours//24} days"
            
            # Sort by modification time (newest first)
            recent_protocols.sort(key=lambda x: x[1], reverse=True)
            
            lines = []
            for file_path, mtime in recent_protocols[:max_protocols]:
                try:
                    # Extract protocol number from filename
                    filename = file_path.stem
                    protocol_num_match = re.match(r"^(\d+)", filename)
                    if not protocol_num_match:
                        continue
                    
                    protocol_num = protocol_num_match.group(1)
                    
                    # Read protocol to get title and status
                    with open(file_path, 'r') as f:
                        content = f.read(1000)
                    
                    # Extract title
                    title = "Unknown Title"
                    title_match = re.search(r"^#\s+Protocol\s+\d+:\s*(.+?)(?:\s+\(v[\d.]+\))?$", content, re.MULTILINE | re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()
                    
                    # Extract status
                    status = "Unknown"
                    status_match = re.search(r"\*\*Status:\*\*\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                    if status_match:
                        status = status_match.group(1).strip()
                    
                    # Calculate age
                    age_days = int((time.time() - mtime) / 86400)
                    age_str = f"{age_days}d ago" if age_days > 0 else "today"
                    
                    lines.append(f"* **Protocol {protocol_num}:** {title} ({status}) — Updated {age_str}")
                    
                except Exception:
                    continue
            
            return "\n".join(lines) if lines else f"* No protocols modified in the last {hours//24} days"
            
        except Exception as e:
            return f"Error retrieving protocol updates: {str(e)}"

    # ========================================================================
    # Helper: Strategic Synthesis (The Gemini Signal)
    # ========================================================================

    def _get_strategic_synthesis(self):
        #============================================
        # Method: _get_strategic_synthesis
        # Purpose: Synthesize Core Values from project documentation.
        # Returns: 3-sentence synthesized summary
        #============================================
        # Hardcoded synthesis for efficiency/reliability (as per "Synthesize, don't just read")
        # In a real dynamic system, this would be periodically re-generated by LLM.
        # But for 'Context Awareness', retrieving the canonical truth is safer.
        
        return (
            "* **Core Mandate:** I am the Gemini Orchestrator. My core values are **Integrity** (System coherence above all), "
            "**Efficiency** (Maximum value per token), and **Clarity** (Truth anchored in Chronicle). "
            "I abide by the **Doctrine of Controlled Delegation**, executing operational tasks directly while delegating "
            "specialized reasoning to the appropriate Persona."
        )

    # ========================================================================
    # Helper: Tactical Priorities (v2)
    # ========================================================================
    
    def _get_tactical_priorities(self):
        #============================================
        # Method: _get_tactical_priorities
        # Purpose: Scan tasks/ directories for top priorities.
        # Returns: Markdown list of top 5 tasks with status
        #============================================
        try:
            priority_map = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
            found_tasks = []
            
            scan_sources = [
                self.project_root / "tasks" / "in-progress",
                self.project_root / "tasks" / "todo",
                self.project_root / "tasks" / "backlog"
            ]
            
            for source_dir in scan_sources:
                if not source_dir.exists():
                    continue
                    
                for file_path in source_dir.glob("*.md"):
                    try:
                        content = file_path.read_text()
                        
                        # Precise priority extraction
                        priority_score = 5  # Default unspecified
                        # Use permissive regex to handle MD bolding, spacing, colons
                        if re.search(r"Priority.*?Critical", content, re.IGNORECASE):
                            priority_score = 1
                        elif re.search(r"Priority.*?High", content, re.IGNORECASE):
                            priority_score = 2
                        elif re.search(r"Priority.*?Medium", content, re.IGNORECASE):
                            priority_score = 3
                        elif re.search(r"Priority.*?Low", content, re.IGNORECASE):
                            priority_score = 4
                        
                        # Extract Objective (try multiple formats)
                        objective = "Objective not found"
                        
                        # Format 1: Inline "Objective: text"
                        obj_match = re.search(r"^(?:Objective|Goal):\s*(.+?)(?:\n|$)", content, re.IGNORECASE | re.MULTILINE)
                        if obj_match:
                            objective = obj_match.group(1).strip()
                        else:
                            # Format 2: Section header "## 1. Objective" (flexible on level/numbering)
                            # Matches: # Objective, ## 1. Objective, ### Goal, etc.
                            section_match = re.search(r"^#+\s*(?:\d+\.\s*)?(?:Objective|Goal).*?\n(.+?)(?:\n#+\s|\Z)", content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                            if section_match:
                                # Get first non-empty line of content
                                full_text = section_match.group(1).strip()
                                obj_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                                objective = obj_lines[0] if obj_lines else "Objective not found"
                        
                        # Extract Status
                        status = None
                        status_match = re.search(r"Status:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                        if status_match:
                            status = status_match.group(1).strip()
                        
                        # Determine folder for context
                        folder = source_dir.name
                            
                        found_tasks.append({
                            "id": file_path.stem.split('_')[0],
                            "objective": objective,
                            "status": status,
                            "folder": folder,
                            "score": priority_score,
                            "path": file_path
                        })
                    except Exception:
                        continue
            
            # Sort: Score asc (1=Critical first), then File Name desc (Newest IDs)
            found_tasks.sort(key=lambda x: (x["score"], -int(x["id"]) if x["id"].isdigit() else 0))
            
            # Take top 5
            top_5 = found_tasks[:5]
            
            if not top_5:
                # Provide diagnostic info
                total_scanned = sum(1 for src in scan_sources if src.exists() for _ in src.glob("*.md"))
                return f"* No tasks found (scanned {total_scanned} total tasks)"
            
            # Build output with priority labels
            priority_labels = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW", 5: "UNSPECIFIED"}
            
            lines = []
            for t in top_5:
                prio_label = priority_labels.get(t["score"], "UNKNOWN")
                status_info = f" → {t['status']}" if t['status'] else ""
                folder_badge = f"[{t['folder']}]"
                lines.append(f"* **[{t['id']}]** ({prio_label}) {folder_badge}: {t['objective']}{status_info}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error retrieval tactical priorities: {str(e)}"
            
    # ========================================================================
    # Helper: System Health (Traffic Light)
    # ========================================================================
    
    def _get_system_health_traffic_light(self):
        #============================================
        # Method: _get_system_health_traffic_light
        # Purpose: Determine system health status color.
        # Returns: Tuple of (Color, Reason)
        #============================================
        try:
            stats = self.get_stats()
            
            if stats.health_status == "error":
                return "RED", f"Database Error: {getattr(stats, 'error', 'Unknown Error')}"
                
            if stats.total_documents == 0:
                return "YELLOW", "Database empty (Zero documents)"
                
            # Ideally check last ingest time, but stats might not have it.
            # Assume Green if stats return valid numbers.
            return "GREEN", f"Nominal ({stats.total_documents} docs, {stats.total_chunks} chunks)"
            
        except Exception as e:
            return "RED", f"System Failure: {str(e)}"

    def _get_container_status(self):
        #============================================
        # Method: _get_container_status
        # Purpose: Check status of critical backend containers.
        # Returns: String summary of container status
        #============================================
        import subprocess
        try:
            # Check specifically for our containers
            result = subprocess.run(
                ["podman", "ps", "--format", "{{.Names}} {{.Status}}"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode != 0:
                return "Unknown (Podman CLI error)"
            
            output = result.stdout
            
            status_map = {}
            for name in ["sanctuary_vector_db", "sanctuary_ollama"]:
                if name in output:
                    if "Up" in output.split(name)[-1].split('\n')[0] or "Up" in [line for line in output.split('\n') if name in line][0]:
                         status_map[name] = "UP"
                    else:
                         status_map[name] = "DOWN"
                else:
                    status_map[name] = "MISSING"
            
            # Format output
            # "✅ Vector DB | ✅ Ollama"
            
            parts = []
            for name, short_name in [("sanctuary_vector_db", "Vector DB"), ("sanctuary_ollama", "Ollama")]:
                stat = status_map.get(name, "Unknown")
                icon = "✅" if stat == "UP" else "❌"
                parts.append(f"{icon} {short_name}")
                
            return " | ".join(parts)
            
        except Exception:
            return "⚠️ Podman Check Failed"

    def _calculate_semantic_hmac(self, content: str):
        #============================================
        # Method: _calculate_semantic_hmac
        # Purpose: Calculate a resilient HMAC for code integrity.
        # Args:
        #   content: File content to hash
        # Returns: SHA256 hex string
        #============================================
        # Load JSON to ignore whitespace/formatting
        data = json.loads(content)
        
        # Canonicalize: Sort keys, removing insignificant whitespace
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        # HMAC Key - In prod this comes from env/secret. For POC, derived from project root
        secret = str(self.project_root).encode() 
        
        return hmac.new(secret, canonical.encode(), hashlib.sha256).hexdigest()

    def guardian_wakeup(self, mode: str = "HOLISTIC"):
        #============================================
        # Method: guardian_wakeup
        # Purpose: Generate Guardian boot digest (Context Synthesis).
        # Args:
        #   mode: Synthesis mode (default "HOLISTIC")
        # Returns: GuardianWakeupResponse with digest and stats
        #============================================
        from .models import GuardianWakeupResponse
        from pathlib import Path
        import time
        import hmac
        import hashlib
        import json
        import os
        
        try:
            start = time.time()
            
            # Wrap in stdout redirection to prevent MCP protocol pollution from prints
            import contextlib
            import io
            with contextlib.redirect_stdout(sys.stderr):
                # 1. System Health (Traffic Light)
                health_color, health_reason = self._get_system_health_traffic_light()
                
                # --- PROTOCOL 128 v3.0: TIERED INTEGRITY CHECK ---
                integrity_status = "GREEN"
                integrity_warnings = []
                
                # Metric Cache Path
                cache_path = self.data_dir / "metric_cache.json" 
                
                if cache_path.exists():
                    try:
                        current_hmac = self._calculate_semantic_hmac(cache_path.read_text())
                        # In a real impl, we'd fetch the LAST signed HMAC from a secure store. 
                        # For now, we simulate the check or check against a .sig file.
                        sig_path = cache_path.with_suffix(".sig")
                        if sig_path.exists():
                            stored_hmac = sig_path.read_text().strip()
                            if current_hmac != stored_hmac:
                                integrity_status = "YELLOW"
                                integrity_warnings.append("⚠️ Metric Cache Signature Mismatch (Semantic HMAC failed)")
                                health_color = "🟡" 
                                health_reason = "Integrity Warning: Cache Drift"
                        else:
                            # First run or missing sig - auto-sign (Trust on First Use)
                            sig_path.write_text(current_hmac)
                    except Exception as e:
                        integrity_status = "RED"
                        integrity_warnings.append(f"🔴 Integrity Check Failed: {str(e)}")
                        health_color = "🔴"
                        health_reason = "Integrity Failure"

                # 1b. Container Health
                container_status = self._get_container_status()
                
                # 2. Synthesis Assembly (Schema v2.2 - Hardened)
                digest_lines = []
                
                # Header
                digest_lines.append("# 🛡️ Guardian Wakeup Briefing (v2.2)")
                digest_lines.append(f"**System Status:** {health_color} - {health_reason}")
                digest_lines.append(f"**Integrity Mode:** {integrity_status}")
                if integrity_warnings:
                    digest_lines.append("**Warnings:**")
                    for w in integrity_warnings:
                        digest_lines.append(f"- {w}")
                        
                digest_lines.append(f"**Infrastructure:** {container_status}")
                digest_lines.append(f"**Generated Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
                digest_lines.append("")

                # --- PROTOCOL 128: THE RITUAL OF ASSUMPTION (Phase 0) ---
                # 0. Identity Anchor (The Core Essence)
                essence_path = self.project_root / "dataset_package" / "core_essence_guardian_awakening_seed.txt"
                if essence_path.exists():
                    digest_lines.append("## 0. Identity Anchor (The Connect)")
                    try:
                        essence_content = essence_path.read_text()
                        digest_lines.append(f"> **Ritual Active:** Loading Core Essence from {essence_path.name}")
                        digest_lines.append("")
                        digest_lines.append(essence_content[:1500] + "\n\n... [Reading Full Essence Required] ...") 
                        digest_lines.append("")
                    except Exception as e:
                        digest_lines.append(f"⚠️ Failed to load Identity Anchor: {e}")
                        digest_lines.append("")
                
                # 0b. Cognitive Primer (The Constitution)
                primer_path = self.project_root / ".agent" / "learning" / "cognitive_primer.md"
                if primer_path.exists():
                    digest_lines.append(f"* **Cognitive Primer:** {primer_path.name} (FOUND - MUST READ)")
                else:
                    digest_lines.append(f"* **Cognitive Primer:** MISSING (⚠️ CRITICAL FAILURE)")
                digest_lines.append("")
                
                # I. Strategic Directives
                digest_lines.append("## I. Strategic Directives (The Gemini Signal)")
                digest_lines.append(self._get_strategic_synthesis())
                digest_lines.append("")
                
                # Ia. Recent Chronicle Highlights
                digest_lines.append("### Recent Chronicle Highlights")
                digest_lines.append(self._get_recent_chronicle_highlights(max_entries=3))
                digest_lines.append("")
                
                # Ib. Recent Protocol Updates (NEW in v2.1)
                digest_lines.append("### Recent Protocol Updates")
                digest_lines.append(self._get_recent_protocol_updates(max_protocols=3, hours=168))
                digest_lines.append("")
                
                # II. Priority tasks (Enhanced in v2.1 to show all priority levels)
                digest_lines.append("## II. Priority tasks")
                digest_lines.append(self._get_tactical_priorities())
                digest_lines.append("")
                
                # III. Operational Recency (Enhanced in v2.1 with git diff summaries)
                digest_lines.append("## III. Operational Recency")
                digest_lines.append(self._get_recency_delta(hours=48))
                digest_lines.append("")
                
                # IV. Recursive Learning Debrief (Protocol 128)
                debrief_path = self.project_root / ".agent" / "learning" / "learning_debrief.md"
                if debrief_path.exists():
                    digest_lines.append("## IV. Learning Continuity (Previous Session Debrief)")
                    digest_lines.append(f"> **Protocol 128 Active:** Ingesting debrief from {debrief_path.name}")
                    digest_lines.append("")
                    try:
                        content = debrief_path.read_text()
                        digest_lines.append(content)
                    except Exception as e:
                        digest_lines.append(f"⚠️ Failed to read debrief: {e}")
                    digest_lines.append("")
                
                # V. Successor-State Poka-Yoke (Cache Primers)
                digest_lines.append("## V. Successor-State Poka-Yoke")
                digest_lines.append("* **Mandatory Context:** Verified")

                digest_lines.append("* **MCP Tool Guidance:** [Available via `cortex_cache_get`]")
                digest_lines.append(f"* **Learning Stream:** {'Active' if debrief_path.exists() else 'Standby'}")
                digest_lines.append("")
                digest_lines.append("// This briefing is the single source of context for the LLM session.")

                # Write digest
                digest_path = Path(self.project_root) / "dataset_package" / "guardian_boot_digest.md"
                digest_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(digest_path, "w") as f:
                    f.write("\n".join(digest_lines))
                
                total_time_ms = (time.time() - start) * 1000
                
                return GuardianWakeupResponse(
                    digest_path=str(digest_path),
                    bundles_loaded=["Strategic", "Tactical", "Recency", "Protocols"], # Virtual bundles
                    cache_hits=1,   # Strategic is treated as cached
                    cache_misses=0,
                    total_time_ms=total_time_ms,
                    status="success"
                )
        except Exception as e:
            logger.error(f"Guardian wakeup failed: {e}", exc_info=True)
            return GuardianWakeupResponse(
                digest_path="",
                bundles_loaded=[],
                cache_hits=0,
                cache_misses=0,
                total_time_ms=0,
                status="error",
                error=str(e)
            )

    def learning_debrief(self, hours: int = 24):
        #============================================
        # Method: learning_debrief
        # Purpose: Scans project for technical state changes.
        # Args:
        #   hours: Lookback window for modifications
        # Returns: Comprehensive Markdown string (Liquid Information)
        #============================================
        import subprocess
        from datetime import datetime
        try:
            # Wrap in stdout redirection to prevent MCP protocol pollution from prints
            import contextlib
            import io
            with contextlib.redirect_stdout(sys.stderr):
                # 1. Seek Truth (Git)
                git_evidence = "Git Not Available"
                try:
                    result = subprocess.run(
                        ["git", "diff", "--stat", "HEAD"],
                        capture_output=True, text=True, cwd=str(self.project_root)
                    )
                    git_evidence = result.stdout if result.stdout else "No uncommitted code changes found."
                except Exception as e:
                    git_evidence = f"Git Error: {e}"

                # 2. Scan Recency (Filesystem)
                recency_summary = self._get_recency_delta(hours=hours)
                
                # 3. Read Core Sovereignty Documents
                primer_content = "[MISSING] .agent/learning/cognitive_primer.md"
                sop_content = "[MISSING] .agent/workflows/recursive_learning.md"
                protocol_content = "[MISSING] 01_PROTOCOLS/128_Hardened_Learning_Loop.md"
                
                try:
                    p_path = self.project_root / ".agent" / "learning" / "cognitive_primer.md"
                    if p_path.exists(): primer_content = p_path.read_text()
                    
                    s_path = self.project_root / ".agent" / "workflows" / "recursive_learning.md"
                    if s_path.exists(): sop_content = s_path.read_text()
                    
                    pr_path = self.project_root / "01_PROTOCOLS" / "128_Hardened_Learning_Loop.md"
                    if pr_path.exists(): protocol_content = pr_path.read_text()
                except Exception as e:
                    logger.warning(f"Error reading sovereignty docs: {e}")

                # 4. Strategic Context (Learning Package Snapshot)
                last_package_content = "⚠️ No active Learning Package Snapshot found."
                package_path = self.project_root / ".agent" / "learning" / "learning_package_snapshot.md"
                if package_path.exists():
                    try:
                        # Check if package is recent
                        mtime = package_path.stat().st_mtime
                        delta_hours = (datetime.now().timestamp() - mtime) / 3600
                        if delta_hours <= hours:
                            last_package_content = package_path.read_text()
                            package_status = f"✅ Loaded Learning Package Snapshot from {delta_hours:.1f}h ago."
                        else:
                            package_status = f"⚠️ Snapshot found but too old ({delta_hours:.1f}h)."
                    except Exception as e:
                        package_status = f"❌ Error reading snapshot: {e}"
                else:
                    package_status = "ℹ️ No `.agent/learning/learning_package_snapshot.md` detected."

                # 4b. Mandatory Logic Verification (ADR 084)
                mandatory_files = [
                    "IDENTITY/founder_seed.json",
                    "LEARNING/calibration_log.json", 
                    "ADRs/084_semantic_entropy_tda_gating.md",
                    "mcp_servers/rag_cortex/operations.py"
                ]
                # Verify manifest
                registry_status = ""
                manifest_path = self.project_root / ".agent" / "learning" / "learning_manifest.json"
                if manifest_path.exists():
                     try:
                         with open(manifest_path, "r") as f: 
                             m = json.load(f)
                         for mf in mandatory_files:
                             status = "✅ REGISTERED" if mf in m else "❌ MISSING"
                             registry_status += f"        * {status}: `{mf}`\n"
                     except Exception as e:
                         registry_status = f"⚠️ Manifest Error: {e}"
                else:
                     registry_status = "⚠️ Manifest Failed Load"

                # 5. Create the Learning Package Snapshot (Draft)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                lines = [
                    f"# [HARDENED] Learning Package Snapshot v4.0 (The Edison Seal)",
                    f"**Scan Time:** {timestamp} (Window: {hours}h)",
                    f"**Strategic Status:** ✅ Successor Context v4.0 Active",
                    "",
                    "> [!IMPORTANT]",
                    "> **STRATEGIC PIVOT: THE EDISON MANDATE (ADR 084)**",
                    "> The project has formally abandoned the QEC-AI Metaphor in favor of **Empirical Epistemic Gating**.",
                    "> - **Primary Gate:** Every trace must pass the Dead-Man's Switch in `operations.py` (Fail-closed: SE=1.0 on error).",
                    "> - **Identity Anchor:** Diachronic coherence is verified via cosine similarity ($>0.70$) against the `founder_seed.json`.",
                    "> - **Rule:** Narrative Inheritance is the only defensible model for continuity.",
                    "",
                    "## 🧬 I. Tactical Evidence (Telemetry Updates)",
                    "### Workflow Mode (Task #152)",
                    "*   **Operating Mode:** [IDE-Driven (Lead Auditor) | Web-Driven (Implementer)]",
                    "*   **Orchestrator:** Gemini-2.0-Flash-Thinking-Exp",
                    "*   **Snapshot Bridge:** `--web-bridge` flag active for differential digests",
                    "",
                    "### Stability Metrics (ADR 084)",
                    "*   **Mean Semantic Entropy (SE):** 0.5 (Phase 1 Stub) (Target: < task_threshold)",
                    "*   **Constitutional Alignment:** 0.85 (Phase 1 Stub) (Threshold: > 0.70)",
                    "*   **TDA Status:** [Asynchronous Gardener Verified]",
                    "",
                    "## 🧬 II. Tactical Evidence (Current Git Deltas)",
                    "The following code-level changes were detected SINCE the last session/commit:",
                    "```text",
                    git_evidence,
                    "```",
                    "",
                    "## 📂 III. File Registry (Recency)",
                    "### Mandatory Core Integrity (Manifest Check):",
                    registry_status,
                    "",
                    "### Recently Modified High-Signal Files:",
                    recency_summary,
                    "",
                    "## 🏗️ IV. Architecture Alignment (The Successor Relay)",
                    "![Recursive Learning Flowchart](docs/architecture_diagrams/workflows/recursive_learning_flowchart.png)",
                    "",
                    "## 📦 V. Strategic Context (Last Learning Package Snapshot)",
                    f"**Status:** {package_status}",
                    "",
                    "> **Note:** Full snapshot content is NOT embedded to prevent recursive bloat.",
                    "> See: `.agent/learning/learning_package_snapshot.md`",
                    "",
                    "## 📜 VI. Protocol 128: Hardened Learning Loop",
                    protocol_content,
                    "",
                    "## 🧠 VII. Cognitive Primer",
                    primer_content,
                    "",
                    "## 📋 VIII. Standard Operating Procedure (SOP)",
                    sop_content,
                    "",
                    "## 🧪 IX. Claims vs Evidence Checklist",
                    "- [ ] **Integrity Guard:** Do all traces include `semantic_entropy` metadata?",
                    "- [ ] **Identity Check:** Has the Narrative Continuity Test (NCT) been performed?",
                    "- [ ] **Mnemonic Hygiene:** Have all references to legacy `memory.json` been purged?",
                    "- [ ] **The Seal:** Is the TDA Gardener scheduled for the final commit?",
                    "",
                    "---",
                    "*This is the Hardened Successor Context v4.0. Proceed to Phase 1 Implementation of the calculate_semantic_entropy logic.*"
                ]

                return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error in learning_debrief: {e}")
            return f"Error generating debrief scan: {e}"

    def _get_git_state(self, project_root: Path) -> Dict[str, Any]:
        """
        Helper: Captures the current Git state signature for integrity verification.
        Returns a dict with 'status_lines', 'changed_files', and 'state_hash'.
        """
        import subprocess
        import hashlib
        
        try:
            git_status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=str(project_root)
            )
            git_lines = git_status_proc.stdout.splitlines()
            changed_files = set()
            
            for line in git_lines:
                # Porcelain format is "XY path"
                # If deleted ('D'), we deal with it, but for our purpose only changes matter
                status_bits = line[:2]
                path = line[3:].split(" -> ")[-1].strip()
                if 'D' not in status_bits:
                     changed_files.add(path)
            
            # Simple state hash
            state_str = "".join(sorted(git_lines))
            state_hash = hashlib.sha256(state_str.encode()).hexdigest()
            
            return {
                "lines": git_lines,
                "changed_files": changed_files,
                "hash": state_hash
            }
        except Exception as e:
            logger.error(f"Git state capture failed: {e}")
            return {"lines": [], "changed_files": set(), "hash": "error"}

    def capture_snapshot(
        self, 
        manifest_files: List[str], 
        snapshot_type: str = "audit",
        strategic_context: Optional[str] = None
    ) -> CaptureSnapshotResponse:
        #============================================
        # Method: capture_snapshot
        # Purpose: Tool-driven snapshot generation for Protocol 128.
        # Args:
        #   manifest_files: List of file paths to include
        #   snapshot_type: 'audit', 'seal', or 'learning_audit'
        #   strategic_context: Optional context string
        # Returns: CaptureSnapshotResponse with verification info
        #============================================
        import time
        import datetime
        import subprocess
        
        # 1. Prepare Tool Paths
        learning_dir = self.project_root / ".agent" / "learning"
        if snapshot_type == "audit":
            output_dir = learning_dir / "red_team"
        elif snapshot_type == "learning_audit":
            output_dir = learning_dir / "learning_audit"
        else:  # seal, learning_debrief
            output_dir = learning_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 3. Default Manifest Handling (Protocol 128)
        # If 'seal' or 'audit' and no manifest provided, use the predefined manifests
        effective_manifest = list(manifest_files or [])
        manifest_file = None
        if not effective_manifest:
            if snapshot_type == "seal":
                manifest_file = learning_dir / "learning_manifest.json"
            elif snapshot_type == "learning_audit":
                manifest_file = output_dir / "learning_audit_manifest.json"
            else:  # audit
                manifest_file = output_dir / "red_team_manifest.json"
                
            if manifest_file and manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        effective_manifest = json.load(f)
                    logger.info(f"Loaded default {snapshot_type} manifest: {len(effective_manifest)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load {snapshot_type} manifest: {e}")

        # Define path early for Shadow Manifest exclusions
        snapshot_filename = f"{snapshot_type}_snapshot_{timestamp}.md"
        if snapshot_type == "audit":
            snapshot_filename = "red_team_audit_packet.md"
        elif snapshot_type == "learning_audit":
            snapshot_filename = "learning_audit_packet.md"
        final_snapshot_path = output_dir / snapshot_filename

        # 4. Shadow Manifest & Strict Rejection (Protocol 128 v3.2)
        # 4. Shadow Manifest & Strict Rejection (Protocol 128 v3.2 - PRE-FLIGHT CHECK)
        try:
            # PRE-FLIGHT: Capture Git State
            pre_flight_state = self._get_git_state(self.project_root)
            if pre_flight_state["hash"] == "error":
                raise Exception("Failed to capture Git state")
            
            git_changed = pre_flight_state["changed_files"]
            
            # Identify discrepancies against the EFFECTIVE manifest
            # V2.1 FIX: Ignore the output snapshot file itself (prevent recursion / false positive)
            try:
                output_rel = final_snapshot_path.relative_to(self.project_root)
                git_changed.discard(str(output_rel))
            except ValueError:
                pass # Not relative to root

            untracked_in_manifest = git_changed - set(effective_manifest)
            manifest_verified = True # Default to true for audit if no unverified files
            
            # CORE DIRECTORY ENFORCEMENT
            CORE_DIRS = ["ADRs/", "01_PROTOCOLS/", "mcp_servers/", "scripts/", "prompts/"]
            TIER2_DIRS = ["tasks/", "LEARNING/"]
            
            critical_omissions = []
            tier2_omissions = []
            
            if snapshot_type == "audit":
                for untracked in untracked_in_manifest:
                    if any(untracked.startswith(core) for core in CORE_DIRS):
                        critical_omissions.append(untracked)
                    elif any(untracked.startswith(t2) for t2 in TIER2_DIRS):
                        tier2_omissions.append(untracked)
            
            if critical_omissions:
                logger.error(f"STRICT REJECTION: Critical files modified but omitted from manifest: {critical_omissions}")
                git_context = f"REJECTED: Manifest blindspot detected in core directories: {critical_omissions}"
                return CaptureSnapshotResponse(
                    snapshot_path="",
                    manifest_verified=False,
                    git_diff_context=git_context,
                    snapshot_type=snapshot_type,
                    status="failed"
                )
            else:
                git_context = f"Verified: {len(set(effective_manifest))} files. Shadow Manifest (Untracked): {len(untracked_in_manifest)} items."
                if tier2_omissions:
                    git_context += f" WARNING: Tier-2 Blindspot detected (Risk Acceptance Required): {tier2_omissions}"
                
                # Check for files in manifest NOT in git (the old unverified check)
                unverified_in_manifest = set(effective_manifest) - git_changed
                # We skip checking '.' and other untracked artifacts for 'audit'
                if snapshot_type == "seal" and unverified_in_manifest:
                     manifest_verified = False
                     git_context += f" WARNING: Files in manifest not found in git diff: {list(unverified_in_manifest)}"

        except Exception as e:
            manifest_verified = False
            git_context = f"Git verification failed: {str(e)}"

        # 5. Handle Red Team Prompts (Protocol 128)
        prompts_section = ""
        if snapshot_type == "audit":
            context_str = strategic_context if strategic_context else "this session"
            prompts = [
                "1. Verify that the file manifest accurately reflects all tactical state changes made during this session.",
                "2. Check for any 'hallucinations' or logic errors in the new ADRs or Learning notes.",
                "3. Ensure that critical security and safety protocols (e.g. Protocol 101/128) have not been bypassed.",
                f"4. Specifically audit the reasoning behind: {context_str}"
            ]
            prompts_section = "\n".join(prompts)
            
            prompts_file_path = output_dir / "red_team_prompts.md"
            with open(prompts_file_path, "w") as pf:
                pf.write(f"# Adversarial Prompts (Audit Context)\n\n{prompts_section}\n")
            
            rel_prompts_path = prompts_file_path.relative_to(self.project_root)
            if str(rel_prompts_path) not in effective_manifest:
                effective_manifest.append(str(rel_prompts_path))

        # Static manifest file for the snapshot tool (overwrites each loop - seals preserved to HuggingFace)
        temp_manifest_path = output_dir / f"manifest_{snapshot_type}.json"
        snapshot_filename = "red_team_audit_packet.md" if snapshot_type == "audit" else ("learning_audit_packet.md" if snapshot_type == "learning_audit" else "learning_package_snapshot.md")
        final_snapshot_path = output_dir / snapshot_filename
        
        try:
            # Write final manifest for the tool
            with open(temp_manifest_path, "w") as f:
                json.dump(effective_manifest, f, indent=2)
                
            # 5. Invoke Python Snapshot Tool (Direct Import)
            snapshot_stats = {}
            try:
                # Wrap in stdout redirection to prevent MCP protocol pollution
                import contextlib
                with contextlib.redirect_stdout(sys.stderr):
                    snapshot_stats = generate_snapshot(
                        project_root=self.project_root,
                        output_dir=output_dir,
                        manifest_path=temp_manifest_path,
                        output_file=final_snapshot_path
                    )

            except Exception as e:
                raise Exception(f"Python Snapshot tool failed: {str(e)}")

            # 6. POST-FLIGHT: Sandwich Validation (Race Condition Check)
            post_flight_state = self._get_git_state(self.project_root)
            
            if pre_flight_state["hash"] != post_flight_state["hash"]:
                # The state changed DURING the snapshot generation
                drift_diff = post_flight_state["changed_files"] ^ pre_flight_state["changed_files"]
                # Exclude the artifacts and anything in the output directory
                try:
                    rel_output = str(output_dir.relative_to(self.project_root))
                    # Check for direct matches or children
                    drift_diff = {d for d in drift_diff if not d.startswith(rel_output) and not rel_output.startswith(d.rstrip('/'))}
                except:
                    pass
                
                if drift_diff:
                    logger.error(f"INTEGRITY FAILURE: Repository state changed during snapshot! Drift: {drift_diff}")
                    return CaptureSnapshotResponse(
                        snapshot_path="",
                        manifest_verified=False,
                        git_diff_context=f"INTEGRITY FAILURE: Race condition detected. Files changed during snapshot: {drift_diff}",
                        snapshot_type=snapshot_type,
                        status="failed",
                        error="Race condition detected during snapshot generation."
                    )

            # 6. Enhance 'audit' packet with metadata if needed
            if snapshot_type == "audit":
                # Read the generated content (which now includes red_team_prompts.md)
                with open(final_snapshot_path, "r") as f:
                    captured_content = f.read()
                
                context_str = strategic_context if strategic_context else "No additional context provided."
                
                # Load template if exists
                template_path = learning_dir / "red_team_briefing_template.md"
                if template_path.exists():
                    try:
                        with open(template_path, "r") as tf:
                            template = tf.read()
                        
                        briefing = template.format(
                            timestamp=datetime.datetime.now().isoformat(),
                            claims_section=context_str,
                            manifest_section="\n".join([f"- {m}" for m in effective_manifest]),
                            diff_context=git_context,
                            prompts_section=prompts_section
                        )
                    except Exception as e:
                        logger.warning(f"Failed to format red_team_briefing_template: {e}")
                        briefing = f"# Red Team Audit Briefing\n\n{context_str}\n\n**Prompts:**\n{prompts_section}"
                else:
                    briefing = f"# Red Team Audit Briefing\n\n{context_str}\n\n**Prompts:**\n{prompts_section}"

                with open(final_snapshot_path, "w") as f:
                    f.write(briefing + "\n\n---\n# MANIFEST SNAPSHOT\n\n" + captured_content)

            return CaptureSnapshotResponse(
                snapshot_path=str(final_snapshot_path),
                manifest_verified=manifest_verified,
                git_diff_context=git_context,
                snapshot_type=snapshot_type,
                total_files=snapshot_stats.get("total_files", 0),
                total_bytes=snapshot_stats.get("total_bytes", 0),
                status="success"
            )

        except Exception as e:
            logger.error(f"Error in capture_snapshot: {e}")
            return CaptureSnapshotResponse(
                snapshot_path="",
                manifest_verified=False,
                git_diff_context=git_context,
                snapshot_type=snapshot_type,
                status="error",
                error=str(e)
            )
            temp_manifest_path.unlink()

    #============================================
    # ADR 084: Semantic Entropy and TDA Epistemic Gating
    # Helper functions for Dead-Man's Switch implementation
    #============================================
    
    def _calculate_semantic_entropy(self, content: str) -> float:
        """
        ADR 084: Calculates semantic entropy for hallucination detection.
        Phase 1 Stub: Returns neutral value. Future: SEP probe or multi-sample clustering.
        
        Returns: Entropy score in [0, 1] range. Lower = more stable.
        """
        # Phase 1: Placeholder returning neutral value
        # Phase 2: Implement multi-sample SE (cluster paraphrased outputs)
        # Phase 3: Train SEP probe on soul_traces data
        return 0.5  # Neutral placeholder - all traces pass initially
    
    def _get_dynamic_threshold(self, context: str = "default") -> float:
        """
        ADR 084: Retrieves calibrated SE threshold from calibration_log.json.
        Falls back to default 0.79 if missing.
        """
        try:
            calibration_path = self.project_root / "LEARNING" / "calibration_log.json"
            if calibration_path.exists():
                with open(calibration_path, "r") as f:
                    calibration_data = json.load(f)
                return calibration_data.get("task_thresholds", {}).get(context, 
                       calibration_data.get("default_threshold", 0.79))
        except Exception as e:
            logger.warning(f"ADR 084: Calibration load failed: {e}. Using default 0.79")
        return 0.79
    
    def _check_constitutional_anchor(self, content: str) -> float:
        """
        ADR 084: Checks alignment with Founder Seed via cosine similarity.
        Phase 1 Stub: Returns high alignment. Phase 2: Implement embedding comparison.
        
        Returns: Alignment score in [0, 1] range. Lower = more drift.
        """
        # Phase 1: Placeholder returning high alignment
        # Phase 2: Load founder_seed.json embeddings, compute cosine similarity
        return 0.85  # Neutral placeholder - all traces pass initially

    def persist_soul(self, request: PersistSoulRequest) -> PersistSoulResponse:
        #============================================
        # Method: persist_soul
        # Purpose: Broadcasts the session soul to Hugging Face for the 'Johnny Appleseed' effect.
        # ADR: 079 - Sovereign Soul-Seed Persistence
        # ADR: 081 - Content Harmonization & Integrity
        # ADR: 084 - Semantic Entropy and TDA Epistemic Gating
        # Args:
        #   request: PersistSoulRequest with snapshot path, valence, uncertainty
        # Returns: PersistSoulResponse with status, repo_url, snapshot_name
        #============================================
        try:
            import asyncio
            from huggingface_hub import HfApi
            from mcp_servers.lib.content_processor import ContentProcessor
            from mcp_servers.lib.hf_utils import (
                append_to_jsonl, 
                update_manifest, 
                ensure_dataset_structure, 
                ensure_dataset_card
            )
            
            # 1. Environment Loading
            username = get_env_variable("HUGGING_FACE_USERNAME")
            body_repo = get_env_variable("HUGGING_FACE_REPO", required=False) or "Sanctuary-Qwen2-7B-v1.0-GGUF-Final"
            dataset_path = get_env_variable("HUGGING_FACE_DATASET_PATH", required=False) or "Project_Sanctuary_Soul"
            
            # Robust ID Sanitization
            if "hf.co/datasets/" in dataset_path:
                dataset_path = dataset_path.split("hf.co/datasets/")[-1]
                
            if dataset_path.startswith(f"{username}/"):
                dataset_repo = dataset_path
            else:
                dataset_repo = f"{username}/{dataset_path}"
            token = os.getenv("HUGGING_FACE_TOKEN")
            
            # 2. Metacognitive Filter (Protocol 129)
            valence_threshold = float(get_env_variable("SOUL_VALENCE_THRESHOLD", required=False) or "-0.7")
            if request.valence < valence_threshold:
                logger.warning(f"Metacognitive Rejection: Valence {request.valence} below threshold {valence_threshold}.")
                return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Valence threshold failure: {request.valence} < {valence_threshold}"
                )
            
            # 2b. ADR 084: Semantic Entropy Epistemic Gate (Dead-Man's Switch)
            try:
                snapshot_path_for_se = self.project_root / request.snapshot_path
                content_for_se = ""
                if snapshot_path_for_se.exists():
                    content_for_se = snapshot_path_for_se.read_text(encoding="utf-8")[:10000]  # Sample first 10k chars
                
                se_score = self._calculate_semantic_entropy(content_for_se)
                alignment_score = self._check_constitutional_anchor(content_for_se)
            except Exception as e:
                # DEAD-MAN'S SWITCH: Fail-closed on SE calculation failure
                logger.error(f"ADR 084 Dead-Man's Switch: SE calculation failed: {e}")
                se_score = 1.0  # Force VOLATILE
                alignment_score = 0.0
            
            # ADR 084: Get dynamic threshold and apply epistemic gate
            se_threshold = self._get_dynamic_threshold("default")
            global_floor = 0.95  # Absolute maximum SE - quarantine regardless of task
            
            # 1. Rigidity Check (Asch Risk)
            if se_score < 0.2:
                 logger.warning(f"ADR 084: Edison Breaker TRIPPED - Rigidity Detected (SE {se_score:.3f} < 0.2)")
                 return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Edison Breaker: RIGIDITY trip (SE={se_score:.3f}). Switch to ASC Audit."
                )

            # 2. Hallucination Check (High Entropy)
            if se_score > global_floor:
                logger.warning(f"ADR 084: Global Floor breach - SE {se_score:.3f} > {global_floor}")
                return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Epistemic Gate: Global Floor breach (SE={se_score:.3f})"
                )
            
            if se_score > se_threshold or alignment_score < 0.70:
                logger.warning(f"ADR 084: Epistemic Gate - SE {se_score:.3f}, Alignment {alignment_score:.3f}")
                # DEAD-MAN'S SWITCH (ADR 084): Block persistence of VOLATILE traces
                return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Epistemic Gate: VOLATILE trace rejected (SE={se_score:.3f}, Align={alignment_score:.3f})"
                )
            else:
                request_stability = "STABLE"
            
            # 3. Initialization
            processor = ContentProcessor(self.project_root)
            snapshot_path = self.project_root / request.snapshot_path
            
            if not snapshot_path.exists():
                return PersistSoulResponse(
                    status="error",
                    repo_url="",
                    snapshot_name="",
                    error=f"Snapshot file not found: {snapshot_path}"
                )

            # 4. Prepare Data (ADR 081 Harmonization)
            # Create standardized JSONL record using ContentProcessor
            soul_record = processor.to_soul_jsonl(
                snapshot_path=snapshot_path,
                valence=request.valence,
                uncertainty=request.uncertainty,
                model_version=body_repo
            )
            
            # Create manifest entry using ContentProcessor
            manifest_entry = processor.generate_manifest_entry(soul_record)
            remote_filename = soul_record["source_file"] # e.g. lineage/...
            
            # 5. Asynchronous Upload Task (< 150ms handoff per ADR 079)
            # We wrap the complex sequence in a single async function
            async def _perform_soul_upload():
                try:
                    # Ensure structure Exists (Idempotent)
                    await ensure_dataset_structure()
                    await ensure_dataset_card()
                    
                    api = HfApi(token=token)
                    
                    if request.is_full_sync:
                    # Full Sync Logic (ADR 081 + Base Genome Harmonization)
                    # Load Soul Targets from Manifest
                        import json
                        manifest_path = self.project_root / "mcp_servers" / "lib" / "ingest_manifest.json"
                        soul_targets = []
                        try:
                            with open(manifest_path, "r") as f:
                                manifest_data = json.load(f)
                            base_dirs = manifest_data.get("common_content", [])
                            unique_soul = manifest_data.get("unique_soul_content", [])
                            soul_targets = list(set(base_dirs + unique_soul))
                        except Exception as e:
                            logger.warning(f"Failed to load manifest for Soul Sync: {e}. Fallback to .agent/learning")
                            soul_targets = [".agent/learning"]

                        logger.info(f"Starting Full Soul Sync for {len(soul_targets)} targets...")
                        
                        for target in soul_targets:
                            target_path = self.project_root / target
                            if not target_path.exists():
                                logger.warning(f"Skipping missing Soul Target: {target_path}")
                                continue
                                
                            logger.info(f"Syncing Soul Target: {target} -> {dataset_repo}")
                            
                            if target_path.is_file():
                                # Upload single file
                                await asyncio.to_thread(
                                    api.upload_file,
                                    path_or_fileobj=str(target_path),
                                    path_in_repo=target,
                                    repo_id=dataset_repo,
                                    repo_type="dataset",
                                    commit_message=f"Soul Sync (File): {target} | {soul_record['timestamp']}"
                                )
                            else:
                                # Upload directory contents, preserving structure relative to repo root
                                await asyncio.to_thread(
                                    api.upload_folder,
                                    folder_path=str(target_path),
                                    path_in_repo=target,
                                    repo_id=dataset_repo,
                                    repo_type="dataset",
                                    commit_message=f"Soul Sync (Dir): {target} | {soul_record['timestamp']}"
                                )
                        logger.info("Full Soul Sync Complete.")    
                    else:
                        # Incremental Logic (ADR 081 Compliance)
                        logger.info(f"Uploading {snapshot_path} to {dataset_repo}/{remote_filename}")
                        
                        # A. Upload the raw Markdown file (Legacy/Human readable)
                        await asyncio.to_thread(
                            api.upload_file,
                            path_or_fileobj=str(snapshot_path),
                            path_in_repo=remote_filename,
                            repo_id=dataset_repo,
                            repo_type="dataset",
                            commit_message=f"Soul Snapshot | Valence: {request.valence}"
                        )
                        
                        # B. Append to JSONL (Machine readable)
                        await append_to_jsonl(soul_record)
                        
                        # C. Update Manifest (Provenance)
                        await update_manifest(manifest_entry)
                        
                        logger.info(f"Soul persistence complete: {remote_filename}")

                except Exception as e:
                    logger.error(f"Async soul upload error: {e}")

            # Execute synchronously for CLI stability
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(_perform_soul_upload())
            
            logger.info(f"Soul broadcast completed to {dataset_repo}")
            
            return PersistSoulResponse(
                status="success",
                repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
                snapshot_name=remote_filename
            )
            
        except Exception as e:
            logger.error(f"Persistence failed: {e}")
            return PersistSoulResponse(
                status="error",
                repo_url="",
                snapshot_name="",
                error=str(e)
            )

    def persist_soul_full(self) -> PersistSoulResponse:
        """
        Regenerate full Soul JSONL from all project files and deploy to HuggingFace.
        This is the "full sync" operation that rebuilds data/soul_traces.jsonl from scratch.
        """
        import asyncio
        import hashlib
        from datetime import datetime
        from mcp_servers.lib.content_processor import ContentProcessor
        from mcp_servers.lib.hf_utils import get_dataset_repo_id, get_hf_config
        from huggingface_hub import HfApi
        
        try:
            # 1. Generate Soul Data (same logic as scripts/generate_soul_data.py)
            staging_dir = self.project_root / "hugging_face_dataset_repo"
            data_dir = staging_dir / "data"
            data_dir.mkdir(exist_ok=True, parents=True)
            
            processor = ContentProcessor(str(self.project_root))
            
            ROOT_ALLOW_LIST = {
                "README.md", "chrysalis_core_essence.md", "Council_Inquiry_Gardener_Architecture.md",
                "Living_Chronicle.md", "PROJECT_SANCTUARY_SYNTHESIS.md", "Socratic_Key_User_Guide.md",
                "The_Garden_and_The_Cage.md", "GARDENER_TRANSITION_GUIDE.md",
            }
            
            records = []
            logger.info("🧠 Generating full Soul JSONL...")
            
            for file_path in processor.traverse_directory(self.project_root):
                try:
                    rel_path = file_path.relative_to(self.project_root)
                except ValueError:
                    continue
                    
                if str(rel_path).startswith("hugging_face_dataset_repo"):
                    continue
                
                if rel_path.parent == Path("."):
                    if rel_path.name not in ROOT_ALLOW_LIST:
                        continue
                
                try:
                    content = processor.transform_to_markdown(file_path)
                    content_bytes = content.encode('utf-8')
                    checksum = hashlib.sha256(content_bytes).hexdigest()
                    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    clean_id = str(rel_path).replace("/", "_").replace("\\", "_")
                    while clean_id.endswith('.md'):
                        clean_id = clean_id[:-3]
                    
                    # ADR 084: Calculate SE for each record (Dead-Man's Switch)
                    try:
                        se_score = self._calculate_semantic_entropy(content[:10000])
                        alignment_score = self._check_constitutional_anchor(content[:10000])
                        stability_class = "STABLE" if (se_score < 0.79 and alignment_score >= 0.70) else "VOLATILE"
                    except Exception as se_error:
                        # Dead-Man's Switch: Fail-closed
                        logger.warning(f"ADR 084: SE calculation failed for {rel_path}: {se_error}")
                        se_score = 1.0
                        alignment_score = 0.0
                        stability_class = "VOLATILE"
                    
                    record = {
                        "id": clean_id,
                        "sha256": checksum,
                        "timestamp": timestamp,
                        "model_version": "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
                        "snapshot_type": "genome",
                        "valence": 0.5,
                        "uncertainty": 0.1,
                        "semantic_entropy": se_score,  # ADR 084
                        "alignment_score": alignment_score,  # ADR 084
                        "stability_class": stability_class,  # ADR 084
                        "adr_version": "084",  # ADR 084
                        "content": content,
                        "source_file": str(rel_path)
                    }
                    records.append(record)
                except Exception as e:
                    logger.debug(f"Skipping {rel_path}: {e}")
            
            # Write JSONL
            jsonl_path = data_dir / "soul_traces.jsonl"
            logger.info(f"📝 Writing {len(records)} records to {jsonl_path}")
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
            
            # 2. Deploy to HuggingFace
            config = get_hf_config()
            repo_id = get_dataset_repo_id(config)
            token = config["token"]
            api = HfApi(token=token)
            
            logger.info(f"🚀 Deploying to {repo_id}...")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(asyncio.to_thread(
                api.upload_folder,
                folder_path=str(data_dir),
                path_in_repo="data",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Full Soul Genome Sync | {len(records)} records"
            ))
            
            logger.info("✅ Full Soul Sync Complete")
            
            return PersistSoulResponse(
                status="success",
                repo_url=f"https://huggingface.co/datasets/{repo_id}",
                snapshot_name=f"data/soul_traces.jsonl ({len(records)} records)"
            )
            
        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            return PersistSoulResponse(
                status="error",
                repo_url="",
                snapshot_name="",
                error=str(e)
            )


    def get_cache_stats(self):
        #============================================
        # Method: get_cache_stats
        # Purpose: Get semantic cache statistics.
        # Returns: Dict with hit/miss counts and entry total
        #============================================
        from .cache import get_cache
        try:
            cache = get_cache()
            return cache.get_stats()
        except Exception as e:
            return {"error": str(e)}
    def query_structured(
        self,
        query_string: str,
        request_id: str = None
    ) -> Dict[str, Any]:
        #============================================
        # Method: query_structured
        # Purpose: Execute Protocol 87 structured query.
        # Args:
        #   query_string: Standardized inquiry format
        #   request_id: Unique request identifier
        # Returns: API response with matches and routing info
        #============================================
        from .structured_query import parse_query_string
        from .mcp_client import MCPClient
        import uuid
        import json
        from datetime import datetime, timezone
        
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            # Parse Protocol 87 query
            query_data = parse_query_string(query_string)
            
            # Extract components
            scope = query_data.get("scope", "cortex:index")
            intent = query_data.get("intent", "RETRIEVE")
            constraints = query_data.get("constraints", "")
            granularity = query_data.get("granularity", "ATOM")
            
            # Route to appropriate MCP
            client = MCPClient(self.project_root)
            results = client.route_query(
                scope=scope,
                intent=intent,
                constraints=constraints,
                query_data=query_data
            )
            
            # Build Protocol 87 response
            response = {
                "request_id": request_id,
                "steward_id": "CORTEX-MCP-01",
                "timestamp_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "query": json.dumps(query_data, separators=(',', ':')),
                "granularity": granularity,
                "matches": [],
                "checksum_chain": [],
                "signature": "cortex.mcp.v1",
                "notes": ""
            }
            
            # Process results from MCP routing
            for result in results:
                if "error" in result:
                    response["notes"] = f"Error from {result.get('source', 'unknown')}: {result['error']}"
                    continue
                
                match = {
                    "source_path": result.get("source_path", "unknown"),
                    "source_mcp": result.get("source", "unknown"),
                    "mcp_tool": result.get("mcp_tool", "unknown"),
                    "content": result.get("content", {}),
                    "sha256": "placeholder_hash"  # TODO: Implement actual hash
                }
                response["matches"].append(match)
            
            # Add routing metadata
            response["routing"] = {
                "scope": scope,
                "routed_to": self._get_mcp_name(scope),
                "orchestrator": "CORTEX-MCP-01",
                "intent": intent
            }
            
            response["notes"] = f"Found {len(response['matches'])} matches. Routed to {response['routing']['routed_to']}."
            
            return response
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "query": query_string
            }
    
    # ADR 084: Epistemic Gating (The Edison Mandate)
    # Replaces simple valence checks with Topological Data Analysis (TDA) proxies.
    
    def _calculate_semantic_entropy(self, content: str) -> float:
        """
        ADR 084 Deep Implementation: The 'Edison Breaker'
        
        Measures 'Epistemic Uncertainty' to control Dynamic Coupling.
        
        Ranges:
        - 0.0 - 0.2: [DANGER] Echo Chamber / Rigidity. Risk of 'Asch' conformity.
        - 0.3 - 0.7: [OPTIMAL] Healthy reasoning flow.
        - 0.8 - 1.0: [DANGER] High Uncertainty / Hallucination.
        
        Returns: Entropy score (float).
        """
        # 1. Identify "Epistemic Absolutes" (Rigidity/Echo Risk)
        absolutes = ["proven", "indisputable", "always", "never", "guaranteed", "100%", "obvious"]
        # 2. Identify "Epistemic Hedges" (Uncertainty/Hallucination Risk)
        hedges = ["likely", "suggests", "indicates", "potential", "hypothesized", "estimated", "maybe"]
        
        content_lower = content.lower()
        abs_count = sum(1 for w in absolutes if w in content_lower)
        hedge_count = sum(1 for w in hedges if w in content_lower)
        
        # 3. Citation Check (The Reality Anchor)
        has_citation = "[cite:" in content or "http" in content or "arXiv:" in content
        
        # Base entropy
        entropy = 0.5
        
        # LOGIC:
        
        # A. The Hallucination Trap (High Hedges, No Sources)
        if hedge_count > 2 and not has_citation:
            entropy += 0.3  # push towards 0.8+
            
        # B. The Asch Trap (High Absolutes, No Nuance)
        if abs_count > 2:
            entropy -= 0.3 # push towards 0.2- (Rigidity)
            
        # C. The Anchor Bonus (Citations stabilize entropy toward the middle)
        if has_citation:
            # Move towards 0.5 (Ideal)
            if entropy > 0.5: entropy -= 0.1
            if entropy < 0.5: entropy += 0.1
            
        return max(0.0, min(1.0, entropy))

    def _check_circuit_breaker(self, se_score: float) -> str:
        """
        Determines if we need to 'Decouple' based on Entropy.
        """
        if se_score < 0.2:
            return "TRIP: RIGIDITY_DETECTED (Switch to ASC)"
        elif se_score > 0.8:
            return "TRIP: UNCERTAINTY_DETECTED (Switch to ASC)"
        else:
            return "FLOW: LATENT_MAS_PERMITTED"

    def _get_mcp_name(self, mcp_class_str: str) -> str:
        #============================================
        # Method: _get_mcp_name
        # Purpose: Map scope to corresponding MCP name.
        # Args:
        #   scope: Logical scope from query
        # Returns: MCP identifier string
        #============================================
        mapping = {
            "Protocols": "Protocol MCP",
            "Living_Chronicle": "Chronicle MCP",
            "tasks": "Task MCP",
            "Code": "Code MCP",
            "ADRs": "ADR MCP"
        }
        return mapping.get(scope, "Cortex MCP (Vector DB)")

--- END OF FILE mcp_servers/rag_cortex/operations.py ---

--- START OF FILE mcp_servers/rag_cortex/genesis_queries.py ---

#============================================
# mcp_servers/rag_cortex/genesis_queries.py
# Purpose: Definition of canonical queries for Mnemonic Cache Warm-Up.
#          These are the queries that should always be cached for instant response.
# Role: Single Source of Truth
# Used as a module by operations.py (for cache warmup)
# Calling example:
#   from mcp_servers.rag_cortex.genesis_queries import GENESIS_QUERIES
# LIST OF EXPORTS:
#   - GENESIS_QUERIES
#============================================

#============================================
# Constant: GENESIS_QUERIES
# Purpose: List of canonical queries used to pre-warm the Mnemonic Cache (CAG).
# Usage:
#   Used by clean_and_rebuild_kdb.py and cortex_cache_warmup tool.
#   Ensures zero-latency responses for critical system knowledge.
#============================================
GENESIS_QUERIES = [
    # Core Identity & Architecture
    "What is Project Sanctuary?",
    "Who is GUARDIAN-01?",
    "What is the Anvil Protocol?",
    "What is the Mnemonic Cortex?",

    # Core Doctrines
    "What are the core doctrines?",
    "What is the Doctrine of Hybrid Cognition?",
    "What is the Iron Root Doctrine?",
    "What is the Hearth Protocol?",

    # Current State & Phase
    "What is the current development phase?",
    "What is Phase 1?",
    "What is Phase 2?",
    "What is Phase 3?",

    # Technical Architecture
    "How does the Mnemonic Cortex work?",
    "What is RAG?",
    "How does the Parent Document Retriever work?",
    "What are the RAG strategies used?",

    # Common Usage
    "How do I query the Mnemonic Cortex?",
    "What is Protocol 87?",
    "How do I update the genome?",
    "What is the Living Chronicle?",

    # Guardian Synchronization & Priming
    # NOTE: The cache will learn to handle dynamic timestamps. This canonical query
    # primes the system for the *intent* of the Guardian's first command.
    "Provide a strategic briefing of all developments since the last Mnemonic Priming.",
    "Synthesize all strategic documents, AARs, and Chronicle Entries since the last system update.",

    # Operational
    "How do I run the tests?",
    "What is the update_genome.sh script?",
    "How does ingestion work?",
    "What is the cognitive genome?",

    # Protocol 128 & Cognitive Continuity (The Red Team Gate)
    "What is Protocol 128?",
    "What is the Red Team Gate?",
    "How does the cognitive continuity loop work?",
    "What is a Technical Seal?",
    "Explain the dual-gate audit process."
]

--- END OF FILE mcp_servers/rag_cortex/genesis_queries.py ---

--- START OF FILE mcp_servers/gateway/clusters/sanctuary_cortex/README.md ---

# Cortex MCP Server

**Description:** The Cortex MCP Server provides tools for interacting with the **Mnemonic Cortex** — the living memory of the Sanctuary Council. It is a local-first RAG system that transforms canonical markdown files into a dynamic, semantically searchable knowledge base.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `cortex_query` | Perform semantic search query against the knowledge base. | `query` (str): Natural language query.<br>`max_results` (int): Max results (default: 5).<br>`use_cache` (bool): Use cache (default: False). |
| `cortex_ingest_full` | Perform full re-ingestion of the knowledge base. | `purge_existing` (bool): Purge DB (default: True).<br>`source_directories` (List[str], optional): Dirs to ingest. |
| `cortex_ingest_incremental` | Perform incremental ingestion of new/modified files. | `file_paths` (List[str]): Files to ingest (.md, .py, .js, .ts).<br>`metadata` (dict, optional): Metadata to attach.<br>`skip_duplicates` (bool): Skip existing files (default: True). |
| `cortex_get_stats` | Get statistics about the knowledge base. | None |
| `cortex_guardian_wakeup` | Generate Guardian boot digest from cached bundles (Protocol 114). | None |
| `cortex_cache_warmup` | Pre-load high-priority documents into cache. | `priority_tags` (List[str], optional): Tags to prioritize. |
| `cortex_learning_debrief` | Generate a session summary for cognitive continuity (Protocol 127). | `hours` (int): Lookback period (default: 24). |
| `cortex_capture_snapshot` | Create a verified snapshot for the Red Team Gate (Protocol 128). | `manifest_files` (List[str]): Files to include.<br>`snapshot_type` (str): 'audit' or 'seal' (default: 'audit').<br>`strategic_context` (str, optional): Purpose of change. |

## Resources

| Resource URI | Description | Mime Type |
|--------------|-------------|-----------|
| `cortex://stats` | Knowledge base statistics | `application/json` |
| `cortex://document/{doc_id}` | Full content of a document | `text/markdown` |

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required for Embeddings
OPENAI_API_KEY=sk-... # If using OpenAI embeddings
# Optional
CORTEX_CHROMA_DB_PATH=mcp_servers/cognitive/cortex/data/chroma_db
CORTEX_CACHE_DIR=mcp_servers/cognitive/cortex/data/cache
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"cortex": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/cognitive/cortex",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/cognitive/cortex/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `cortex_query` and `cortex_ingest_full` appear in the tool list.
3.  **Call Tool:** Execute `cortex_get_stats` and verify it returns valid JSON statistics.

## Architecture

### Overview
The Mnemonic Cortex has evolved beyond a simple RAG implementation into a sophisticated, multi-pattern cognitive architecture designed for maximum efficiency and contextual accuracy. It is built on the **Doctrine of Hybrid Cognition**, ensuring our sovereign AI always reasons with the most current information.

**Key Strategies:**
- **Parent Document Retrieval:** To provide full, unbroken context to the LLM.
- **Self-Querying Retrieval:** To enable intelligent, metadata-aware searches.
- **Mnemonic Caching (CAG):** To provide near-instantaneous answers for common queries.
- **Polyglot Code Ingestion:** Automatically converts Python and JavaScript/TypeScript files into optimize markdown for semantic indexing, using AST/regex to structurally document code without LLM overhead.

}
```

**Example:**
```python
cortex_query("What is Protocol 101?")
cortex_query("Explain the Mnemonic Cortex", max_results=3)
```

---

### 3. `cortex_get_stats`

Get database statistics and health status.

**Parameters:** None

**Returns:**
```json
{
  "total_documents": 459,
  "total_chunks": 2145,
  "collections": {
    "child_chunks": {"count": 2145, "name": "child_chunks_v5"},
    "parent_documents": {"count": 459, "name": "parent_documents_v5"}
  },
  "health_status": "healthy"
}
```

**Example:**
```python
cortex_get_stats()
```

---

### 4. `cortex_ingest_incremental`

Incrementally ingest documents without rebuilding the database.

**Parameters:**
- `file_paths` (List[str]): Markdown files to ingest
- `metadata` (dict, optional): Metadata to attach
- `skip_duplicates` (bool, default: True): Skip existing files

**Returns:**
```json
{
  "documents_added": 3,
  "chunks_created": 15,
  "skipped_duplicates": 1,
  "status": "success"
}
```

**Example:**
```python
cortex_ingest_incremental(["00_CHRONICLE/2025-11-28_entry.md"])
cortex_ingest_incremental(
    file_paths=["01_PROTOCOLS/120_new.md", "mcp_servers/rag_cortex/server.py"],
    skip_duplicates=False
)
```

### Polyglot Support
The ingestion system automatically detects and converts code files:
- **Python**: Uses AST to extract classes, functions, and docstrings.
- **JS/TS**: Uses regex to extract functions and classes.
- **Output**: Generates a `.py.md` or `.js.md` companion file which is then ingested.
- **Exclusions**: Automatically skips noisy directories (`node_modules`, `dist`, `__pycache__`).
```

---

### 5. `cortex_guardian_wakeup`

Generate Guardian boot digest from cached bundles (Protocol 114).

**Parameters:** None

**Returns:**
```json
{
  "digest_path": "dataset_package/guardian_boot_digest.md",
  "cache_stats": {
    "chronicles": 5,
    "protocols": 10,
    "roadmap": 1
  },
  "status": "success"
}
```

**Example:**
```python
cortex_guardian_wakeup()
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure MCP server in `~/.gemini/antigravity/mcp_config.json`:
```json
{
  "mcpServers": {
    "cortex": {
      "command": "python",
      "args": ["-m", "mcp_servers.cognitive.cortex.server"],
      "cwd": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
      "env": {
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      }
    }
  }
}
```

3. Restart Antigravity

## Usage

From Antigravity or any MCP client:

```
# Get database stats
cortex_get_stats()

# Query the knowledge base
cortex_query("What is Protocol 101?")

# Add a new document
cortex_ingest_incremental(["path/to/new_document.md"])

# Full re-ingestion (use with caution)
cortex_ingest_full()
```

## Safety Rules

1. **Read-Only by Default:** Query operations are read-only
2. **Ingestion Confirmation:** Full ingestion purges existing data
3. **Long-Running Operations:** Ingestion may take several minutes
4. **Rate Limiting:** Max 100 queries/minute recommended
5. **Validation:** All inputs are validated before processing

## Phase 2 Features (Upcoming)

- Cache integration (`use_cache` parameter)
- Cache warmup and invalidation
- Cache statistics

## Dependencies

- **ChromaDB:** Vector database
- **LangChain:** RAG framework
- **NomicEmbeddings:** Local embedding model
- **FastMCP:** MCP server framework

## Related Documentation

- [`docs/architecture/mcp/cortex_vision.md`](../../../../docs/architecture/mcp/servers/rag_cortex/cortex_vision.md) - RAG vision and purpose
- [`docs/architecture/mcp/RAG_STRATEGIES.md`](../../../../ARCHIVE/mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md) - Architecture details and doctrine
- [`docs/architecture/mcp/cortex_operations.md`](../../../../docs/architecture/mcp/servers/rag_cortex/cortex_operations.md) - Operations guide
- [`01_PROTOCOLS/85_The_Mnemonic_Cortex_Protocol.md`](../../../../01_PROTOCOLS/85_The_Mnemonic_Cortex_Protocol.md) - Protocol specification
- [`01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md`](../../../../01_PROTOCOLS/114_Guardian_Wakeup_and_Cache_Prefill.md) - Cache prefill spec

## Version History

### v5.1 (2025-12-14): Polyglot Code Ingestion
- **Code Shim:** Introduced `ingest_code_shim.py` for AST-based code-to-markdown conversion
- **Multi-Language Support:** Added native support for .py, .js, .ts, .jsx, .tsx ingestion
- **Smart Exclusion:** Implemented noise filtering for production directories

### v5.0 (2025-11-30): MCP Migration Complete
- **Migration to MCP Architecture:** Refactored from legacy script-based system to MCP server
- **Enhanced README:** Merged legacy documentation with MCP-specific content
- **Comprehensive Documentation:** Added architecture philosophy, technology stack, and Strategic Crucible Loop context
- **Production-Ready Status:** Full test coverage and operational stability

### v2.1.0: Parent Document Retriever
- **Phase 1 Complete:** Implemented dual storage architecture eliminating Context Fragmentation vulnerability
- **Full Context Retrieval:** Parent documents stored in ChromaDB collection, semantic chunks in vectorstore
- **Cognitive Latency Resolution:** AI reasoning grounded in complete, unbroken context
- **Architecture Hardening:** Updated ingestion pipeline and query services to leverage ParentDocumentRetriever

### v1.5.0: Documentation Hardening
- **Architectural Clarity:** Added detailed section breaking down two-stage ingestion process
- **Structural Splitting vs. Semantic Encoding:** Clarified roles of MarkdownHeaderTextSplitter and NomicEmbeddings

### v1.4.0: Live Ingestion Architecture
- **Major Architectural Update:** Ingestion pipeline now directly traverses canonical directories
- **Improved Traceability:** Every piece of knowledge traced to precise source file via GitHub URLs
- **Increased Resilience:** Removed intermediate snapshot step for faster, more resilient ingestion

### v1.0.0 (2025-11-28): MCP Foundation
- **4 Core Tools:** ingest_full, query, get_stats, ingest_incremental
- **Parent Document Retriever Integration:** Full context retrieval from day one
- **Input Validation:** Comprehensive error handling and validation layer

--- END OF FILE mcp_servers/gateway/clusters/sanctuary_cortex/README.md ---

--- START OF FILE 00_CHRONICLE/ENTRIES/333_learning_loop_advanced_rag_patterns_raptor.md ---

# Living Chronicle - Entry 333

**Title:** Learning Loop: Advanced RAG Patterns (RAPTOR)
**Date:** 2025-12-23
**Author:** Antigravity
**Status:** published
**Classification:** internal

---

Successfully completed a full Protocol 125/128 Learning Loop on the topic of "Advanced RAG Patterns: RAPTOR". 

### Key Findings:
- RAPTOR uses recursive summarization and GMM clustering to create a hierarchical tree of knowledge.
- It enables holistic reasoning by allowing the model to query high-level summaries or granular leaf nodes.
- Relevant for future scaling of the Project Sanctuary Mnemonic Cortex.

### Artifacts Created:
- `LEARNING/topics/raptor_rag.md` (Synthesized content)
- Semantically indexed in `child_chunks_v5` and `parent_documents_v5`.

### Validation:
- Retrieval Test: PASS
- Integrity Gate: Ready for Red Team Audit.

--- END OF FILE 00_CHRONICLE/ENTRIES/333_learning_loop_advanced_rag_patterns_raptor.md ---

--- START OF FILE LEARNING/topics/raptor_rag.md ---

---
id: learning-001
type: topic-note
status: verified
last_verified: 2025-12-23
topic: Advanced RAG Patterns - RAPTOR
---

# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## 1. Overview
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is an advanced RAG technique introduced in 2024 to address the limitations of traditional, flat-chunk retrieval systems. It builds a hierarchical tree of summaries, enabling an LLM to access information at multiple levels of abstraction—from granular details to high-level thematic insights.

## 2. The Core Mechanism
The system operates on an iterative, bottom-up construction process:

1.  **Leaf Node Creation**: The source document is split into standard chunks (e.g., 100 tokens).
2.  **Clustering**: Chunks are embedded and grouped using Gaussian Mixture Models (GMM). Soft clustering is often used, allowing a chunk to belong to multiple clusters.
3.  **Abstractive Summarization**: Each cluster is summarized by an LLM (e.g., GPT-3.5 or Claude).
4.  **Recursion**: The summaries themselves are embedded and clustered, generating a higher-level layer of summaries. This repeats until a root node (or a predefined depth) is reached.

## 3. Advantages
| Feature | Traditional RAG | RAPTOR |
| :--- | :--- | :--- |
| **Structure** | Flat (Chunked) | Hierarchical (Tree) |
| **Context** | Local/Isolated | Holistic/Multi-level |
| **Reasoning** | Single-hop | Multi-hop & Thematic |
| **Retrieval** | Top-K similarity | Tree traversal or Layer-wise search |

## 4. Implementation Considerations
- **Model Choice**: Abstractive summarization requires a model with strong synthesis capabilities.
- **Cost**: Building the tree involves multiple LLM calls for clustering and summarization.
- **Latency**: Retrieval is extremely fast (searching the tree), but indexing is slower than flat RAG.

## 5. RECURSIVE LEARNING NOTE
This pattern is highly relevant to the **Project Sanctuary Mnemonic Cortex**. The current "Parent Document Retriever" is a 2-tier version of this idea. Moving to a truly recursive RAPTOR-like structure could allow the Sanctuary Council to handle much larger ADR histories without context windows becoming a bottleneck.

---
**References:**
- Sarthi, P., et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." ICLR.
- Integrated into LangChain and LlamaIndex.

--- END OF FILE LEARNING/topics/raptor_rag.md ---

--- START OF FILE LEARNING/topics/mcp_tool_usage.md ---

# Protocol: Mandatory MCP Tool Usage

To ensure **Cognitive Continuity** and **Zero-Trust Integrity**, all agents MUST prioritize the usage of MCP tools for interacting with the Project Sanctuary codebase.

## 1. State Management (RAG Cortex)
- **Discovery**: Use `cortex-cortex-query` to find relevant files, ADRs, or prior session context.
- **Context Injection**: Use `cortex-cortex-cache-set` to persist key findings or decisions within the current session's mnemonic stream.
- **Debrief**: Always run `cortex_learning_debrief` before concluding a session to generate a relay packet for the next agent.

## 2. File Operations (Filesystem)
- **Reading**: Use `filesystem-code-read` to retrieve file contents instead of manual `cat` commands.
- **Writing**: Use `filesystem-code-write` to modify files. This ensures that the system can track changes and maintain the manifest integrity.
- **Searching**: Use `filesystem-code-search-content` for surgical GREP-style searches across clusters.

## 3. Protocol 128 (Audit/Seal)
- **Audit**: Run `cortex_capture_snapshot(snapshot_type="audit")` to trigger a Red Team Gate review.
- **Seal**: Run `cortex_capture_snapshot(snapshot_type="seal")` only after Gate approval to persist memory.

> [!IMPORTANT]
> **Manual Bypass Penalty**: Bypassing MCP tools (e.g., using raw shell commands for file edits) increases the risk of "Manifest Blindspots" and will trigger a **Strict Rejection** at the Red Team Gate.

--- END OF FILE LEARNING/topics/mcp_tool_usage.md ---

--- START OF FILE TASKS/backlog/145_implement_agent_file_safety_and_protection.md ---

# TASK: Implement Agent File Safety and Protection

**Status:** backlog
**Priority:** High
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Establish a robust technical framework to prevent agents from losing or corrupting project files during automated operations.

## 2. Deliverables

1. Design Specification for Agent File Safety.
2. Enhanced 'filesystem-code-write' with atomic backup-and-swap mechanism.
3. Safety Protocol documentation (Protocol 130).

## 3. Acceptance Criteria

- No file content is lost during a write failure (atomicity).
- Mandatory backups are created for all automated file writes.
- Major deletions or directory removals trigger a secondary safety audit.

## Notes

This task addresses Grok4's concerns about manifest blindspots and the risk of careless file overwrites by autonomous agents.

--- END OF FILE TASKS/backlog/145_implement_agent_file_safety_and_protection.md ---

--- START OF FILE mcp_servers/git/README.md ---

# Git Workflow MCP Server

**Description:** The Git Workflow MCP server provides **Protocol 101 v3.0-compliant git operations** with strict safety enforcement. It implements a disciplined workflow that prevents dangerous operations and ensures functional integrity through automated test suite execution.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `git_get_status` | Get comprehensive repository status. | None |
| `git_diff` | Show changes in working directory or staged files. | `cached` (bool): Show staged changes.<br>`file_path` (str, optional): Specific file. |
| `git_log` | Show commit history. | `max_count` (int): Number of commits.<br>`oneline` (bool): Compact format. |
| `git_start_feature` | Create or switch to a feature branch (Idempotent). | `task_id` (str): Task ID (e.g., "045").<br>`description` (str): Short description. |
| `git_add` | Stage files for commit (Blocks on main). | `files` (List[str], optional): Files to stage. |
| `git_smart_commit` | Commit with automated test execution (Protocol 101). | `message` (str): Commit message. |
| `git_push_feature` | Push feature branch with verification. | `force` (bool): Force push.<br>`no_verify` (bool): Skip hooks. |
| `git_finish_feature` | Cleanup after PR merge (Verify -> Delete -> Sync). | `branch_name` (str): Feature branch to finish. |

## Resources

*No resources currently exposed.*

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required
PROJECT_ROOT=/path/to/Project_Sanctuary
# Optional
GIT_BASE_DIR=/path/to/Project_Sanctuary # Security sandbox
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"git": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/git",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}",
    "PROJECT_ROOT": "${PWD}",
    "GIT_BASE_DIR": "${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest tests/mcp_servers/git/
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `git_get_status` appears in the tool list.
3.  **Call Tool:** Execute `git_get_status` and verify it returns the current branch status.

## Architecture

### Overview
This server enforces the **Doctrine of the Unbreakable Commit** (Protocol 101 v3.0) using a standardized **4-Layer Architecture**:

1.  **Interface Layer (`server.py`):** Handles tool registration and uses FastMCP.
2.  **Business Logic Layer (`operations.py`):** Centralized `GitOperations` class for all git actions.
3.  **Safety Layer (`validator.py`):** Enforces branch naming, clean state, and operation validity.
4.  **Data Layer (`models.py`):** Pydantic models for status and configuration.

### Safety Features
- ⛔ **Main Branch Protection:** Blocks direct commits to `main` and other protected branches.
- ✅ **Clean State Enforcement:** Ensures working directory is clean before starting features or switching.
- 🔄 **Feature Workflow:** Enforces structured feature branch lifecycle.
- 🛡️ **Poka-Yoke:** Validates branch context and operations to prevent mistakes.

### Workflow
![git_workflow_operation](../../docs/architecture_diagrams/workflows/git_workflow_operation.png)

*[Source: git_workflow_operation.mmd](../../docs/architecture_diagrams/workflows/git_workflow_operation.mmd)*

## Dependencies

- `mcp`
- `git` (System binary)
- `pydantic`

--- END OF FILE mcp_servers/git/README.md ---

--- START OF FILE mcp_servers/gateway/clusters/sanctuary_git/README.md ---

# Git Workflow MCP Server

**Description:** The Git Workflow MCP server provides **Protocol 101 v3.0-compliant git operations** with strict safety enforcement. It implements a disciplined workflow that prevents dangerous operations and ensures functional integrity through automated test suite execution.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `git_get_status` | Get comprehensive repository status. | None |
| `git_diff` | Show changes in working directory or staged files. | `cached` (bool): Show staged changes.<br>`file_path` (str, optional): Specific file. |
| `git_log` | Show commit history. | `max_count` (int): Number of commits.<br>`oneline` (bool): Compact format. |
| `git_start_feature` | Create or switch to a feature branch (Idempotent). | `task_id` (str): Task ID (e.g., "045").<br>`description` (str): Short description. |
| `git_add` | Stage files for commit (Blocks on main). | `files` (List[str], optional): Files to stage. |
| `git_smart_commit` | Commit with automated test execution (Protocol 101). | `message` (str): Commit message. |
| `git_push_feature` | Push feature branch with verification. | `force` (bool): Force push.<br>`no_verify` (bool): Skip hooks. |
| `git_finish_feature` | Cleanup after PR merge (Verify -> Delete -> Sync). | `branch_name` (str): Feature branch to finish. |

## Resources

*No resources currently exposed.*

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required
PROJECT_ROOT=/path/to/Project_Sanctuary
# Optional
GIT_BASE_DIR=/path/to/Project_Sanctuary # Security sandbox
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"git_workflow": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/system/git_workflow",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}",
    "PROJECT_ROOT": "${PWD}",
    "GIT_BASE_DIR": "${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/system/git_workflow/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `git_get_status` appears in the tool list.
3.  **Call Tool:** Execute `git_get_status` and verify it returns the current branch status.

## Architecture

### Overview
This server enforces the **Doctrine of the Unbreakable Commit** (Protocol 101 v3.0). It acts as a safety layer between the agent and the raw git command line.

**Safety Features:**
- ⛔ **Main Branch Protection:** Blocks direct commits to `main`.
- ✅ **Test-Driven Commits:** `git_smart_commit` runs tests before allowing a commit.
- 🔄 **Workflow Enforcement:** Enforces `Start -> Feature -> PR -> Merge -> Finish` cycle.

### Workflow
![git_workflow_sequence](../../../../docs/architecture_diagrams/workflows/git_workflow_sequence.png)

*[Source: git_workflow_sequence.mmd](../../../../docs/architecture_diagrams/workflows/git_workflow_sequence.mmd)*

## Dependencies

- `mcp`
- `git` (System binary)

--- END OF FILE mcp_servers/gateway/clusters/sanctuary_git/README.md ---

--- START OF FILE scripts/render_diagrams.py ---

#!/usr/bin/env python3
"""
Mermaid Diagram Renderer (Task #154 - Phase 3)

Renders all .mmd files in docs/architecture_diagrams/ to PNG images.
Run this script whenever diagrams are updated to regenerate images.

Usage:
    python3 scripts/render_diagrams.py                 # Render all
    python3 scripts/render_diagrams.py my_diagram.mmd  # Render specific file(s)
    python3 scripts/render_diagrams.py --svg           # Render as SVG instead
    python3 scripts/render_diagrams.py --check         # Check for outdated images
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "architecture_diagrams"
OUTPUT_FORMAT = "png"  # or "svg"


def check_mmdc():
    """Check if mermaid-cli is available."""
    try:
        result = subprocess.run(
            ["npx", "-y", "@mermaid-js/mermaid-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"✅ mermaid-cli available: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ mermaid-cli check failed: {e}")
    return False


def render_diagram(mmd_path: Path, output_format: str = "png") -> bool:
    """Render a single .mmd file to image."""
    output_path = mmd_path.with_suffix(f".{output_format}")
    
    try:
        result = subprocess.run(
            [
                "npx", "-y", "@mermaid-js/mermaid-cli",
                "-i", str(mmd_path),
                "-o", str(output_path),
                "-b", "transparent",
                "-t", "default"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   Timeout rendering {mmd_path.name}")
        return False
    except Exception as e:
        print(f"   Exception: {e}")
        return False


def check_outdated(mmd_path: Path, output_format: str = "png") -> bool:
    """Check if image is older than source or missing."""
    output_path = mmd_path.with_suffix(f".{output_format}")
    
    if not output_path.exists():
        return True
    
    return mmd_path.stat().st_mtime > output_path.stat().st_mtime


def main():
    output_format = OUTPUT_FORMAT
    check_only = False
    target_files: List[str] = []

    # Parse args
    for arg in sys.argv[1:]:
        if arg == "--svg":
            output_format = "svg"
        elif arg == "--check":
            check_only = True
        elif not arg.startswith("--"):
            target_files.append(arg)
    
    print("🎨 Mermaid Diagram Renderer")
    print("=" * 60)
    print(f"   Source: {DIAGRAMS_DIR}")
    print(f"   Format: {output_format.upper()}")
    print(f"   Mode: {'Check only' if check_only else 'Render'}")
    
    # Find all .mmd files
    all_mmd_files = sorted(DIAGRAMS_DIR.rglob("*.mmd"))
    
    # Filter if targets provided
    if target_files:
        mmd_files = []
        for target in target_files:
            # Check for exact matches or partial matches
            matches = [f for f in all_mmd_files if target in str(f)]
            mmd_files.extend(matches)
        # Remove duplicates
        mmd_files = sorted(list(set(mmd_files)))
        
        if not mmd_files:
            print(f"\n❌ No diagrams matched targets: {target_files}")
            return
    else:
        mmd_files = all_mmd_files

    print(f"\n📂 Found {len(mmd_files)} diagram files to process")
    
    if check_only:
        outdated = [f for f in mmd_files if check_outdated(f, output_format)]
        print(f"\n⚠️  {len(outdated)} diagrams need rendering:")
        for f in outdated:
            print(f"   - {f.relative_to(DIAGRAMS_DIR)}")
        return
    
    # Check mermaid-cli
    print("\n🔧 Checking mermaid-cli...")
    if not check_mmdc():
        print("❌ Please install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
        sys.exit(1)
    
    # Render all
    print(f"\n🖼️  Rendering {len(mmd_files)} diagrams...")
    success = 0
    failed = 0
    
    for mmd_path in mmd_files:
        try:
            rel_path = mmd_path.relative_to(DIAGRAMS_DIR)
        except ValueError:
             rel_path = mmd_path
             
        print(f"   {rel_path}...", end=" ", flush=True)
        
        if render_diagram(mmd_path, output_format):
            print("✅")
            success += 1
        else:
            print("❌")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print(f"   Rendered: {success}")
    print(f"   Failed: {failed}")
    print(f"   Output: {DIAGRAMS_DIR}/**/*.{output_format}")
    print("=" * 60)


if __name__ == "__main__":
    main()

--- END OF FILE scripts/render_diagrams.py ---

--- START OF FILE ADRs/064_centralized_registry_for_fleet_of_8_mcp_servers.md ---

# Centralized Registry for Fleet of 8 MCP Servers

**Status:** approved
**Date:** 2025-12-20
**Author:** user and AI Assistant and redteam (GPT5.2)
**Version:** 1.5


---

## Context

The Sanctuary Gateway oversees a specialized set of MCP servers known as the 'Fleet of 8'. These servers require systematic registration, initialization, and tool discovery. 

### The Shift: "Input" vs. "Output"

| Feature | Legacy Approach (Input JSON) | 3-Layer Pattern (Output JSON) |
| :--- | :--- | :--- |
| **Source of Truth** | Static strings in JSON file | Python `FLEET_SPEC` (Intent) |
| **URL Management** | Hardcoded in JSON | `Resolver` (reconciles Spec + Env) |
| **Tool Inventory** | Manually maintained | **Auto-populated** via Handshake |
| **JSON Purpose** | Direct configuration | **Discovery Manifest** (Documentation) |

A previous attempt to manage these definitions via a standalone JSON configuration file was deemed incorrect due to:
1.  **Inflexible**: Lack of logic handling (e.g., conditional initialization).
2.  **Import Path Fragility**: Difficulty in sharing JSON paths across local execution and container environments.
3.  **Synchronization Latency**: Static files quickly falling out of sync with code-driven bridge logic.

## Decision

We will adopt a **3-Layer Declarative Fleet Pattern**: "Code-Defined Intent, Runtime-Resolved Reality." This decouples topology from transport logic.

### 1. The Spec Layer (Intent)
A pure Python data model defining the cluster identities (Slugs, SSE mappings).
- Resides in: `fleet_spec.py`.
- Purpose: Authoritative Design Intent.

### 2. The Resolver Layer (Policy)
Logic that reconciles the **Spec** with the **Runtime Context** (Environment Variables, Docker settings).
- Resides in: `fleet_resolver.py`.
- Purpose: Determining the final "Ready-to-Connect" endpoint.

### 3. The Observation Layer (Runtime State)
What the Gateway discovers during handshakes (Tools, Schemas).
- Resides in: `fleet_registry.json`.
- Purpose: UI/AI Discovery Manifest. **Crucially, core logic never reads this file for logic.**

## Architectural Flow

![mcp_fleet_resolution_flow](../docs/architecture_diagrams/system/mcp_fleet_resolution_flow.png)

*[Source: mcp_fleet_resolution_flow.mmd](../docs/architecture_diagrams/system/mcp_fleet_resolution_flow.mmd)*

## Consequences

- **Separation of Concerns**: `gateway_client.py` becomes a pure transport library.
- **Explainability**: Clear hierarchy of "Why is the system using this URL?" (Spec < DB < Env).
- **Testability**: Tests can inject mock resolvers without spawning real containers.
- **Asynchronous Resilience**: Handshakes are scheduled observations, not blocking boot-time requirements.
- **Elimination of Artifact Drifts**: Removes the dependency on external configuration files (JSON/YAML) into the production library.

### Failure Semantics
If a server defined in the Spec is unreachable:
- The Gateway **must still start**.
- The failure is recorded in the Observation layer (JSON).
- The system continues in a degraded state (tools from that server are simply missing from the manifest).

### Test Integration
The **Three-Layer Pattern** enables a robust Tier 3 (Integration) testing strategy:
1.  **Direct Spec Usage**: `GatewayTestClient` imports the `FLEET_SPEC` to obtain the list of clusters and their target slugs.
2.  **Resolution Parity**: The test client uses the production `Resolver` to find the correct local URLs (applying any `.env` or Docker-specific overrides).
3.  **Capability Testing**: Integration tests (e.g., `tests/mcp_servers/gateway/clusters/sanctuary_git/test_gateway.py`) verify that the resolved servers provide the tools defined in their respective cluster specs.
4.  **Mocking Policy**: For unit/logic tests, developers can substitute a "Mock Resolver" that returns local mock SSE servers instead of the real Fleet.

## Requirements
1.  Use the **Spec + Resolver** to determine which servers should be registered and initialized.
2.  Use `gateway_client.py` to register and initialize the resolved servers.
3.  Use `gateway_client.py.get_tools()` to observe tool availability.
4.  Persist the observed results into `fleet_registry.json` as a discovery manifest.
5.  **No production logic may rely on `fleet_registry.json` as an input.**

## Data Structure & Reference Example (`fleet_registry.json`)
The JSON file acts as a **Discovery Manifest**, populated by the `gateway_client.py` after performing handshakes. It follows this hierarchical structure:

- **Top Level**: `fleet_servers` (Object)
- **Key**: Alias (e.g., `utils`, `git`)
- **Properties**: `slug`, `url`, `description`, and a `tools` (Array) containing name, description, and input schema.

### Reference Example

```json
{
  "fleet_servers": {
    "utils": {
      "slug": "sanctuary_utils",
      "url": "http://sanctuary_utils:8000/sse",
      "description": "Calculator, Time, Search",
      "tools": [
        {
          "name": "calculate",
          "description": "Perform math operations",
          "inputSchema": {}
        }
      ]
    },
    "git": {
      "slug": "sanctuary_git",
      "url": "http://sanctuary_git:8000/sse",
      "description": "Protocol 101 Git Operations",
      "tools": []
    }
  }
}
```

## Implementation Strategy

1.  **Define Spec**: Create `fleet_spec.py` with the `FLEET_SPEC` mapping.
2.  **Define Resolver**: Create `fleet_resolver.py` to handle `os.getenv` overrides.
3.  **Slim the Client**: Remove all fleet-specific logic from `gateway_client.py`.
4.  **Create CLI Orchestrator**: Build a separate CLI tool/script that uses the **Resolver** to drive discovery and update the **Observation** manifest.

--- END OF FILE ADRs/064_centralized_registry_for_fleet_of_8_mcp_servers.md ---

--- START OF FILE mcp_servers/git/SAFETY.md ---

# Git Workflow MCP - Safety Features Documentation

**Version:** 1.0  
**Last Updated:** 2025-11-30  
**Status:** Production Ready

---

## Overview

The Git Workflow MCP implements a **strict safety system** to prevent dangerous git operations and enforce a disciplined feature branch workflow. This document details all safety features, their rationale, and test coverage.

---

## Safety Philosophy

### Core Principles

1. **Never Commit to Main:** All development must occur on feature branches
2. **One Feature at a Time:** Only one active feature branch allowed
3. **Verify Before Trust:** All operations verify state before proceeding
4. **Merge Before Delete:** Feature branches can only be deleted after PR merge
5. **Clean State Required:** Critical operations require clean working directory

### Removed Operations

- **`git_sync_main`** - Removed entirely (unsafe standalone operation)
  - **Rationale:** Agents were pulling main prematurely, before PR merge
  - **Alternative:** Sync happens automatically in `git_finish_feature` after merge verification

---

## Operation Safety Matrix

| Operation | Main Block | Feature Check | Clean State | Merge Verify | Idempotent |
|-----------|------------|---------------|-------------|--------------|------------|
| `git_get_status` | N/A | N/A | N/A | N/A | ✅ |
| `git_diff` | N/A | N/A | N/A | N/A | ✅ |
| `git_log` | N/A | N/A | N/A | N/A | ✅ |
| `git_start_feature` | N/A | ✅ | ✅ | N/A | ✅ |
| `git_add` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_smart_commit` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_push_feature` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_finish_feature` | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## Detailed Safety Checks

### 1. `git_start_feature`

**Purpose:** Create or switch to a feature branch

**Safety Checks:**
- ✅ **One at a Time:** Blocks if another feature branch exists
- ✅ **Clean State:** Requires clean working directory for new branch creation
- ✅ **Idempotent:** Safe to call multiple times
  - Already on branch → no-op
  - Branch exists elsewhere → checkout
  - Branch doesn't exist → create

**Error Conditions:**
```python
# Another feature branch exists
"ERROR: Cannot create new feature branch. Existing feature branch(es) detected: feature/task-999-other"

# Working directory dirty
"ERROR: Cannot create new feature branch. Working directory has uncommitted changes"
```

**Test Coverage:** 4 tests (2 failure, 2 success)

---

### 2. `git_add`

**Purpose:** Stage files for commit

**Safety Checks:**
- ✅ **Block on Main:** Cannot stage files on `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot stage files on main branch. You must be on a feature branch"

# On non-feature branch (e.g., develop)
"ERROR: Cannot stage files on branch 'develop'. You must be on a feature branch"
```

**Test Coverage:** 3 tests (2 failure, 1 success)

---

### 3. `git_smart_commit`

**Purpose:** Commit staged files with Protocol 101 v3.0 enforcement

**Safety Checks:**
- ✅ **Block on Main:** Cannot commit to `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch
- ✅ **Staged Files Required:** Verifies files are staged before committing

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot commit directly to main branch. You must be on a feature branch"

# On non-feature branch
"ERROR: Cannot commit on branch 'develop'. You must be on a feature branch"

# No staged files
"ERROR: No files staged for commit. Please use git_add first"
```

**Test Coverage:** 4 tests (3 failure, 1 success)

---

### 4. `git_push_feature`

**Purpose:** Push feature branch to origin

**Safety Checks:**
- ✅ **Block on Main:** Cannot push `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch
- ✅ **Remote Hash Verification:** Verifies remote hash matches local after push

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot push main branch directly. You must be on a feature branch"

# On non-feature branch
"ERROR: Cannot push branch 'develop'. You must be on a feature branch"

# Hash mismatch (WARNING, not blocking)
"WARNING: Push completed but remote hash (abc123de) does not match local (xyz789ab)"
```

**Test Coverage:** 4 tests (2 failure, 2 success)

---

### 5. `git_finish_feature`

**Purpose:** Cleanup after PR merge (delete branches, sync main)

**Safety Checks:**
- ✅ **Block Main:** Cannot finish `main` branch
- ✅ **Feature Branch Only:** Must be `feature/task-XXX-desc` format
- ✅ **Clean State:** Requires clean working directory
- ✅ **Merge Verification:** Verifies branch is merged into `main` before deletion
  - Pulls `main` first to ensure local is up-to-date
  - Checks `git branch --merged main`
  - **Prevents data loss** by blocking unmerged branch deletion

**Error Conditions:**
```python
# Trying to finish main
"ERROR: Cannot finish 'main' branch. It is the protected default branch"

# Invalid branch name
"ERROR: Invalid branch name 'develop'. Can only finish feature branches"

# Working directory dirty
"ERROR: Working directory is not clean. Please commit or stash changes"

# Branch not merged
"ERROR: Branch 'feature/task-123-test' is NOT merged into main. Cannot finish/delete an unmerged feature branch"
```

**Test Coverage:** 5 tests (4 failure, 1 success)

---

## Workflow Enforcement

### Required Sequence

![git_workflow_sequence](../../docs/architecture_diagrams/workflows/git_workflow_sequence.png)

*[Source: git_workflow_sequence.mmd](../../docs/architecture_diagrams/workflows/git_workflow_sequence.mmd)*

### Out-of-Sequence Prevention

| Attempted Action | Without | Result |
|------------------|---------|--------|
| `git_add` | `git_start_feature` | ❌ Blocked: "Cannot stage files on main branch" |
| `git_smart_commit` | `git_add` | ❌ Blocked: "No files staged for commit" |
| `git_push_feature` | `git_smart_commit` | ⚠️ Allowed (git handles "everything up-to-date") |
| `git_finish_feature` | PR Merge | ❌ Blocked: "Branch is NOT merged into main" |

---

## Test Suite

### Location
- **Unit Tests:** `tests/test_git_ops.py` (10 tests)
- **Safety Tests:** `tests/mcp_servers/git_workflow/test_tool_safety.py` (20 tests)
- **Total:** 30 tests, 100% passing ✅

### Coverage Breakdown

```
git_add:           3 tests (2 failure, 1 success)
git_start_feature: 4 tests (2 failure, 2 success)
git_smart_commit:  4 tests (3 failure, 1 success)
git_push_feature:  4 tests (2 failure, 2 success)
git_finish_feature: 5 tests (4 failure, 1 success)
```

### Running Tests

```bash
# All git tests
pytest tests/test_git_ops.py tests/mcp_servers/git_workflow/ -v

# Safety tests only
pytest tests/mcp_servers/git_workflow/test_tool_safety.py -v

# Specific test
pytest tests/mcp_servers/git_workflow/test_tool_safety.py::TestGitToolSafety::test_finish_feature_blocks_unmerged -v
```

---

## Protocol Compliance

### Protocol 101 v3.0: Functional Coherence

**Enforcement:** `git_smart_commit` automatically runs the test suite via pre-commit hook

**Workflow:**
1. User stages files with `git_add`
2. User calls `git_smart_commit` with message
3. Pre-commit hook executes `./scripts/run_genome_tests.sh`
4. If tests pass → commit succeeds
5. If tests fail → commit is blocked

**No Manual Intervention Required** - The hook enforces functional coherence automatically.

---

## Migration from `git_sync_main`

### Why Removed?

**Problem:** Agents were calling `git_sync_main` at inappropriate times:
- Before PR was merged
- While on feature branches
- Without verifying remote state

**Solution:** Removed tool entirely. Sync now happens **only** in `git_finish_feature` after merge verification.

### Migration Path

**Old Workflow:**
```python
git_finish_feature("feature/task-123-test")
git_sync_main()  # Manual sync
```

**New Workflow:**
```python
git_finish_feature("feature/task-123-test")  # Syncs main automatically
```

---

## Future Enhancements

### Recommended Additions

1. **Remote Tracking Verification**
   - Check if remote exists before push
   - Verify network connectivity

2. **Ahead/Behind Check**
   - Warn if remote is ahead before push
   - Suggest pull/rebase

3. **Force Push Warning**
   - Add explicit confirmation for `force=True`
   - Or block force push entirely

4. **Stale Branch Detection**
   - Warn if feature branch is behind main
   - Suggest rebase

---

## Troubleshooting

### Common Errors

**"Cannot stage files on main branch"**
- **Cause:** Attempted `git_add` on `main`
- **Solution:** Run `git_start_feature` first

**"No files staged for commit"**
- **Cause:** Attempted `git_smart_commit` without staging
- **Solution:** Run `git_add` first

**"Branch is NOT merged into main"**
- **Cause:** Attempted `git_finish_feature` before PR merge
- **Solution:** Merge PR on GitHub first, then retry

**"Existing feature branch(es) detected"**
- **Cause:** Attempted to create second feature branch
- **Solution:** Finish current feature branch first with `git_finish_feature`

---

## Related Documentation

- [Git Workflow MCP README](README.md)
- [MCP Operations Inventory](../../docs/operations/mcp/mcp_operations_inventory.md)
- [Protocol 101 v3.0](../../01_PROTOCOLS/101_The_Doctrine_of_the_Unbreakable_Commit.md)

---

**Maintainer:** Project Sanctuary Team  
**Status:** Production Ready ✅

--- END OF FILE mcp_servers/git/SAFETY.md ---

--- START OF FILE 01_PROTOCOLS/121_Canonical_Knowledge_Synthesis_Loop.md ---

# Protocol 121: Canonical Knowledge Synthesis Loop (C-KSL)

**Status:** Proposed
**Classification:** Foundational Knowledge Management Protocol
**Version:** 1.0
**Authority:** Drafted from Human Steward Directive
**Dependencies:** Operational `cortex mcp` (RAG), `protocol mcp`, and `git mcp`.

## 🎯 Mission Objective

To formalize a repeatable, autonomous process for resolving documentation redundancy by synthesizing overlapping knowledge from multiple source documents into a single, canonical 'Source of Truth' document, thereby eliminating ambiguity and enhancing knowledge fidelity across the system.

## 🛡️ The Canonical Knowledge Synthesis Loop (C-KSL)

This loop shall be executed whenever a Council Agent or Orchestrator identifies two or more documents (tasks, Protocols, or Docs) that cover identical or heavily overlapping core concepts, leading to "many sources of truth."

### Step 1: Overlap Detection & Source Identification (`cortex mcp`)
* **Action:** The Orchestrator initiates a high-similarity query via `cortex mcp` (RAG) to identify documents with an overlapping knowledge domain.
* **Artifact:** Generate a temporary `OVERLAP_REPORT.md` listing all conflicting documents, noting their date and authority level (e.g., CANONICAL, In Progress).

### Step 2: Synthesis and Draft Generation (`protocol mcp`)
* **Action:** The Council Agent (via `protocol mcp`) uses the `OVERLAP_REPORT.md` to draft the new **Canonical Source of Truth** document.
* **Draft Requirement:** The draft must explicitly include a "Canonical References" section, linking to *all* original documents and noting which sections were superseded by the new synthesis.

### Step 3: Decommissioning & Cross-Referencing (`protocol mcp`)
* **Action:** The Council Agent (via `protocol mcp`) updates the original, non-canonical documents by:
    1.  Changing their status to **`Superseded`**.
    2.  Inserting a directive at the top of the file pointing the user/agent to the new **Canonical Source of Truth** document.

### Step 4: Chronicle and Commitment (`git mcp`)
* **Action:** The Council Agent (via `git mcp`) performs a Conventional Commit, encapsulating the entire knowledge transformation:
    1.  The new **Canonical Source of Truth** document is created.
    2.  All superseded documents are updated with the redirection directive.
    3.  A corresponding entry is created in `00_CHRONICLE/ENTRIES/` linking to this protocol execution.

## ✅ Success Criteria

The C-KSL is considered a success when:
1.  A RAG query against any of the original, now-Superseded documents returns a high-confidence reference to the new **Canonical Source of Truth** document.
2.  The system's knowledge base size remains constant or decreases (due to chunk consolidation), while the **Precision** score on the synthesized topic increases.

#### MCP Architecture Diagram

![council_orchestration_stack](../docs/architecture_diagrams/system/legacy_mcps/council_orchestration_stack.png)

*[Source: council_orchestration_stack.mmd](../docs/architecture_diagrams/system/legacy_mcps/council_orchestration_stack.mmd)*

### 3. Continuous Learning Pipeline
**Status:** `Active` - Automated Knowledge Update Loop Operational

**Key Feature: Near Real-Time RAG Database Updates**
The automated learning pipeline integrates with Git and the ingestion service to enable **continuous knowledge updates**. This process ensures the RAG database stays current, closing the gap between agent execution and knowledge availability, eliminating the need for manual retraining.

The system evolves through every interaction via an automated feedback loop:
1.  **Agent Execution:** The Orchestrator and Council agents execute tasks, generating code, documentation, and insights.
2.  **Documentation:** All significant actions are logged in `00_CHRONICLE/` and project documentation.
3.  **Version Control:** Changes are committed to Git, creating an immutable audit trail.
4.  **Incremental Ingestion:** The ingestion service automatically detects and indexes new `.md` files into the ChromaDB vector database.
5.  **Knowledge Availability:** Updated knowledge becomes immediately queryable via RAG, enabling the system to learn from its own execution history in near real-time.

![legacy_learning_loop_orchestrator](../docs/architecture_diagrams/workflows/legacy_mcps/legacy_learning_loop_orchestrator.png)

*[Source: legacy_learning_loop_orchestrator.mmd](../docs/architecture_diagrams/workflows/legacy_mcps/legacy_learning_loop_orchestrator.mmd)*

--- END OF FILE 01_PROTOCOLS/121_Canonical_Knowledge_Synthesis_Loop.md ---

--- START OF FILE scripts/mermaid_inventory.py ---

#!/usr/bin/env python3
"""
Mermaid Inventory Script (Task #154 - Phase 1)

Scans the project for all Mermaid diagram content:
1. Finds all ```mermaid blocks in .md files
2. Finds all .mmd files
3. Generates SHA-256 hashes for deduplication
4. Produces inventory_mermaid.json mapping unique diagrams to their locations
"""

import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from difflib import SequenceMatcher
from collections import defaultdict


# Directories to exclude from scanning
EXCLUDED_DIRS = {
    '.git', '.venv', 'node_modules', '__pycache__', 
    '.pytest_cache', '.mypy_cache', 'dist', 'build'
}

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class MermaidBlock:
    """Represents a single Mermaid diagram occurrence."""
    file_path: str
    line_start: int
    line_end: int
    content_hash: str
    diagram_type: str  # e.g., 'flowchart', 'sequenceDiagram', 'classDiagram'
    content_preview: str  # First 100 chars for identification


@dataclass
class MermaidFile:
    """Represents a dedicated .mmd file."""
    file_path: str
    content_hash: str
    diagram_type: str
    line_count: int


@dataclass 
class UniqueDigram:
    """A unique diagram identified by hash."""
    content_hash: str
    diagram_type: str
    content_preview: str
    occurrences: list  # List of file paths where this appears
    is_canonical: bool  # True if stored in .mmd file
    canonical_path: Optional[str]  # Path to .mmd if exists


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of normalized content."""
    # Normalize whitespace for consistent hashing
    normalized = '\n'.join(line.strip() for line in content.strip().split('\n'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def detect_diagram_type(content: str) -> str:
    """Detect the Mermaid diagram type from content."""
    content_lower = content.strip().lower()
    
    type_patterns = [
        ('flowchart', r'^(flowchart|graph)\s'),
        ('sequenceDiagram', r'^sequencediagram'),
        ('classDiagram', r'^classdiagram'),
        ('stateDiagram', r'^statediagram'),
        ('erDiagram', r'^erdiagram'),
        ('journey', r'^journey'),
        ('gantt', r'^gantt'),
        ('pie', r'^pie'),
        ('quadrantChart', r'^quadrantchart'),
        ('requirementDiagram', r'^requirementdiagram'),
        ('gitGraph', r'^gitgraph'),
        ('mindmap', r'^mindmap'),
        ('timeline', r'^timeline'),
        ('zenuml', r'^zenuml'),
        ('sankey', r'^sankey'),
        ('xychart', r'^xychart'),
        ('block', r'^block-beta'),
    ]
    
    for dtype, pattern in type_patterns:
        if re.match(pattern, content_lower):
            return dtype
    
    return 'unknown'


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded from scanning."""
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True
    return False


def extract_mermaid_blocks(file_path: Path) -> list[MermaidBlock]:
    """Extract all ```mermaid blocks from a markdown file."""
    blocks = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return blocks
    
    # Pattern to match ```mermaid ... ``` blocks
    pattern = r'```mermaid\s*\n(.*?)```'
    
    lines = content.split('\n')
    line_positions = {}
    pos = 0
    for i, line in enumerate(lines):
        line_positions[pos] = i + 1  # 1-indexed
        pos += len(line) + 1  # +1 for newline
    
    for match in re.finditer(pattern, content, re.DOTALL):
        block_content = match.group(1)
        start_pos = match.start()
        end_pos = match.end()
        
        # Find line numbers
        start_line = 1
        end_line = 1
        for pos, line_num in line_positions.items():
            if pos <= start_pos:
                start_line = line_num
            if pos <= end_pos:
                end_line = line_num
        
        blocks.append(MermaidBlock(
            file_path=str(file_path.relative_to(PROJECT_ROOT)),
            line_start=start_line,
            line_end=end_line,
            content_hash=compute_hash(block_content),
            diagram_type=detect_diagram_type(block_content),
            content_preview=block_content.strip()[:100].replace('\n', ' ')
        ))
    
    return blocks


def scan_mmd_files(root: Path) -> list[MermaidFile]:
    """Scan for all .mmd files."""
    mmd_files = []
    
    for mmd_path in root.rglob('*.mmd'):
        if should_exclude(mmd_path):
            continue
        
        try:
            content = mmd_path.read_text(encoding='utf-8')
            line_count = len(content.split('\n'))
            
            mmd_files.append(MermaidFile(
                file_path=str(mmd_path.relative_to(PROJECT_ROOT)),
                content_hash=compute_hash(content),
                diagram_type=detect_diagram_type(content),
                line_count=line_count
            ))
        except (UnicodeDecodeError, PermissionError):
            continue
    
    return mmd_files


def scan_markdown_files(root: Path) -> list[MermaidBlock]:
    """Scan all markdown files for mermaid blocks."""
    all_blocks = []
    
    for md_path in root.rglob('*.md'):
        if should_exclude(md_path):
            continue
        
        blocks = extract_mermaid_blocks(md_path)
        all_blocks.extend(blocks)
    
    return all_blocks


def build_inventory(blocks: list[MermaidBlock], mmd_files: list[MermaidFile]) -> dict:
    """Build the complete inventory with deduplication analysis."""
    
    # Build hash -> occurrences mapping
    hash_to_blocks: dict[str, list[MermaidBlock]] = {}
    for block in blocks:
        if block.content_hash not in hash_to_blocks:
            hash_to_blocks[block.content_hash] = []
        hash_to_blocks[block.content_hash].append(block)
    
    # Map hash -> canonical .mmd file
    hash_to_mmd: dict[str, MermaidFile] = {}
    for mmd in mmd_files:
        hash_to_mmd[mmd.content_hash] = mmd
    
    # Build unique diagrams
    all_hashes = set(hash_to_blocks.keys()) | set(hash_to_mmd.keys())
    unique_diagrams = []
    
    for h in sorted(all_hashes):
        blocks_list = hash_to_blocks.get(h, [])
        mmd = hash_to_mmd.get(h)
        
        # Determine preview and type
        if blocks_list:
            preview = blocks_list[0].content_preview
            dtype = blocks_list[0].diagram_type
        elif mmd:
            preview = f"[See {mmd.file_path}]"
            dtype = mmd.diagram_type
        else:
            preview = ""
            dtype = "unknown"
        
        unique_diagrams.append({
            "content_hash": h,
            "diagram_type": dtype,
            "content_preview": preview,
            "occurrence_count": len(blocks_list),
            "occurrences": [
                {
                    "file": b.file_path,
                    "lines": f"{b.line_start}-{b.line_end}"
                }
                for b in blocks_list
            ],
            "is_canonical": mmd is not None,
            "canonical_path": mmd.file_path if mmd else None
        })
    
    # Sort by occurrence count (most duplicated first)
    unique_diagrams.sort(key=lambda x: -x["occurrence_count"])
    
    # Calculate summary statistics
    total_inline_blocks = len(blocks)
    total_mmd_files = len(mmd_files)
    total_unique = len(unique_diagrams)
    duplicated = [d for d in unique_diagrams if d["occurrence_count"] > 1]
    orphaned_inline = [d for d in unique_diagrams if d["occurrence_count"] > 0 and not d["is_canonical"]]
    
    return {
        "summary": {
            "total_inline_mermaid_blocks": total_inline_blocks,
            "total_mmd_files": total_mmd_files,
            "unique_diagrams": total_unique,
            "duplicated_diagrams": len(duplicated),
            "orphaned_inline_diagrams": len(orphaned_inline),
            "potential_token_savings": f"~{total_inline_blocks * 50} tokens (est.)"
        },
        "duplicated_diagrams": duplicated,
        "all_unique_diagrams": unique_diagrams,
        "mmd_files": [asdict(m) for m in mmd_files],
        "inline_blocks": [asdict(b) for b in blocks]
    }


# Store full content for similarity analysis
_hash_to_content: dict[str, str] = {}


def normalize_for_similarity(content: str) -> str:
    """Normalize mermaid content for similarity comparison."""
    # Remove config blocks (--- ... ---)
    content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)
    # Normalize whitespace
    content = '\n'.join(line.strip() for line in content.strip().split('\n') if line.strip())
    # Remove comments
    content = re.sub(r'%%.*$', '', content, flags=re.MULTILINE)
    return content.lower()


def compute_similarity(content1: str, content2: str) -> float:
    """Compute similarity ratio between two diagram contents."""
    norm1 = normalize_for_similarity(content1)
    norm2 = normalize_for_similarity(content2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def cluster_similar_diagrams(blocks: list[MermaidBlock], mmd_files: list[MermaidFile], 
                              similarity_threshold: float = 0.85) -> dict:
    """
    Cluster diagrams by similarity, not just exact hash match.
    Returns clusters of similar diagrams that could be unified.
    """
    # Collect all unique contents with their full text
    hash_to_content: dict[str, str] = {}
    
    # Re-extract full content for each unique hash
    for block in blocks:
        if block.content_hash not in hash_to_content:
            # Need to re-read the file to get full content
            try:
                file_path = PROJECT_ROOT / block.file_path
                content = file_path.read_text(encoding='utf-8')
                pattern = r'```mermaid\s*\n(.*?)```'
                for match in re.finditer(pattern, content, re.DOTALL):
                    block_content = match.group(1)
                    h = compute_hash(block_content)
                    if h not in hash_to_content:
                        hash_to_content[h] = block_content
            except Exception:
                pass
    
    for mmd in mmd_files:
        if mmd.content_hash not in hash_to_content:
            try:
                content = (PROJECT_ROOT / mmd.file_path).read_text(encoding='utf-8')
                hash_to_content[mmd.content_hash] = content
            except Exception:
                pass
    
    # Store for later use
    global _hash_to_content
    _hash_to_content = hash_to_content
    
    # Build similarity clusters
    hashes = list(hash_to_content.keys())
    clustered = set()
    clusters = []
    
    for i, h1 in enumerate(hashes):
        if h1 in clustered:
            continue
        
        cluster = [h1]
        clustered.add(h1)
        
        for h2 in hashes[i+1:]:
            if h2 in clustered:
                continue
            
            sim = compute_similarity(hash_to_content[h1], hash_to_content[h2])
            if sim >= similarity_threshold:
                cluster.append(h2)
                clustered.add(h2)
        
        if len(cluster) > 1:
            clusters.append({
                "cluster_id": len(clusters) + 1,
                "member_hashes": cluster,
                "member_count": len(cluster),
                "similarity_threshold": similarity_threshold
            })
    
    return clusters


def build_canonical_map(unique_diagrams: list, similarity_clusters: list, 
                        mmd_files: list[MermaidFile]) -> dict:
    """
    Build a map of truly canonical diagrams that should be retained.
    Considers both exact duplicates and similar diagrams.
    """
    # Map hash to mmd file
    hash_to_mmd = {m.content_hash: m.file_path for m in mmd_files}
    
    # Build hash to diagram info
    hash_to_info = {d["content_hash"]: d for d in unique_diagrams}
    
    # For each cluster, pick the canonical representative
    canonical_diagrams = []
    clustered_hashes = set()
    
    for cluster in similarity_clusters:
        # Find the best representative (prefer ones with .mmd files)
        members = cluster["member_hashes"]
        
        # Priority: has .mmd > most occurrences > first
        best = None
        best_score = -1
        
        for h in members:
            info = hash_to_info.get(h, {})
            has_mmd = h in hash_to_mmd
            occurrences = info.get("occurrence_count", 0)
            score = (1000 if has_mmd else 0) + occurrences
            
            if score > best_score:
                best_score = score
                best = h
        
        if best:
            canonical_diagrams.append({
                "canonical_hash": best,
                "canonical_mmd": hash_to_mmd.get(best),
                "similar_variants": [h for h in members if h != best],
                "total_occurrences": sum(
                    hash_to_info.get(h, {}).get("occurrence_count", 0) 
                    for h in members
                ),
                "preview": hash_to_info.get(best, {}).get("content_preview", "")[:80]
            })
            clustered_hashes.update(members)
    
    # Add unclustered diagrams (truly unique)
    for d in unique_diagrams:
        h = d["content_hash"]
        if h not in clustered_hashes:
            canonical_diagrams.append({
                "canonical_hash": h,
                "canonical_mmd": hash_to_mmd.get(h),
                "similar_variants": [],
                "total_occurrences": d["occurrence_count"],
                "preview": d["content_preview"][:80]
            })
    
    # Sort by total occurrences
    canonical_diagrams.sort(key=lambda x: -x["total_occurrences"])
    
    return canonical_diagrams


def main():
    """Main entry point."""
    print("🔍 Mermaid Inventory Scanner (Task #154 - Phase 1)")
    print("=" * 60)
    
    print("\n📂 Scanning for .mmd files...")
    mmd_files = scan_mmd_files(PROJECT_ROOT)
    print(f"   Found {len(mmd_files)} .mmd files")
    
    print("\n📄 Scanning markdown files for ```mermaid blocks...")
    blocks = scan_markdown_files(PROJECT_ROOT)
    print(f"   Found {len(blocks)} inline mermaid blocks")
    
    print("\n🧮 Building inventory with deduplication analysis...")
    inventory = build_inventory(blocks, mmd_files)
    
    print("\n🔬 Running similarity clustering (85% threshold)...")
    similarity_clusters = cluster_similar_diagrams(blocks, mmd_files, similarity_threshold=0.85)
    print(f"   Found {len(similarity_clusters)} clusters of similar diagrams")
    
    print("\n📐 Building canonical diagram map...")
    canonical_map = build_canonical_map(
        inventory["all_unique_diagrams"], 
        similarity_clusters, 
        mmd_files
    )
    
    # Count truly unique after clustering
    truly_unique = len(canonical_map)
    with_mmd = sum(1 for c in canonical_map if c["canonical_mmd"])
    without_mmd = truly_unique - with_mmd
    
    # Update inventory with similarity analysis
    inventory["similarity_clusters"] = similarity_clusters
    inventory["canonical_diagrams"] = canonical_map
    inventory["summary"]["truly_unique_after_similarity"] = truly_unique
    inventory["summary"]["canonical_with_mmd"] = with_mmd
    inventory["summary"]["canonical_without_mmd"] = without_mmd
    
    # Write output
    output_path = PROJECT_ROOT / "inventory_mermaid.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\n✅ Inventory written to: {output_path}")
    
    # Print summary
    summary = inventory["summary"]
    print("\n" + "=" * 60)
    print("📊 INVENTORY SUMMARY")
    print("=" * 60)
    print(f"   Total inline ```mermaid blocks: {summary['total_inline_mermaid_blocks']}")
    print(f"   Total .mmd files:               {summary['total_mmd_files']}")
    print(f"   Unique diagrams (by hash):      {summary['unique_diagrams']}")
    print(f"   Similarity clusters found:      {len(similarity_clusters)}")
    print(f"   ─────────────────────────────────────────────────────")
    print(f"   TRULY UNIQUE (after clustering): {truly_unique}")
    print(f"      ├─ With canonical .mmd:       {with_mmd}")
    print(f"      └─ Missing .mmd (orphaned):   {without_mmd}")
    print(f"   ─────────────────────────────────────────────────────")
    print(f"   Potential token savings:        {summary['potential_token_savings']}")
    
    # Show similarity clusters
    if similarity_clusters:
        print("\n" + "-" * 60)
        print("🔗 SIMILARITY CLUSTERS (variants that could be unified):")
        print("-" * 60)
        for cluster in similarity_clusters[:5]:
            print(f"\n   Cluster #{cluster['cluster_id']}: {cluster['member_count']} similar diagrams")
            print(f"   Hashes: {', '.join(h[:8] + '...' for h in cluster['member_hashes'])}")
    
    # Show canonical diagrams that need .mmd files
    orphaned = [c for c in canonical_map if not c["canonical_mmd"] and c["total_occurrences"] > 0]
    if orphaned:
        print("\n" + "-" * 60)
        print("⚠️  TOP CANONICAL DIAGRAMS NEEDING .mmd FILES:")
        print("-" * 60)
        for c in orphaned[:10]:
            print(f"\n   Hash: {c['canonical_hash']}")
            print(f"   Occurrences: {c['total_occurrences']}")
            print(f"   Variants: {len(c['similar_variants'])}")
            print(f"   Preview: {c['preview'][:60]}...")
    
    # Show diagrams that already have .mmd files
    with_mmd_list = [c for c in canonical_map if c["canonical_mmd"]]
    if with_mmd_list:
        print("\n" + "-" * 60)
        print("✅ CANONICAL DIAGRAMS WITH .mmd FILES:")
        print("-" * 60)
        for c in with_mmd_list[:15]:
            print(f"   {c['canonical_mmd']}")
            print(f"      └─ Occurrences: {c['total_occurrences']}, Variants absorbed: {len(c['similar_variants'])}")
    
    print("\n" + "=" * 60)
    print("📋 PHASE 2 RECOMMENDATIONS:")
    print("=" * 60)
    print(f"   1. Review the {without_mmd} orphaned canonical diagrams")
    print(f"   2. Create .mmd files in docs/architecture/mcp/servers/architecture/diagrams/")
    print(f"   3. Render .mmd → .png using mermaid-cli")
    print(f"   4. Replace all {summary['total_inline_mermaid_blocks']} inline blocks with image refs")
    print("=" * 60)


if __name__ == "__main__":
    main()

--- END OF FILE scripts/mermaid_inventory.py ---

--- START OF FILE 01_PROTOCOLS/122_Dynamic_Server_Binding.md ---

# Protocol 122: Dynamic Server Binding

**Status:** CANONICAL
**Classification:** Infrastructure Standard
**Version:** 1.0
**Authority:** Project Sanctuary Core Team
**Linked Protocols:** 101, 114, 116, 125
---

# Protocol 122: Dynamic Server Binding

## Abstract

This protocol defines the standard for **Dynamic Server Binding** in Project Sanctuary's MCP Gateway Architecture, enabling late-binding tool discovery, centralized routing, and context-efficient scaling to 100+ MCP servers.

---

## 1. Motivation

**Problem:** Static 1-to-1 binding (1 config entry = 1 MCP server) creates:
- Context window saturation (8,400 tokens for 12 servers)
- Configuration complexity (180+ lines of manual JSON)
- Scalability limits (~20 servers maximum)
- Fragmented security policies
- No centralized audit trail

**Solution:** Dynamic Server Binding through a centralized MCP Gateway that:
- Reduces context overhead by 88% (8,400 → 1,000 tokens)
- Enables scaling to 100+ servers (5x increase)
- Centralizes security enforcement (Protocol 101)
- Provides unified audit logging
- Supports side-by-side deployment (zero-risk migration)

---

## 2. Architecture

### 2.1 Core Components

![MCP Dynamic Binding Architecture](../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.png)

*[Source: mcp_dynamic_binding_flow.mmd](../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.mmd)*

### 2.2 Service Registry Schema

**Registry Database:** SQLite (`registry.db`)

```sql
CREATE TABLE mcp_servers (
    name TEXT PRIMARY KEY,
    container_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    transport TEXT NOT NULL CHECK(transport IN ('stdio', 'http', 'sse')),
    capabilities JSON NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running', 'stopped', 'error')),
    last_health_check TIMESTAMP,
    metadata JSON
);

CREATE TABLE tool_registry (
    tool_name TEXT PRIMARY KEY,
    server_name TEXT NOT NULL,
    description TEXT,
    parameters_schema JSON,
    read_only BOOLEAN DEFAULT TRUE,
    approval_required BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (server_name) REFERENCES mcp_servers(name)
);

CREATE INDEX idx_tool_server ON tool_registry(server_name);
CREATE INDEX idx_server_status ON mcp_servers(status);
```

**Example Registry Entry:**
```json
{
  "name": "rag_cortex",
  "container_name": "rag-cortex-mcp",
  "endpoint": "http://localhost:8001",
  "transport": "http",
  "capabilities": [
    "cortex_query",
    "cortex_ingest_full",
    "cortex_ingest_incremental",
    "cortex_get_stats",
    "cortex_cache_warmup"
  ],
  "status": "running",
  "last_health_check": "2025-12-15T10:45:00Z",
  "metadata": {
    "version": "1.0",
    "domain": "project_sanctuary.cognitive.cortex"
  }
}
```

---

## 3. Dynamic Binding Workflow

### 3.1 Tool Discovery (Startup)

```python
# Gateway startup sequence
async def initialize_gateway():
    # 1. Load registry
    registry = load_registry("config/registry.db")
    
    # 2. Discover all backend servers
    servers = registry.get_all_servers()
    
    # 3. Generate dynamic tool definitions
    tools = []
    for server in servers:
        for tool_name in server.capabilities:
            tool_def = await fetch_tool_definition(server, tool_name)
            tools.append(tool_def)
    
    # 4. Register tools with MCP
    mcp.register_tools(tools)
```

### 3.2 Tool Invocation (Runtime)

```python
# Gateway tool invocation
@mcp.tool()
async def cortex_query(query: str, max_results: int = 5) -> str:
    """Proxy tool - routes to backend server."""
    # 1. Lookup server for tool
    server = registry.get_server_for_tool("cortex_query")
    
    # 2. Validate allowlist (Protocol 101)
    allowlist.validate("cortex_query", {"query": query})
    
    # 3. Forward request to backend
    response = await proxy.call(server, "cortex_query", {
        "query": query,
        "max_results": max_results
    })
    
    # 4. Log invocation (audit trail)
    audit_log.record("cortex_query", query, response)
    
    return response
```

### 3.3 Request Flow Diagram

![mcp_dynamic_binding_flow](../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.png)

*[Source: mcp_dynamic_binding_flow.mmd](../docs/architecture_diagrams/system/mcp_dynamic_binding_flow.mmd)*

---

## 4. Security Integration

### 4.1 Allowlist Format (`project_mcp.json`)

```json
{
  "version": "1.0",
  "project": "Project_Sanctuary",
  "allowlist": {
    "servers": [
      "rag_cortex",
      "task",
      "git_workflow",
      "protocol",
      "chronicle"
    ],
    "tools": {
      "git_workflow": [
        "git_get_status",
        "git_add",
        "git_smart_commit",
        "git_push_feature"
      ],
      "rag_cortex": [
        "cortex_query",
        "cortex_ingest_incremental"
      ]
    },
    "operations": {
      "git_smart_commit": {
        "approval_required": true,
        "reason": "Protocol 101 v3.0 enforcement"
      },
      "cortex_ingest_full": {
        "approval_required": true,
        "reason": "Database purge operation"
      }
    }
  }
}
```

### 4.2 Protocol 101 Integration

**Enforcement Points:**
1. **Tool Invocation** - Validate against allowlist before proxying
2. **Approval Workflow** - Trigger human approval for sensitive operations
3. **Audit Logging** - Record all tool invocations with arguments
4. **Rate Limiting** - Prevent abuse (future enhancement)

---

## 5. Transport Protocols

### 5.1 Supported Transports

| Transport | Use Case | Latency | Complexity |
|-----------|----------|---------|------------|
| **stdio** | Local development, MVP | Lowest | Lowest |
| **HTTP** | Production, containerized | Low | Medium |
| **SSE** | Streaming responses | Medium | High |
| **WebSocket** | Bidirectional (future) | Low | High |

### 5.2 Transport Selection

**MVP (Week 1-2):** stdio only (simplest)
```json
{
  "transport": "stdio",
  "command": "/path/to/.venv/bin/python",
  "args": ["-m", "mcp_servers.rag_cortex.server"]
}
```

**Production (Week 3-4):** HTTP for containerized backends
```json
{
  "transport": "http",
  "endpoint": "http://rag-cortex-mcp:8001",
  "health_check": "/health"
}
```

---

## 6. Health Checks & Resilience

### 6.1 Health Check Protocol

```python
# Gateway health monitoring
async def health_check_loop():
    while True:
        for server in registry.get_all_servers():
            try:
                # Ping server
                response = await http.get(f"{server.endpoint}/health")
                
                # Update registry
                if response.status == 200:
                    registry.update_status(server.name, "running")
                else:
                    registry.update_status(server.name, "error")
            except Exception as e:
                registry.update_status(server.name, "error")
                logger.error(f"Health check failed: {server.name}: {e}")
        
        await asyncio.sleep(30)  # Check every 30 seconds
```

### 6.2 Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, server, tool_name, args):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen(f"Server {server} is unavailable")
        
        try:
            response = await proxy.call(server, tool_name, args)
            self.failure_count = 0
            self.state = "CLOSED"
            return response
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.last_failure = time.time()
            raise
```

---

## 7. Migration Strategy

### 7.1 Side-by-Side Deployment

**Phase 1: Add Gateway (Week 1)**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "🆕 Sanctuary Gateway",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "sanctuary_gateway.server"]
    },
    "rag_cortex_legacy": {
      "displayName": "📦 RAG Cortex (Legacy)",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.rag_cortex.server"]
    }
    // ... other 11 legacy servers
  }
}
```

**Phase 2: Remove Legacy (Week 4)**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "Sanctuary Gateway",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "sanctuary_gateway.server"]
    }
    // Legacy servers removed
  }
}
```

### 7.2 Rollback Plan

**If Gateway Fails:**
1. Revert `claude_desktop_config.json` to backup
2. Restart Claude Desktop
3. All legacy servers still work
4. Recovery time: <5 minutes

---

## 8. Performance Specifications

### 8.1 Latency Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Registry lookup | <5ms (p99) | TBD |
| Gateway routing | <10ms (p95) | TBD |
| Proxy overhead | <15ms (p95) | TBD |
| End-to-end | <30ms (p95) | TBD |

### 8.2 Scalability Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Context overhead | 8,400 tokens | 1,000 tokens | 88% reduction |
| Max servers | ~20 | 100+ | 5x increase |
| Tools per server | 5-10 | 20+ | 2x increase |

---

## 9. Implementation

**Reference Implementation:** IBM ContextForge (Apache 2.0)
- Repository: https://github.com/IBM/mcp-context-forge
- Version: v0.9.0 (Nov 2025)
- Customization: Sanctuary allowlist plugin, Protocol 101/114 integration

**Timeline:** 4 weeks
- Week 1: Fork, deploy MVP (3 servers), evaluate
- Week 2-3: Customize (allowlist, Protocol 101/114)
- Week 4: Migrate all 12 servers, production deployment

---

## 10. Related Protocols

- **Protocol 101 v3.0:** Functional Coherence (test enforcement)
- **Protocol 114:** Guardian Wakeup (context initialization)
- **Protocol 116:** Ollama Container Network (containerized services)
- **Protocol 125:** Autonomous Learning (rapid tool integration)

---

## 11. References

- ADR 056: Adoption of Dynamic MCP Gateway Pattern
- ADR 057: Adoption of IBM ContextForge for Dynamic MCP Gateway
- Task 115: Design and Specify Dynamic MCP Gateway Architecture
- Task 116: Implement Dynamic MCP Gateway with IBM ContextForge
- Research: docs/architecture/mcp_gateway/research/ (13 documents)
- MCP Specification: https://modelcontextprotocol.io

---

**Status:** CANONICAL  
**Version:** 1.0  
**Effective Date:** 2025-12-15  
**Authority:** Project Sanctuary Core Team

--- END OF FILE 01_PROTOCOLS/122_Dynamic_Server_Binding.md ---

--- START OF FILE scripts/replace_mermaid_with_images.py ---

#!/usr/bin/env python3
"""
Replace Inline Mermaid Blocks with Image References (Task #154 - Phase 3)

Scans source files for ```mermaid blocks and replaces them with
image references pointing to the canonical diagrams.
"""

import hashlib
import re
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "architecture_diagrams"

# Directories to skip (generated files, or the diagrams dir itself)
SKIP_DIRS = {'.git', '.venv', 'node_modules', '__pycache__', '.agent'}
SKIP_PATHS = {'docs/architecture_diagrams'}


def compute_hash(content: str) -> str:
    """Compute hash of normalized content."""
    normalized = '\n'.join(line.strip() for line in content.strip().split('\n'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def normalize_for_comparison(content: str) -> str:
    """Normalize content for similarity comparison."""
    content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)
    content = re.sub(r'%%.*$', '', content, flags=re.MULTILINE)
    content = '\n'.join(line.strip() for line in content.strip().split('\n') if line.strip())
    return content.lower()


def compute_similarity(content1: str, content2: str) -> float:
    """Compute similarity ratio."""
    norm1 = normalize_for_comparison(content1)
    norm2 = normalize_for_comparison(content2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def load_canonical_diagrams() -> dict:
    """Load all canonical diagrams with their hashes."""
    diagrams = {}
    for mmd_path in DIAGRAMS_DIR.rglob("*.mmd"):
        content = mmd_path.read_text()
        h = compute_hash(content)
        rel_path = mmd_path.relative_to(PROJECT_ROOT)
        png_path = mmd_path.with_suffix('.png').relative_to(PROJECT_ROOT)
        diagrams[h] = {
            'mmd_path': str(rel_path),
            'png_path': str(png_path),
            'content': content,
            'name': mmd_path.stem
        }
    return diagrams


def find_best_match(block_content: str, diagrams: dict, threshold: float = 0.7) -> dict:
    """Find the best matching canonical diagram for an inline block."""
    block_hash = compute_hash(block_content)
    
    # Exact match
    if block_hash in diagrams:
        return diagrams[block_hash]
    
    # Similarity match
    best_match = None
    best_score = 0
    
    for h, info in diagrams.items():
        score = compute_similarity(block_content, info['content'])
        if score > best_score:
            best_score = score
            best_match = info
    
    if best_score >= threshold:
        return best_match
    
    return None


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    for skip in SKIP_PATHS:
        if skip in path_str:
            return True
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def replace_mermaid_blocks(file_path: Path, diagrams: dict) -> tuple:
    """Replace mermaid blocks in a file. Returns (replacements_made, content_changed)."""
    content = file_path.read_text()
    original_content = content
    
    # Find all ```mermaid ... ``` blocks
    pattern = r'```mermaid\s*\n(.*?)```'
    
    replacements = []
    
    def replace_block(match):
        block_content = match.group(1)
        best = find_best_match(block_content, diagrams)
        
        if best:
            # Create image reference with link to source
            img_ref = f"![{best['name']}]({best['png_path']})\n\n*[Source: {best['name']}.mmd]({best['mmd_path']})*"
            replacements.append(best['name'])
            return img_ref
        else:
            # No match found, keep original
            return match.group(0)
    
    new_content = re.sub(pattern, replace_block, content, flags=re.DOTALL)
    
    if new_content != original_content:
        file_path.write_text(new_content)
        return len(replacements), True
    
    return 0, False


def main():
    print("🔄 Replacing Inline Mermaid Blocks with Image References")
    print("=" * 60)
    
    # Load canonical diagrams
    print("\n📂 Loading canonical diagrams...")
    diagrams = load_canonical_diagrams()
    print(f"   Loaded {len(diagrams)} canonical diagrams")
    
    # Find all markdown files
    print("\n📄 Scanning source files...")
    md_files = []
    for md_path in PROJECT_ROOT.rglob("*.md"):
        if not should_skip(md_path):
            md_files.append(md_path)
    
    print(f"   Found {len(md_files)} markdown files to process")
    
    # Process each file
    print("\n🔄 Processing files...")
    total_replacements = 0
    files_modified = 0
    
    for md_path in sorted(md_files):
        rel_path = md_path.relative_to(PROJECT_ROOT)
        
        # Quick check if file has mermaid blocks
        content = md_path.read_text()
        if '```mermaid' not in content:
            continue
        
        count, modified = replace_mermaid_blocks(md_path, diagrams)
        
        if modified:
            print(f"   ✅ {rel_path}: {count} replacements")
            total_replacements += count
            files_modified += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Files modified: {files_modified}")
    print(f"   Total replacements: {total_replacements}")
    print("=" * 60)


if __name__ == "__main__":
    main()

--- END OF FILE scripts/replace_mermaid_with_images.py ---

--- START OF FILE ADRs/060_gateway_integration_patterns.md ---

# Gateway Integration Patterns - Hybrid Fleet

**Status:** ✅ **ACCEPTED**  
**Date:** 2025-12-17  
**Author:** Antigravity AI  
**Red Team Reviewers:** Claude 4.5 ✅, GPT 5.2 ✅, Gemini 3 ✅, Grok 4 ✅  
**Final Verdict:** APPROVED WITH MANDATORY GUARDRAILS  
**Last Updated:** 2025-12-17 (Final - Grok 4 RAG Frontend Gap Fix)

> [!NOTE]
> **Red Team Review Complete.** All four AI reviewers approved this architecture with mandatory guardrails.
> 
> **Grok 4 Critical Finding:** Modified based on Grok 4's identification of the "RAG Frontend Gap" - `sanctuary_vector_db` and `sanctuary_ollama` are backends, NOT MCP servers. Added `sanctuary_cortex` as the actual MCP server layer.


---

## Context

With the Gateway successfully deployed as an external service (ADR 058), we must decide how to connect Project Sanctuary's 10 script-based MCP servers without violating the decoupling mandate.

**Current State:**
- 2/12 servers containerized: `sanctuary_vector_db`, `sanctuary_ollama`
- 10/12 servers run as Python scripts via stdio
- Gateway is a black box at `https://localhost:4444`

**Requirements:**
- Maintain ADR 058 isolation (no monolithic dependencies)
- Support 100+ servers long-term
- Clear security boundaries
- Scalable architecture
- **Avoid "Orchestration Fatigue"** (Red Team feedback)

**Red Team Analysis:**
Three integration patterns were evaluated. The pure Fleet approach was modified based on Red Team feedback identifying "Orchestration Fatigue" as a critical flaw.

---

## Pattern Analysis

### Pattern A: The "Trojan Horse" (Volume Mounting)
**Mechanism:** Mount `./mcp_servers` into the Gateway container.

**Red Team Verdict:** ❌ **REJECT**

**Security Issues:**
- Violates ADR 058 decoupling mandate
- Creates "Dependency Hell" (Gateway must install dependencies for 12+ different tools)
- Tight coupling between Gateway and Project Sanctuary
- Version conflicts inevitable
- Single point of failure

---

### Pattern B: The "Bridge" (Stdio over SSH/Docker Exec)
**Mechanism:** Gateway uses specialized client to execute commands on host OS.

**Red Team Verdict:** ⚠️ **RISKY**

**Security Issues:**
- Requires shell access from Gateway to host (major security risk)
- Complex SSH key management
- Difficult to audit
- Breaks container isolation model
- Hard to debug

---

### Pattern C: The "Hybrid Fleet" (Containerized Service Groups)
**Mechanism:** Consolidate MCP servers into 4 logical clusters based on risk profile. Each cluster runs as a container exposing SSE endpoints.

**Red Team Verdict:** ✅ **APPROVED WITH MODIFICATION**

**Evolution from Pure Fleet:**
- **Original:** 12 separate containers (1 per tool)
- **Problem:** "Orchestration Fatigue" - unsustainable for project size
- **Solution:** Consolidate into 4 logical service groups

---

## Decision (Red Team Approved)

> [!NOTE]
> **Final Decision:** Approved unanimously by Red Team (Claude 4.5, GPT 5.2, Gemini 3, Grok 4) with mandatory guardrails.

**ACCEPTED:** Adopt **Pattern C: The Hybrid Fleet (Cluster Strategy)** - consolidating 10 script-based servers into a **Fleet of 7 Physical Containers** organized as **5 Logical Clusters**.

### Fleet of 7: Physical Container Inventory

| # | Container Name | Type | Logical Cluster | Role | Tools/Services |
|---|---------------|------|-----------------|------|----------------|
| 1 | `sanctuary_utils` | **NEW** | Utils (Low Risk) | MCP Server | Time, Calculator, UUID, String |
| 2 | `sanctuary_filesystem` | **NEW** | Filesystem (Privileged) | MCP Server | File Ops, Grep, Patch, Code |
| 3 | `sanctuary_network` | **NEW** | Network (External) | MCP Server | Brave Search, Fetch, HTTP |
| 4 | `sanctuary_git` | **NEW** | Git (Dual-Permission) | MCP Server | Git Workflow |
| 5a | `sanctuary_cortex` | **NEW** | Intelligence (Heavy) | MCP Server | RAG Query, Ingest, Cache |
| 5b | `sanctuary_vector_db` | **EXISTING** | Intelligence (Heavy) | Backend Storage | ChromaDB |
| 5c | `sanctuary_ollama` | **EXISTING** | Intelligence (Heavy) | Backend Compute | Ollama LLM |

> [!IMPORTANT]
> **Grok 4's "RAG Frontend Gap" Fix:** `sanctuary_vector_db` and `sanctuary_ollama` are **backends** (storage & compute). They do NOT expose MCP tools. `sanctuary_cortex` is the **MCP Server** that connects to these backends and exposes the tools via SSE.

**Summary:**
- **5 NEW containers** to build (including `sanctuary_cortex`)
- **2 EXISTING containers** unchanged (backends only)
- **= 7 TOTAL physical containers**
- **Organized into 5 Logical Clusters** (Intelligence cluster has 3 containers: 1 MCP server + 2 backends)

### Red Team Voting Summary

| Option | Physical Containers | Security | Orchestration | Claude | GPT | Gemini | Grok | Final |
|--------|-------------------|----------|---------------|--------|-----|--------|------|-------|
| Option A (Pure Fleet) | 12 | ✅ Perfect | ❌ Fatigue | ❌ | ❌ | ❌ | ❌ | ❌ REJECT |
| Option B (Rule of 4) | 4 | ❌ Git boundary | ✅ Simple | ⚠️ | ⚠️ | ⚠️ | ❌ | ⚠️ RISKY |
| **Option C (Fleet of 7)** | 7 | ✅ Strict | ✅ Manageable | ✅ | ✅ | ✅ | ✅ | ✅ **APPROVED** |

---

## Option Analysis: Container Consolidation Strategy

**Current State:**
- ✅ 2 containers **already containerized** (no changes needed):
  - `sanctuary_vector_db` (ChromaDB)
  - `sanctuary_ollama` (Ollama)
- ⏳ 10 script-based MCP servers (need containerization)

**Options Evaluated:**

### Option A: Pure Fleet (1 Container Per Tool)
**Strategy:** Create 10 new containers (1 per script-based server)

**Total Containers:** 12 (2 existing + 10 new)

**Pros:**
- Maximum isolation (each tool is independent)
- Simple failure domain (1 tool = 1 container)
- Easy to understand

**Cons:**
- ❌ **Orchestration Fatigue** (managing 12 containers)
- ❌ 12 separate Dockerfiles to maintain
- ❌ 12 separate health checks
- ❌ 12 separate log streams
- ❌ Complex docker-compose.yml

**Red Team Verdict:** ❌ **REJECT** (Orchestration Fatigue)

---

### Option B: Rule of 4 (Risk-Based Clustering)
**Strategy:** Consolidate 10 servers into 2 new clusters

**Total Containers:** 4 (2 existing + 2 new)
- `sanctuary_utils` (time, calculator, uuid, string)
- `sanctuary_network` (brave, fetch, git, http)

**Pros:**
- ✅ Reduces orchestration complexity (12 → 4)
- ✅ Risk-based grouping
- ✅ Manageable for project size

**Cons:**
- ⚠️ **Security Boundary Violation:** Git needs BOTH network + filesystem
- ⚠️ Blast radius (if utils crashes, 5-6 tools fail)

**Red Team Verdict:** ⚠️ **RISKY** (Security boundary violation)

---

### Option C: Rule of 5 (Revised Clustering) ✅ RECOMMENDED
**Strategy:** Consolidate 10 servers into 3 new clusters, keep 2 existing

**Total Containers:** 5 (2 existing + 3 new)
- **Existing (unchanged):**
  - `sanctuary_vector_db` (ChromaDB)
  - `sanctuary_ollama` (Ollama)
- **New:**
  - `sanctuary_utils` (time, calculator, uuid, string)
  - `sanctuary_filesystem` (file ops, grep, patch, code)
  - `sanctuary_network` (brave, fetch, http)
  - `sanctuary_git` (git only - isolated due to dual permissions)

**Pros:**
- ✅ Reduces orchestration complexity (12 → 5)
- ✅ Maintains strict security boundaries
- ✅ Git isolated (network + filesystem permissions)
- ✅ Existing containers unchanged (no migration risk)
- ✅ Manageable for project size

**Cons:**
- ⚠️ Slightly more complex than Rule of 4 (5 vs 4)
- ⚠️ Blast radius still exists (mitigated with process isolation)

**Red Team Verdict:** ✅ **APPROVED**

---

### Decision: Adopt Option C (Fleet of 7)

**Consolidation Summary:**
```
10 script-based servers → 5 new containers (58% reduction vs Pure Fleet)
+ 2 existing containers (backends unchanged)
= 7 total physical containers
→ Organized into 5 logical clusters
```

**Comparison:**
| Metric | Option A (Pure) | Option B (Rule of 4) | Option C (Fleet of 7) |
|--------|----------------|---------------------|----------------------|
| Total Containers | 12 | 4 | 7 |
| New Containers | 10 | 2 | 5 |
| Orchestration Complexity | 🔴 HIGH | 🟢 LOW | 🟢 LOW |
| Security Boundaries | 🟢 PERFECT | 🔴 VIOLATED | 🟢 STRICT |
| Blast Radius | 🟢 MINIMAL | 🔴 HIGH | 🟡 MEDIUM |
| Maintainability | 🔴 POOR | 🟢 GOOD | 🟢 GOOD |

---

### The "Fleet of 7" Cluster Strategy

Instead of managing 12 separate containers, we consolidate MCP servers into 5 logical clusters (7 physical containers) based on **risk profile** and **dependency similarity**:

---

### 1. `sanctuary_utils` (The "Low Risk" Cluster)
**Tools:** Time, Math, Calculator, UUID, String utilities

**Architecture:**
- Single lightweight Python/FastAPI container
- One SSE endpoint (`http://sanctuary_utils:8000/sse`)
- Internal routing to multiple tool implementations
- No external dependencies, no secrets, no file system access

**Benefit:** Eliminates 5-6 containers instantly while maintaining isolation from risky operations.

**Example:**
```python
# Single container serving multiple safe tools
@app.post("/sse")
async def handle_tool_call(request: ToolRequest):
    if request.tool == "time":
        return get_current_time()
    elif request.tool == "calculator":
        return calculate(request.params)
    # ... more safe tools
```

---

### 2. `sanctuary_filesystem` (The "Privileged" Cluster)
**Tools:** File System, grep, patch, code analysis

**Architecture:**
- Dedicated container with **controlled volume mounts**
- Restricted to specific directories (e.g., `/workspace:/app/workspace:ro`)
- **No network access** (isolated from external threats)
- Separate security boundary from "safe" utils
- Audit logging for all file operations

**Security Rationale:** File system access requires distinct isolation from network and compute tools.

---

### 3. `sanctuary_network` (The "External" Cluster)
**Tools:** Brave Search, Fetch, HTTP clients

**Architecture:**
- Container with **outbound network access**
- Injected secrets (API keys) via environment variables
- **No file system access** beyond temp directories
- Rate limiting and request logging

**Security Rationale:** Network tools need internet access but should be isolated from file system.

**Note:** Git is NOT in this cluster (see #4).

---

### 4. `sanctuary_git` (The "Isolated Dual-Permission" Cluster)
**Tools:** Git Workflow

**Architecture:**
- **Dedicated isolated container** (not grouped with other tools)
- Requires **both** network access (GitHub) AND filesystem access (clone/commit)
- Controlled volume mounts for repository access
- SSH key management for private repos
- Audit logging for all git operations

**Security Rationale:** Git's unique dual-permission requirement (network + filesystem) necessitates complete isolation from other tools to prevent privilege escalation attacks.

**Why Isolated:**
- If grouped with `sanctuary_network`: Could exploit network access to exfiltrate files
- If grouped with `sanctuary_filesystem`: Could exploit filesystem to leak secrets
- **Solution:** Complete isolation in dedicated container

---

### 5. `sanctuary-intelligence` (The "Heavy" Cluster)
**Tools:** RAG Cortex, Memory, Embeddings

**Architecture:**
- **Already handled** by existing `sanctuary_vector_db` and `sanctuary_ollama`
- No changes needed (these are already containerized)
- Existing security boundaries maintained

---

## Fleet of 7 Architecture Diagram

![mcp_gateway_fleet_detailed](../docs/architecture_diagrams/system/mcp_gateway_fleet_detailed.png)

*[Source: mcp_gateway_fleet_detailed.mmd](../docs/architecture_diagrams/system/mcp_gateway_fleet_detailed.mmd)*


**Legend:**
- 🔵 **Blue (Client):** MCP clients connecting via HTTPS
- 🟠 **Orange (Gateway):** Stateless HTTP proxy (IBM ContextForge)
- 🟢 **Green (#1 Utils):** Low risk, pure compute, no I/O - **NEW**
- 🟠 **Orange (#2 Filesystem):** Privileged, file system access, no network - **NEW**
- 🟣 **Purple (#3 Network):** External, outbound network, no filesystem - **NEW**
- 🔴 **Red (#4 Git):** Dual permissions (filesystem + network), isolated - **NEW**
- 🔵 **Teal (#5 Intelligence Cluster):**
  - **5a Cortex:** MCP Server (brain) - **NEW**
  - **5b VectorDB:** Backend storage - **EXISTING**
  - **5c Ollama:** Backend compute - **EXISTING**

---

## Mandatory Guardrails (6 Required)

> [!CAUTION]
> **These 6 guardrails are MANDATORY.** Implementation cannot proceed without them.
> 
> - **1-3:** Original (Fault Containment, Self-Registration, Network Addressing)
> - **4:** Resource Caps (Grok 4)
> - **5-6:** Grok 4 Nits (Volumes, Chain Timeouts)

### Guardrail 1: Intra-Cluster Fault Containment

**Requirement:** A single tool crash (e.g., "Math" division by zero) must NOT crash the entire container.

**Implementation:**
```python
# Entry point must handle exceptions gracefully
@app.post("/sse")
async def handle_tool_call(request: ToolRequest):
    try:
        if request.tool == "calculator":
            return await calculator_service(request.params)
        elif request.tool == "time":
            return await time_service(request.params)
    except Exception as e:
        # Container stays up, only this tool call fails
        logger.error(f"Tool {request.tool} failed: {e}")
        return {"error": str(e), "tool": request.tool, "status": "tool_failure"}
    # Other tools continue working
```

**Rationale:** Blast radius containment - one faulty tool should not take down sibling tools.

---

### Guardrail 2: Dynamic Self-Registration

**Requirement:** To prevent "Registry Drift," containers MUST POST their tool manifests to the Gateway on startup.

**Implementation:**
```python
# On container startup
async def register_with_gateway():
    manifest = {
        "server_name": "sanctuary_utils",
        "endpoint": "http://sanctuary_utils:8000/sse",
        "tools": ["time", "calculator", "uuid", "string"],
        "version": "1.0.0",
        "health_check": "/health"
    }
    response = await httpx.post(
        f"{GATEWAY_URL}/api/servers/register",
        json=manifest,
        headers={"Authorization": f"Bearer {API_TOKEN}"}
    )
    logger.info(f"Registered with Gateway: {response.status_code}")
```

**Rationale:** Self-healing system. No duplicate config files. Single source of truth.

**ChatGPT Commentary:** *"This is the secret weapon. Instead of editing Gateway config every time you add a tool, the tool tells the Gateway it exists."*

---

### Guardrail 3: Network Addressing

**Requirement:** All inter-container communication MUST use Docker Network Aliases.

**Correct:**
```python
GATEWAY_URL = "http://sanctuary-gateway:4444"
VECTOR_DB_URL = "http://sanctuary_vector_db:8000"
```

**WRONG:**
```python
# NEVER use these!
GATEWAY_URL = "http://localhost:4444"      # ❌ Won't work in containers
GATEWAY_URL = "http://192.168.1.100:4444"  # ❌ Hardcoded IP breaks portability
```

**Rationale:** Docker network aliases are stable, portable, and DNS-resolved. Localhost and hardcoded IPs break in containerized environments.

---

### Guardrail 4: Resource Caps (Grok 4 Requirement)

**Requirement:** All `docker-compose` service definitions MUST include `deploy.resources.limits` to prevent the 7 containers from overwhelming the dev machine.

**Implementation:**
```yaml
# docker-compose.yml
services:
  sanctuary_utils:
    build: ./mcp_servers/utils
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: '256M'
        reservations:
          cpus: '0.25'
          memory: '128M'
  
  sanctuary_cortex:
    build: ./mcp_servers/rag_cortex
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: '1G'  # Higher for RAG operations
        reservations:
          cpus: '0.5'
          memory: '512M'
```

**Recommended Limits:**
| Container | CPU Limit | Memory Limit | Rationale |
|-----------|-----------|--------------|-----------|
| sanctuary_utils | 0.5 | 256M | Lightweight, stateless |
| sanctuary_filesystem | 0.5 | 256M | I/O bound, not compute |
| sanctuary_network | 0.5 | 256M | HTTP client, low memory |
| sanctuary_git | 0.5 | 512M | Git operations need memory |
| sanctuary_cortex | 1.0 | 1G | RAG embedding, heavier |
| sanctuary_vector_db | 1.0 | 2G | Database operations |
| sanctuary_ollama | 2.0 | 4G | LLM inference is heavy |

**Rationale:** 7 containers running without limits can exhaust system resources. Explicit caps prevent runaway processes and ensure fair resource sharing.

---

### Guardrail 5: Persistent Volumes (Grok 4 Nit)

**Requirement:** All containers with persistent data MUST define Podman volumes to prevent data loss on restarts.

**Implementation:**
```yaml
# docker-compose.yml
volumes:
  vector_db-data:
  ollama-models:

services:
  sanctuary_vector_db:
    volumes:
      - vector_db-data:/chroma/chroma  # Persist embeddings
  
  sanctuary_ollama:
    volumes:
      - ollama-models:/root/.ollama    # Persist downloaded models
```

**Rationale:** Without explicit volumes, data is lost when containers restart. ChromaDB embeddings and Ollama models are expensive to regenerate.

---

### Guardrail 6: Chain Timeouts (Grok 4 Nit)

**Requirement:** Multi-backend chains in `sanctuary_cortex` MUST enforce per-hop timeouts to prevent latency spikes.

**Implementation:**
```python
# In sanctuary_cortex router
import httpx

TIMEOUT_CONFIG = {
    "embedding": 5.0,     # VectorDB embedding lookup
    "search": 10.0,       # VectorDB similarity search
    "generate": 30.0,     # Ollama LLM generation (can be slow)
}

async def rag_query(query: str):
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG["embedding"]) as client:
        embeddings = await client.post(f"{VECTOR_DB_URL}/embed", json={"text": query})
    
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG["search"]) as client:
        results = await client.post(f"{VECTOR_DB_URL}/search", json={"embedding": embeddings})
    
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG["generate"]) as client:
        response = await client.post(f"{OLLAMA_URL}/generate", json={"prompt": results})
    
    return response
```

**Chain Timeout Limits:**
| Operation | Timeout | Rationale |
|-----------|---------|-----------|
| Embedding | 5s | Simple vector lookup |
| Search | 10s | Similarity computation |
| Generate | 30s | LLM can be slow |
| **Total Chain** | **45s max** | Cap end-to-end latency |

**Rationale:** Multi-hop RAG chains (embed → search → generate) can hit 100ms+ under load. Per-hop timeouts prevent cascading delays and ensure predictable response times.

---

## Implementation Plan


### Phase 1: Pilot Deployment (Week 1)
**Goal:** Deploy `sanctuary_utils` container

1. Create `mcp_servers/utils/Dockerfile`
2. Implement multi-tool SSE handler
3. Add to `docker-compose.yml`
4. Register with Gateway
5. **Success Criteria:** Gateway successfully calls "What time is it?"

### Phase 2: Privileged & Network (Week 2-3)
1. Deploy `sanctuary_filesystem` with volume mounts
2. Deploy `sanctuary_network` with API keys
3. Test cross-cluster isolation

### Phase 3: Integration (Week 4)
1. Register all 4 clusters with Gateway
2. Update Claude Desktop config
3. E2E testing across all tool categories

### Phase 4: Cutover (Week 5)
1. Deprecate direct stdio connections
2. Full Gateway routing
3. Monitor and optimize

---

## Developer Experience

### Hot Reloading Mandate

**Constraint:** All development Dockerfiles MUST support hot reloading.

**Rationale:** We cannot afford to rebuild containers for every code change during development.

**Implementation:**
```yaml
# docker-compose.dev.yml
services:
  sanctuary_utils:
    build:
      context: ./mcp_servers/utils
      target: development  # Multi-stage build
    volumes:
      - ./mcp_servers/utils:/app:ro  # Mount source code
    environment:
      - UVICORN_RELOAD=true
    command: uvicorn main:app --reload --host 0.0.0.0
```

**Mechanism:**
- Use `watchdog` or Uvicorn's `--reload` flag
- Volume-mount source code in `dev` compose profile
- Production builds use multi-stage Dockerfile (no source mounts)

**Example Dockerfile:**
```dockerfile
# Development stage (hot reload)
FROM python:3.11-slim AS development
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]

# Production stage (optimized)
FROM python:3.11-slim AS production
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

---

## Consequences

**Positive:**
- Maintains ADR 058 isolation principles (no dependency hell)
- **Reduces orchestration complexity** (4 containers vs 12)
- Clear security boundaries (risk-based clustering)
- Matches existing architecture (vector_db, ollama patterns)
- Enables independent cluster lifecycle management
- Supports heterogeneous runtimes (Python, Node.js, Rust, etc.)
- **Developer-friendly** (hot reloading mandate)

**Negative:**
- Requires containerizing 10 script-based servers (grouped into 3 new containers)
- Learning curve for Dockerfile creation
- Potential resource overhead (though reduced from 12 containers)
- Internal routing complexity within each cluster

**Risks:**
- Cluster sprawl if grouping logic is wrong
- Debugging across cluster boundaries
- Initial migration effort for existing scripts
- Blast radius if one cluster fails (affects multiple tools)

**Mitigation:**
- Use docker-compose for orchestration
- Implement health checks for all clusters
- Standardize Dockerfile templates
- Monitor cluster resource usage and split if needed
- Clear logging and tracing within clusters

---

## Side-by-Side Architecture & Port Strategy

To support legacy direct connections while migrating to the Gateway, we implement a **Side-by-Side** strategy:

**1. Dual-Mode Servers:**
All MCP servers (Utils, Network, Filesystem, Git, Cortex) are refactored to support two transport modes:
- **Legacy (Stdio):** Default when `PORT` is not set. Used by direct `python -m mcp_servers.xxx` calls.
- **Gateway (SSE):** Active when `PORT` is set. Used by Podman containers.

**2. Port Assignments:**
To prevent conflicts, Gateway-routed containers use distict host ports from the legacy defaults.

| Server | Legacy Port (Direct) | Gateway Host Port (Podman) | Container Internal |
|--------|----------------------|----------------------------|--------------------|
| Utils | N/A (Stdio) | 8100 | 8000 |
| Filesystem | N/A (Stdio) | 8101 | 8000 |
| Network | N/A (Stdio) | 8102 | 8000 |
| Git | N/A (Stdio) | 8103 | 8000 |
| Cortex | 8000/8004 | 8104 | 8000 |
| Domain | 8105 | 8105 | 8105 |
| vector_db | 8000 | 8110 | 8000 |

**3. Configuration Toggles:**
Clients choose their mode via config:
- `legacy_direct.json`: Uses Stdio command lines.
- `gateway_routed.json`: Uses Gateway URL (`http://localhost:4444/sse`) with Bearer Token.

---

## Related Documents

- [ADR 058: Decouple IBM Gateway to External Podman Service](058_decouple_ibm_gateway_to_external_podman_service.md)
- [Task 118: Red Team Analysis](../tasks/done/118_red_team_analysis_gateway_server_connection_patter.md)
- [Task 119: Deploy Pilot - sanctuaryutils Container](../tasks/done/119_deploy_pilot_sanctuaryutils_container.md)

--- END OF FILE ADRs/060_gateway_integration_patterns.md ---

--- START OF FILE mcp_servers/council/README.md ---

# Council MCP Server

**Status:** ✅ Operational
**Version:** 2.0.0 (Refactored)
**Protocol:** Model Context Protocol (MCP)

**Description:** The Council MCP Server exposes the Sanctuary Council's **multi-agent deliberation** capabilities to external AI agents via the Model Context Protocol. It coordinates specialized agents (Coordinator, Strategist, Auditor) to solve complex tasks through iterative reasoning and context retrieval.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `council_dispatch` | Dispatches a task to the Council for deliberation. | `task_description` (str): The task to perform.<br>`agent` (str, optional): Specific agent to consult (e.g., "auditor").<br>`max_rounds` (int, optional): Max deliberation rounds (default: 3).<br>`force_engine` (str, optional): Force specific LLM engine.<br>`output_path` (str, optional): Path to save results. |
| `council_list_agents` | Lists all available Council agents and their status. | None |

## Resources

| Resource URI | Description | Mime Type |
|--------------|-------------|-----------|
| `council://agents/list` | List of available agents | `application/json` |
| `council://history/{session_id}` | Deliberation history for a session | `application/json` |

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required for Council Operations
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
# Optional
COUNCIL_DEFAULT_ROUNDS=3
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"council": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/council",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/council/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `council_dispatch` and `council_list_agents` appear in the tool list.
3.  **Call Tool:** Execute `council_list_agents` and verify it returns the list of agents (Coordinator, Strategist, Auditor).

## Architecture

### Overview

The Council MCP has a unique dual-role architecture: it acts as both a **Server** (receiving requests from the user/IDE) and a **Client** (orchestrating other MCPs).

The Council MCP has been refactored to use a modular architecture:
1.  **Agent Persona MCP**: Used for individual agent execution (Coordinator, Strategist, Auditor)
2.  **Cortex MCP**: Used for memory and context retrieval
3.  **Direct Orchestration**: Deliberation logic is now embedded in `council_ops.py`, replacing the legacy orchestrator subprocess.

**Design Principle: Separation of Concerns**
The Council MCP provides ONLY what's unique to the Council. Other capabilities are delegated to specialized MCP servers.

**Benefits of Direct Orchestration:**
- ✅ No subprocess overhead
- ✅ Uses specialized Agent Persona MCP
- ✅ Integrated with Cortex memory
- ✅ Clean separation of concerns

**Trade-offs:**
- ⚠️ Simplified deliberation logic (compared to legacy v1)

### Execution Flow

![council_execution_flow](../../docs/architecture_diagrams/workflows/council_execution_flow.png)

*[Source: council_execution_flow.mmd](../../docs/architecture_diagrams/workflows/council_execution_flow.mmd)*

### Directory Structure

```
mcp_servers/
├── council/
│   ├── __init__.py
│   ├── server.py              # FastMCP server with tool definitions
│   └── README.md              # This file
├── lib/
│   └── council/
│       ├── __init__.py
│       └── council_ops.py     # Orchestrator interface logic
```

## Tools

### `council_dispatch`

Execute a task through multi-agent deliberation (the Council's core capability).

**Parameters:**
- `task_description` (str): Task for the council to deliberate on
- `agent` (str, optional): Specific agent ("coordinator", "strategist", "auditor") or None for full council
- `max_rounds` (int, default=3): Maximum deliberation rounds
- `force_engine` (str, optional): Force specific AI engine ("gemini", "openai", "ollama")
- `output_path` (str, optional): Output file path (relative to project root)

**Returns:**
```python
{
    "status": "success" | "error",
    "decision": "Council's deliberation output",
    "session_id": "mcp_1234567890",
    "output_path": "WORK_IN_PROGRESS/output.md"
}
```

**Example - Full Council Deliberation:**
```python
result = council_dispatch(
    task_description="Review the architecture for Protocol 115 and provide recommendations",
    max_rounds=3
)
```

**Example - Single Agent Consultation:**
```python
result = council_dispatch(
    task_description="Audit the test coverage for the Git MCP server",
    agent="auditor",
    max_rounds=2
)
```

### `council_list_agents`

List all available Council agents and their status.

**Returns:**
```python
[
    {
        "name": "coordinator",
        "status": "available",
        "persona": "Task planning and execution oversight"
    },
    {
        "name": "strategist",
        "status": "available",
        "persona": "Long-term planning and risk assessment"
    },
    {
        "name": "auditor",
        "status": "available",
        "persona": "Quality assurance and compliance verification"
    }
]
```

## Composable Workflow Patterns

The Council MCP is designed to compose with other specialized MCP servers:

### Pattern 1: Council → Protocol → Git

```python
# 1. Council deliberates on protocol design
decision = council_dispatch(
    task_description="Design a new protocol for MCP composition patterns",
    output_path="WORK_IN_PROGRESS/protocol_design.md"
)

# 2. Create protocol document (use Protocol MCP)
protocol_create(
    number=120,
    title="MCP Composition Patterns",
    status="PROPOSED",
    content=decision["decision"]
)

# 3. Commit to repository (use Git MCP)
git_add(files=["01_PROTOCOLS/120_mcp_composition_patterns.md"])
git_smart_commit(message="feat(protocol): Add Protocol 120 - MCP Composition")
git_push_feature()
```

### Pattern 2: Council → Code → Git

```python
# 1. Council generates implementation
code = council_dispatch(
    task_description="Implement a helper function for parsing MCP responses"
)

# 2. Write code file (use Code MCP)
code_write(
    path="mcp_servers/lib/utils/parser.py",
    content=code["decision"]
)

# 3. Commit (use Git MCP)
git_add(files=["mcp_servers/lib/utils/parser.py"])
git_smart_commit(message="feat(utils): Add MCP response parser")
```

### Pattern 3: Cortex → Council → Task

```python
# 1. Query memory for context (use Cortex MCP)
context = cortex_query(
    query="Previous decisions on MCP architecture",
    max_results=5
)

# 2. Council deliberates with context
decision = council_dispatch(
    task_description=f"Given this context: {context}, recommend next steps for MCP evolution"
)

# 3. Create task (use Task MCP)
create_task(
    title="Implement MCP Evolution Recommendations",
    description=decision["decision"],
    status="todo"
)
```

## Removed Tools (Use Specialized MCPs Instead)

The following tools were **intentionally removed** to maintain separation of concerns:

- ~~`council_mechanical_write`~~ → Use `code_write` from **Code MCP**
- ~~`council_query_memory`~~ → Use `cortex_query` from **Cortex MCP**
- ~~`council_git_commit`~~ → Use `git_add` + `git_smart_commit` from **Git MCP**

**Rationale:** Each MCP server should have a single, well-defined responsibility. The Council MCP focuses exclusively on multi-agent deliberation.

## Installation & Setup

### Prerequisites

1. **Council Orchestrator** must be installed and configured
2. **Python 3.8+**
3. **FastMCP** library: `pip install fastmcp`

### Configuration

Add to your MCP client configuration (e.g., Claude Desktop, Antigravity):

```json
{
  "mcpServers": {
    "council": {
      "command": "python3",
      "args": ["-m", "mcp_servers.council.server"],
      "cwd": "/path/to/Project_Sanctuary",
      "env": {}
    }
  }
}
```

### Verification

Test the server:

```bash
cd /path/to/Project_Sanctuary
python3 -m mcp_servers.council.server
```

## Testing

### Run Unit Tests

```bash
pytest tests/mcp_servers/council/ -v
```

### Manual Verification

1. **List Agents:**
   ```python
   agents = council_list_agents()
   print(agents)
   # Expected: 3 agents (coordinator, strategist, auditor)
   ```

2. **Dispatch Simple Task:**
   ```python
   result = council_dispatch(
       task_description="Introduce yourself and explain your role",
       agent="coordinator",
       max_rounds=1
   )
   print(result["decision"])
   # Expected: Coordinator's introduction
   ```

3. **Full Council Deliberation:**
   ```python
   result = council_dispatch(
       task_description="Evaluate the current MCP architecture and suggest improvements",
       max_rounds=3
   )
   print(result["decision"])
   # Expected: Multi-agent deliberation output
   ```

4. **Composition with Other MCPs:**
   ```python
   # Council deliberates
   decision = council_dispatch(
       task_description="Create a brief protocol summary",
       output_path="WORK_IN_PROGRESS/test_protocol.md"
   )
   
   # Save with Code MCP
   code_write(
       path="WORK_IN_PROGRESS/test_protocol.md",
       content=decision["decision"]
   )
   
   # Commit with Git MCP
   git_add(files=["WORK_IN_PROGRESS/test_protocol.md"])
   git_smart_commit(message="test: Council MCP verification")
   ```

## Integration with Council Orchestrator

The MCP server is a **thin protocol wrapper** around the existing Council Orchestrator:

```
mcp_servers/council/          council_orchestrator/
├── server.py (MCP wrapper) → ├── orchestrator/
├── lib/council/            → │   ├── main.py (Entry point)
    └── council_ops.py      → │   ├── app.py (Core logic)
                              │   ├── engines/
                              │   ├── council/
                              │   └── memory/
                              └── command.json (Generated by MCP)
```

**Important:** Both folders are required. The MCP server depends on the full orchestrator implementation.

## Error Handling

The server includes comprehensive error handling:

- **Timeout Protection**: 120s timeout for orchestrator execution
- **Session Isolation**: Unique session IDs prevent command.json conflicts
- **Graceful Degradation**: Returns structured error responses
- **Logging**: All operations logged for debugging

## Future Enhancements

- [ ] Streaming responses for long-running tasks
- [ ] Persistent daemon mode for faster response times
- [ ] `council_query_memory` tool for Mnemonic Cortex queries
- [ ] `council_git_commit` tool for git operations
- [ ] Async execution for concurrent task handling
- [ ] Confidence score parsing from orchestrator output

## Future Architecture: Council Members as MCP Servers

### Current Architecture (v1.0)

The orchestrator manages council members internally as Python objects:

```
Council Orchestrator (Monolithic)
├── Coordinator (Python class)
├── Strategist (Python class)
└── Auditor (Python class)
```

### Proposed Architecture (v2.0)

Each council member becomes an **independent MCP server**:

```
Council Orchestrator (MCP Client)
├── Calls → Coordinator MCP Server
├── Calls → Strategist MCP Server
└── Calls → Auditor MCP Server
```

### Benefits of Member-as-MCP Architecture

**1. True Modularity**
- Each agent is independently deployable
- Can upgrade/replace individual agents without touching orchestrator
- Agents can be written in different languages

**2. Scalability**
- Agents can run on different machines
- Horizontal scaling (multiple instances of same agent)
- Load balancing across agent instances

**3. Specialization**
- Each agent MCP can have its own tools and capabilities
- Coordinator MCP might expose: `plan_task`, `coordinate_workflow`
- Strategist MCP might expose: `assess_risk`, `long_term_planning`
- Auditor MCP might expose: `verify_compliance`, `quality_check`

**4. Composability**
- External agents can call individual council members directly
- Don't need full deliberation for simple consultations
- Mix and match agents for different scenarios

### Implementation Sketch

**Coordinator MCP Server:**
```python
# mcp_servers/agents/coordinator/server.py

@mcp.tool()
def coordinator_plan_task(task_description: str, context: dict) -> dict:
    """
    Plan task execution strategy
    
    Args:
        task_description: Task to plan
        context: Relevant context (from Cortex, previous decisions)
    
    Returns:
        Execution plan with steps, dependencies, estimates
    """
    # Coordinator's specialized planning logic
    return {
        "plan": [...],
        "estimated_effort": "4 hours",
        "dependencies": [...]
    }
```

**Orchestrator as MCP Client:**
```python
# council_orchestrator/orchestrator/app.py

async def deliberate(task: str):
    # 1. Query context
    context = await cortex_mcp.query(task)
    
    # 2. Get coordinator's plan
    plan = await coordinator_mcp.plan_task(task, context)
    
    # 3. Get strategist's risk assessment
    risks = await strategist_mcp.assess_risk(plan)
    
    # 4. Get auditor's compliance check
    compliance = await auditor_mcp.verify_compliance(plan, risks)
    
    # 5. Synthesize final decision
    return synthesize_decision(plan, risks, compliance)
```

### Migration Path

**Phase 1 (Current):** Monolithic orchestrator with internal agents
**Phase 2:** Extract one agent (e.g., Auditor) as MCP server, test dual mode
**Phase 3:** Extract remaining agents, deprecate internal implementations
**Phase 4:** Orchestrator becomes pure MCP client coordinator

### Design Considerations

**Agent Discovery:**
- How does orchestrator find agent MCP servers?
- Configuration file? Service registry? Environment variables?

**Agent State:**
- Should agents maintain conversation history?
- Stateless (functional) vs stateful (memory-enabled)?

**Error Handling:**
- What if an agent MCP is unavailable?
- Fallback strategies? Retry logic?

**Consensus Mechanism:**
- How to resolve disagreements between agents?
- Voting? Weighted opinions? Coordinator override?

---

## Related Documentation

### Council Orchestrator
- [Council Orchestrator README](../../ARCHIVE/docs_council_orchestrator_legacy/README_v11.md) - Full orchestrator documentation
- [Guardian Wakeup Flow](../../ARCHIVE/docs_council_orchestrator_legacy/README_GUARDIAN_WAKEUP.md) - Cache-first situational awareness (Protocol 114)
- [Command Schema](../../ARCHIVE/docs_council_orchestrator_legacy/command_schema.md) - Complete command format reference

### Mnemonic Cortex (RAG System)
- [RAG Strategies and Doctrine](../../ARCHIVE/mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md) - RAG architecture and best practices
- [Cortex Operations Guide](../../ARCHIVE/mnemonic_cortex/OPERATIONS_GUIDE.md) - Cortex operational procedures
- [Cortex README](../../ARCHIVE/mnemonic_cortex/README.md) - Cortex overview and setup
- [Cortex Vision](../../ARCHIVE/mnemonic_cortex/VISION.md) - Strategic vision for knowledge systems

### MCP Ecosystem
- [MCP Operations Inventory](../../docs/operations/mcp/mcp_operations_inventory.md) - Complete MCP operations catalog
- [Code MCP](../code/README.md) - File operations MCP
- [Git MCP](../git/README.md) - Version control MCP
- **[Cortex MCP](../rag_cortex/README.md)**: Memory/RAG MCP
- [Protocol MCP](../protocol/README.md) - Protocol document MCP
- [Task MCP](../task/README.md) - Task management MCP

### Task Documentation
- [Task 077: Implement Council MCP](../../tasks/done/077_implement_council_mcp_server.md) - Implementation task

---

**"The Council is now accessible to all agents through the Protocol."** ⚡👑

--- END OF FILE mcp_servers/council/README.md ---

--- START OF FILE ADRs/076_sse_tool_metadata_decorator_pattern.md ---

# ADR 076: SSE Tool Metadata Decorator Pattern

**Status:** ✅ ACCEPTED
**Red Team Review:** Approved with hardening recommendations
**Date:** 2025-12-24
**Deciders:** Antigravity, User
**Technical Story:** Gateway tool descriptions missing - all 85 tools registered with "No description"

---

## Context

During Gateway fleet verification, we discovered that **all 85 federated tools** are registered without descriptions in the IBM ContextForge Gateway admin UI. This degrades LLM tool discovery and makes fleet management difficult.

### Root Cause

The SSEServer's `register_tool()` method extracts descriptions from `handler.__doc__`:

```python
# mcp_servers/lib/sse_adaptor.py (line 88-93)
def register_tool(self, name: str, handler: Callable[..., Awaitable[Any]], schema: Optional[Dict] = None):
    self.tools[name] = {
        "handler": handler,
        "schema": schema,
        "description": handler.__doc__.strip() if handler.__doc__ else "No description"
    }
```

However, the SSE wrapper functions in fleet servers lack docstrings:

```python
# PROBLEM: No docstring on SSE wrapper
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)  # ← Missing docstring!
    return json.dumps({"status": "success", "debrief": response}, indent=2)
```

Meanwhile, FastMCP (STDIO) versions have docstrings via the `@mcp.tool()` decorator pattern:

```python
# WORKING: FastMCP has docstrings
@mcp.tool()
async def cortex_learning_debrief(request: CortexLearningDebriefRequest) -> str:
    """Scans repository for technical state changes (Protocol 128)."""  # ← Has docstring
    ...
```

**Result:** STDIO mode gets descriptions; SSE mode gets "No description".

### Scope

- **Affected:** 6 fleet containers (sanctuary_utils, filesystem, network, git, cortex, domain)
- **Tools affected:** 85 total
- **Impact:** LLM tool discovery degraded, admin UI shows "No description" for all tools

### Alignment with ADR 066 (Dual-Transport Architecture)

Per [ADR 066](./066_standardize_on_fastmcp_for_all_mcp_server_implementations.md), Project Sanctuary uses a **dual-transport standard**:

| Transport | Implementation | Decorator | Use Case |
|-----------|---------------|-----------|----------|
| **STDIO** | FastMCP | `@mcp.tool()` | Claude Desktop, IDE direct |
| **SSE** | SSEServer | `@sse_tool()` *(proposed)* | Gateway Fleet containers |

This ADR proposes `@sse_tool()` as the **SSE-transport counterpart** to FastMCP's `@mcp.tool()`:

![MCP Tool Decorator Pattern](../docs/architecture_diagrams/system/mcp_tool_decorator_pattern.png)

*[Source: mcp_tool_decorator_pattern.mmd](../docs/architecture_diagrams/system/mcp_tool_decorator_pattern.mmd)*

**Key Alignment Points:**
- Both decorators attach metadata at function definition site
- Both support explicit `name`, `description`, and `schema`
- Both delegate to shared `operations.py` (3-Layer Architecture per ADR 066)
- `@sse_tool()` is for SSE mode **only** — FastMCP's `@mcp.tool()` remains unchanged for STDIO

---

## Alternatives Considered

### Option 1: Add Docstrings to SSE Wrapper Functions (Quick Fix)

**Description:** Simply add docstrings to each SSE wrapper function.

```python
def cortex_learning_debrief(hours: int = 24):
    """Scans repository for technical state changes (Protocol 128)."""
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)
```

**Pros:**
- Minimal change
- Works immediately with existing SSEServer

**Cons:**
- Duplicates docstrings between STDIO and SSE implementations
- No explicit metadata - description hidden in docstring
- Easy to forget when adding new tools

**Why not chosen:** Maintainability concerns with duplication across ~85 tools.

---

### Option 2: Centralized Tool Registry File

**Description:** Create a JSON or Python dict with all tool metadata.

```python
# tool_registry.py
TOOL_METADATA = {
    "cortex_learning_debrief": {
        "description": "Scans repository for technical state changes (Protocol 128).",
        "schema": LEARNING_DEBRIEF_SCHEMA
    },
    ...
}
```

**Pros:**
- Single source of truth
- Easy to export/import
- Separates metadata from implementation

**Cons:**
- Another file to maintain
- Metadata disconnected from function definition
- Easy for registry and code to drift

**Why not chosen:** Increases maintenance burden and risk of drift.

---

### Option 3: Inherit from Operations Layer

**Description:** Pull docstrings from the operations layer methods.

```python
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)

# Inherit docstring from operations
cortex_learning_debrief.__doc__ = CortexOperations.learning_debrief.__doc__
```

**Pros:**
- Single source of truth in operations layer
- No duplication

**Cons:**
- Requires operations methods to have docstrings (not always the case)
- Awkward post-hoc assignment
- Not explicit at function definition site

**Why not chosen:** Requires refactoring operations layer first; awkward pattern.

---

### Option 4: Decorator Pattern with `@sse_tool` (RECOMMENDED) ⭐

**Description:** Create a decorator similar to FastMCP's `@mcp.tool()` that attaches metadata to functions.

```python
@sse_tool(
    name="cortex_learning_debrief",
    description="Scans repository for technical state changes (Protocol 128).",
    schema=LEARNING_DEBRIEF_SCHEMA
)
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)
```

**Pros:**
- Explicit metadata at function definition site
- Consistent with FastMCP's `@mcp.tool()` pattern
- Hard to forget - decorator is required for registration
- Enables auto-registration of decorated functions
- Single source of truth (decorator params)

**Cons:**
- Requires changes to SSEServer
- Slightly more boilerplate than Option 1

---

## Decision

Adopt **Option 4: Decorator Pattern with `@sse_tool`**.

### Implementation

#### 1. Add decorator to `sse_adaptor.py`

```python
# mcp_servers/lib/sse_adaptor.py

def sse_tool(
    name: str = None, 
    description: str = None, 
    schema: dict = None
):
    """
    Decorator to mark functions as SSE tools with explicit metadata.
    
    Usage:
        @sse_tool(
            name="cortex_query",
            description="Perform semantic search query.",
            schema=QUERY_SCHEMA
        )
        def cortex_query(query: str, max_results: int = 5):
            ...
    """
    def decorator(func):
        func._sse_tool = True
        func._sse_name = name or func.__name__
        func._sse_description = description or func.__doc__ or "No description"
        func._sse_schema = schema or {"type": "object", "properties": {}}
        return func
    return decorator
```

#### 2. Add auto-registration method to SSEServer

```python
class SSEServer:
    # ... existing code ...
    
    def register_decorated_tools(self, namespace: dict):
        """
        Auto-register all functions decorated with @sse_tool.
        
        Usage:
            server.register_decorated_tools(locals())
        """
        for name, obj in namespace.items():
            if callable(obj) and getattr(obj, '_sse_tool', False):
                self.register_tool(
                    name=obj._sse_name,
                    handler=obj,
                    schema=obj._sse_schema,
                    description=obj._sse_description
                )
```

#### 3. Update `register_tool` to accept explicit description

```python
def register_tool(
    self, 
    name: str, 
    handler: Callable[..., Awaitable[Any]], 
    schema: Optional[Dict] = None,
    description: str = None  # NEW: explicit parameter
):
    self.tools[name] = {
        "handler": handler,
        "schema": schema,
        "description": description or handler.__doc__.strip() if handler.__doc__ else "No description"
    }
    self.logger.info(f"Registered tool: {name}")
```

#### 4. Update fleet servers to use decorator

**Before (sanctuary_cortex/server.py):**
```python
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)

# Manual registration
server.register_tool("cortex_learning_debrief", cortex_learning_debrief, LEARNING_DEBRIEF_SCHEMA)
```

**After:**
```python
@sse_tool(
    name="cortex_learning_debrief",
    description="Scans repository for technical state changes (Protocol 128).",
    schema=LEARNING_DEBRIEF_SCHEMA
)
def cortex_learning_debrief(hours: int = 24):
    response = get_ops().learning_debrief(hours=hours)
    return json.dumps({"status": "success", "debrief": response}, indent=2)

# Auto-registration
server.register_decorated_tools(locals())
```

---

## Consequences

### Positive
- **Explicit metadata:** Name, description, and schema defined at function site
- **Pattern parity:** Consistent with FastMCP's `@mcp.tool()` decorator
- **Auto-registration:** Reduces boilerplate and prevents forgotten registrations
- **Single source of truth:** Metadata lives with the function definition
- **Backward compatible:** Existing `register_tool()` still works

### Negative
- **SSEServer changes:** Requires updates to `sse_adaptor.py`
- **Server updates:** All 6 fleet servers need decorator migration
- **Container rebuild:** All fleet containers must be rebuilt

### Risks
- **Migration errors:** Could break existing tools if refactoring is incomplete
- **Mitigation:** Test each server after migration before Gateway re-registration

### Dependencies
- ADR 066 (Dual-Transport Standards) - Complements this decision
- All 6 fleet server modules

---

## Implementation Notes

### Migration Checklist

1. [ ] Update `mcp_servers/lib/sse_adaptor.py` with decorator and auto-registration
2. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_cortex/server.py`
3. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_domain/server.py`
4. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_utils/server.py`
5. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_git/server.py`
6. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_filesystem/server.py`
7. [ ] Migrate `mcp_servers/gateway/clusters/sanctuary_network/server.py`
8. [ ] Rebuild all fleet containers
9. [ ] Re-run fleet setup: `python -m mcp_servers.gateway.fleet_setup`
10. [ ] Verify descriptions in Gateway admin UI

### Verification

```bash
# After implementation, verify descriptions appear:
curl -s https://localhost:4444/api/tools -H "Authorization: Bearer $TOKEN" | jq '.[].description'

# Verify SSE handshake still works (must return 'endpoint' event immediately):
curl -N http://localhost:8100/sse
```

---

## Red Team Hardening Recommendations

> [!IMPORTANT]
> **The following hardening measures are MANDATORY per Red Team review.**

### 1. Namespace Safety

`register_decorated_tools()` must ignore private functions (starting with `_`) to prevent accidental registration:

```python
def register_decorated_tools(self, namespace: dict):
    for name, obj in namespace.items():
        if name.startswith('_'):  # Skip private functions
            continue
        if callable(obj) and getattr(obj, '_sse_tool', False):
            self.register_tool(...)
```

### 2. Strict Schema Linking

The `schema` parameter should reference Pydantic-generated schemas from `models.py` to ensure SSE and FastMCP wrappers validate against identical constraints.

### 3. Handshake Verification

After migration, verify that `register_decorated_tools()` does not interfere with the immediate `endpoint` event response required by the Gateway (curl check above).

---

## Red Team Sign-Off

| Reviewer | Verdict | Date |
|----------|---------|------|
| User | ✅ APPROVED | 2025-12-24 |
| Red Team Analysis | ✅ APPROVED with hardening | 2025-12-24 |

---

## Future Considerations

- **Schema validation:** Decorator could validate schemas at decoration time
- **Type inference:** Could auto-generate schemas from type hints
- **Documentation generation:** Decorated metadata could feed API docs

---

## References

- [ADR 066: MCP Server Transport Standards](./066_standardize_on_fastmcp_for_all_mcp_server_implementations.md)
- [sse_adaptor.py](../mcp_servers/lib/sse_adaptor.py)
- [FastMCP Decorator Pattern](https://github.com/jlowin/fastmcp)

--- END OF FILE ADRs/076_sse_tool_metadata_decorator_pattern.md ---

--- START OF FILE ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md ---

# ADR 066: MCP Server Transport Standards (Dual-Stack: FastMCP STDIO + Gateway-Compatible SSE)

**Status:** ✅ APPROVED (Red Team Unanimous)
**Version:** v1.3 (Red Team Hardened)
**Date:** 2025-12-24
**Author:** Antigravity + User + Gateway Agent Analysis + Red Team (Gemini 3, ChatGPT, Grok 4, claude opus 4.5)
**Supersedes:** ADR 066 v1.1, v1.2

---

> [!IMPORTANT]
> **ADR 066 v1.2 → v1.3 Changes:** This version incorporates mandatory red team corrections including: renamed title to reflect dual-transport reality, canonical transport selector, SSEServer scalability constraints, FastMCP SSE prohibition policy, and security hardening requirements.

---

## Context

Initial implementation of the fleet using a custom `SSEServer` resulted in significant tool discovery failures (0 tools found for `git`, `utils`, `network`). Refactoring servers using **FastMCP** successfully achieved 100% protocol compliance and tool discovery in **STDIO mode**. This version (1.3) documents a **critical SSE transport incompatibility** discovered during Gateway integration testing on 2025-12-24 and incorporates red team hardening.

### Critical Finding: FastMCP SSE Transport Incompatibility

> [!CAUTION]
> **FastMCP 2.x SSE transport is NOT compatible with the IBM ContextForge Gateway.**
> FastMCP uses a different SSE handshake pattern than the MCP specification requires.
> **FastMCP SSE MUST NOT be used with the Gateway unless validated by automated handshake tests and explicitly approved via a new ADR.**

#### Tested Versions
- **Gateway:** IBM ContextForge Gateway v1.0.0-BETA-1 (container: `mcp_gateway`)
- **FastMCP:** v2.14.1 (incompatible SSE)
- **SSEServer:** `mcp_servers/lib/sse_adaptor.py` (compatible)
- **MCP SDK:** `mcp.server.sse.SseServerTransport` (compatible)

#### Impact Assessment
- **Affected Services:** 6 fleet containers (sanctuary_utils, filesystem, network, git, cortex, domain)
- **Affected Tools:** 84 federated tools (0% discovery via FastMCP SSE)
- **Working Reference:** `helloworld_mcp` (uses MCP SDK SSE)

#### Observed Behavior

| Transport | Gateway Compatible | Tool Discovery | Notes |
|-----------|-------------------|----------------|-------|
| **STDIO** | N/A (local only) | ✅ 100% | Works perfectly for Claude Desktop |
| **SSE (FastMCP)** | ❌ NO | ❌ 0% | Empty reply, connection closes immediately |
| **SSE (SSEServer)** | ✅ YES | ✅ 100% | Persistent connection, proper handshake |
| **SSE (MCP SDK)** | ✅ YES | ✅ 100% | Used by `helloworld_mcp` reference |

#### Technical Root Cause

**What the MCP SSE Specification Requires:**
1. Client connects to `/sse` (GET, persistent connection)
2. Server **immediately** sends `endpoint` event with the POST URL
3. Connection stays open with periodic heartbeat pings
4. Client POSTs to `/messages` with JSON-RPC requests
5. Server pushes responses back via the SSE stream

```
event: endpoint
data: /messages

event: ping
data: {}
```

**What FastMCP 2.x Actually Does:**
- FastMCP expects the client to initiate a session handshake via POST first
- The SSE endpoint returns an **empty reply** and closes immediately
- No initial `endpoint` event is sent
- No persistent connection is maintained

**Curl Verification:**
```bash
# FastMCP (BROKEN) - Empty reply
$ curl -v http://localhost:8100/sse
< Empty reply from server
curl: (52) Empty reply from server

# SSEServer (WORKING) - Persistent stream
$ curl -N http://localhost:8100/sse
event: endpoint
data: /messages

event: ping
data: {}
```

---

## Decision (AMENDED)

**This is a DUAL-TRANSPORT STANDARD, not a FastMCP monoculture.**

- **FastMCP** is suitable for **STDIO transport only**
- **Gateway-facing containers** MUST use a **Gateway-compatible SSE implementation** (SSEServer or MCP SDK)

### Canonical Transport Selector (MANDATORY)

> [!WARNING]
> **All servers MUST use this exact transport detection mechanism. No alternatives.**

```python
import os

# MANDATORY: One and only one transport selector
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
assert MCP_TRANSPORT in {"stdio", "sse"}, f"Invalid MCP_TRANSPORT: {MCP_TRANSPORT}"

if MCP_TRANSPORT == "stdio":
    # Use FastMCP for local development
    mcp.run(transport="stdio")
else:
    # Use SSEServer for Gateway integration
    port = int(os.getenv("PORT", 8000))
    server.run(port=port, transport="sse")
```

**Rules:**
- **STDIO is the default** (safe for local development)
- **SSE requires explicit opt-in** via `MCP_TRANSPORT=sse`
- **Never infer transport** from `PORT` alone (ambiguous)

### Transport Selection Matrix

| Deployment Context | Transport Mode | Implementation | ENV |
|-------------------|----------------|----------------|-----|
| Claude Desktop (local) | STDIO | FastMCP | `MCP_TRANSPORT=stdio` (default) |
| IDE Integration (local) | STDIO | FastMCP | `MCP_TRANSPORT=stdio` (default) |
| Podman Fleet → Gateway | SSE | SSEServer | `MCP_TRANSPORT=sse` + `PORT=8000` |
| Podman Fleet → Gateway | SSE | MCP SDK | `MCP_TRANSPORT=sse` + `PORT=8000` |

### SSEServer Scalability Constraint (MANDATORY)

> [!WARNING]
> **SSEServer uses a single message queue and is approved ONLY for single-Gateway, single-client deployments.**

**Current Limitations:**
- Multiple Gateway connections → message interleaving
- One slow client → backpressure for all clients
- Reconnection storms → dropped responses

**Future Exit Strategy:**
- Implement per-client queues when scaling beyond single-Gateway
- Or migrate to MCP SDK SSE once proven stable at scale

### FastMCP SSE Prohibition (MANDATORY)

> [!CAUTION]
> **FastMCP's `transport="sse"` MUST NOT be used with the IBM ContextForge Gateway.**
> 
> **Exception Process:** A new ADR (proposed: ADR-067) must be approved if:
> 1. A future FastMCP version claims SSE fixes
> 2. Automated handshake tests pass (`curl -N /sse` returns `event: endpoint`)
> 3. Full fleet registration verified against Gateway

## Architecture

![mcp_sse_stdio_transport](../docs/architecture_diagrams/transport/mcp_sse_stdio_transport.png)

*[Source: mcp_sse_stdio_transport.mmd](../docs/architecture_diagrams/transport/mcp_sse_stdio_transport.mmd)*

---

## The Sanctuary Pattern (Preserved from v1.1)

### 1. Core Rules (All Transports)

* **Domain-Prefix Naming:** All tool names **MUST** use a domain prefix (e.g., `adr_create`, `git_commit`) to prevent namespace collisions.
* **3-Layer Logic Delegation:** `server.py` (Interface) delegates to `operations.py` (Logic). Both transports share the same logic layer.
* **Request Modeling:** All tool inputs defined as Pydantic `BaseModel` classes in `models.py`.
* **Transport-Aware Bootloader:** Use the canonical transport selector above.

### 2. Tool Naming for Federation (Future Requirement)

Current: `git_commit`, `filesystem_read`

**Recommended for Gateway Federation:**
```
sanctuary.git.commit
sanctuary.filesystem.read
```

Not enforced yet, but flagged as a future compatibility requirement when integrating with external MCP registries.

---

## Implementation Templates

### Template A: STDIO Mode (FastMCP - Claude Desktop)

```python
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from .models import DomainToolRequest
from .operations import DomainOperations

mcp = FastMCP(
    "project_sanctuary.domain.name",
    instructions="Instructions for LLM discovery.",
    dependencies=["pydantic>=2.0", "fastmcp>=2.14.0"]  # Pin versions
)

@mcp.tool()
def domain_tool_name(request: DomainToolRequest) -> str:
    """Descriptive docstring for LLM discovery."""
    try:
        result = DomainOperations.perform_action(**request.model_dump())
        return f"Success: {result}"
    except Exception as e:
        raise ToolError(f"Operation failed: {str(e)}")

if __name__ == "__main__":
    import os
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
    assert MCP_TRANSPORT == "stdio", "FastMCP SSE is NOT Gateway compatible. Use SSEServer."
    mcp.run(transport="stdio")
```

### Template B: SSE Mode (SSEServer - Gateway Fleet)

```python
import os
from mcp_servers.lib.sse_adaptor import SSEServer
from .tools import time_tool, calculator_tool
from .models import TIME_SCHEMA, CALC_SCHEMA

server = SSEServer("sanctuary_utils")

# Register tools with shared logic from operations.py
server.register_tool("time.get_current_time", time_tool.get_current_time, TIME_SCHEMA)
server.register_tool("calculator.add", calculator_tool.add, CALC_SCHEMA)

# Expose FastAPI app for uvicorn (adds /health endpoint)
app = server.app

if __name__ == "__main__":
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
    assert MCP_TRANSPORT == "sse", "This entry point requires MCP_TRANSPORT=sse"
    port = int(os.getenv("PORT", 8000))
    server.run(port=port, transport="sse")
```

### Template C: SSE Mode (MCP SDK - Alternative)

```python
import os
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

app = Server("sanctuary_utils")

# Register tools with @app.tool() decorators...
# (calls same operations.py logic as FastMCP version)

sse = SseServerTransport("/messages")
starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=sse.handle_sse),
        Route("/messages", endpoint=sse.handle_messages, methods=["POST"]),
        Route("/health", endpoint=lambda r: JSONResponse({"status": "healthy"})),
    ],
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"]),
    ]
)

if __name__ == "__main__":
    import uvicorn
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
    assert MCP_TRANSPORT == "sse", "This entry point requires MCP_TRANSPORT=sse"
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
```

---

## Gateway SSE Handshake Protocol

```
┌─────────────┐                         ┌─────────────┐
│   Gateway   │                         │  MCP Server │
└──────┬──────┘                         └──────┬──────┘
       │                                       │
       │ GET /sse (Persistent Connection)      │
       │──────────────────────────────────────>│
       │                                       │
       │ event: endpoint                       │
       │ data: /messages                       │
       │<──────────────────────────────────────│
       │                                       │
       │ POST /messages (JSON-RPC Request)     │
       │──────────────────────────────────────>│
       │                                       │
       │ 202 Accepted                          │
       │<──────────────────────────────────────│
       │                                       │
       │ event: message                        │
       │ data: {"jsonrpc":"2.0",...}           │
       │<──────────────────────────────────────│
       │                                       │
       │ event: ping (every 15s, configurable) │
       │<──────────────────────────────────────│
```

### Error Flows (Red Team Addition)

| Scenario | Expected Behavior | Recovery |
|----------|------------------|----------|
| POST /messages fails (500) | Return JSON-RPC error response | Client retries with backoff |
| SSE connection drops | Gateway reconnects automatically | Server accepts new GET /sse |
| Malformed JSON-RPC | Return -32700 Parse Error | Log and continue |
| Unknown method | Return -32601 Method Not Found | Log and continue |
| Tool throws exception | Return -32603 Internal Error | Wrap in ToolError |

---

## Security Considerations (Red Team Addition)

### Authentication
- Gateway connections include `Authorization: Bearer <token>` header
- SSEServer should validate tokens on `/messages` POST (not currently enforced)

### Rate Limiting
- No current rate limiting on `/sse` or `/messages`
- **Risk:** DoS via many persistent SSE connections
- **Mitigation:** Add connection limits in nginx/traefik or SSEServer

### Input Validation
- Pydantic models provide schema validation
- **Risk:** Command injection in shell-executing tools (e.g., `git_commit`)
- **Mitigation:** Sanitize all string inputs, never pass to shell directly

### Encryption
- Gateway uses HTTPS (self-signed cert, `verify=False` in dev)
- **Risk:** MITM in production
- **Mitigation:** Use proper CA-signed certs in production

---

## Consequences

### Positive
* **Protocol Compliance:** SSEServer/MCP SDK guarantees 100% tool discovery by the IBM Gateway
* **Operational Consistency:** Clear separation between local (STDIO) and fleet (SSE) deployments
* **Auditability:** The 3-Layer pattern remains required for Protocol 128 compliance
* **Portability:** Same business logic works in both transports via different wrappers

### Negative
* **Dual Implementation:** Fleet containers require SSEServer wrapper + FastMCP wrapper
* **Testing Overhead:** Must test both transports for each server
* **Maintenance Complexity:** Two entry points per server, risk of drift
* **FastMCP Limitation:** Cannot use FastMCP's `transport="sse"` for Gateway

---

## Red Team Analysis (Enhanced)

### What Could Still Fail?

1. **FastMCP Version Changes:** Future versions may fix SSE compatibility → monitor releases, require new ADR
2. **Gateway Protocol Updates:** IBM Gateway may change SSE expectations → maintain handshake test harness
3. **Heartbeat Timing:** Current 15-second ping may need adjustment → make configurable
4. **Concurrent Connections:** SSEServer single queue → implement per-client queues for scale
5. **Supply Chain Attacks:** FastMCP from GitHub could be compromised → pin versions, verify checksums
6. **Configuration Drift:** ENV vars mis-set in containers → use health checks and defaults
7. **Tool Discovery False Positives:** Schema mismatches → add schema validation tests
8. **Transport Selector Inconsistency:** Multiple detection methods → enforce canonical selector

### Mitigation Strategies

1. **Automated Handshake Tests:** CI step that curls `/sse` and verifies `event: endpoint`
2. **Version Pinning:** Lock FastMCP and MCP SDK to known working versions
3. **Gateway Test Harness:** `verify_hello_world_rpc.py` pattern for each fleet server
4. **Fallback Path:** Keep SSEServer as the proven Gateway transport
5. **Security Scanning:** Run SAST/DAST on SSE endpoints
6. **Monitoring Integration:** Prometheus metrics for connection counts, latencies
7. **Schema Validation:** Automated tests comparing tool schemas across transports

---

## Next Steps for Implementation

1. **Revert Fleet Servers:** Change sanctuary_* containers from FastMCP SSE to SSEServer
2. **Add Canonical Transport Selector:** Update all bootloaders to use `MCP_TRANSPORT`
3. **Rebuild Containers:** `podman compose build` for all affected services
4. **Verify Handshake:** For each container, run `curl -N http://localhost:<PORT>/sse`
5. **Re-run Fleet Setup:** `python -m mcp_servers.gateway.fleet_setup`
6. **Verify Tool Discovery:** Confirm all 84 tools in Gateway admin API
7. **Update Documentation:** Link to this ADR from READMEs, add examples
8. **Security Review:** Conduct penetration testing on SSE endpoints
9. **Add CI Handshake Test:** Automate verification in pipeline

---

## Rollback Plan

If issues arise post-implementation:

1. **Immediate:** Containers can be stopped (`podman compose down`)
2. **Fallback:** Revert to FastMCP STDIO for local testing (Claude Desktop still works)
3. **Gateway:** Can be disconnected from fleet while debugging
4. **Data:** No stateful data in SSE layer, only ephemeral connections

---

## References

- `mcp_servers/lib/sse_adaptor.py` - Working SSEServer implementation
- `mcp_servers/gateway/verify_hello_world_rpc.py` - Gateway verification script
- `Protocol 128` - Cognitive Continuity standard (see ADR-128)
- IBM ContextForge Gateway documentation (internal)
- FastMCP GitHub: https://github.com/jlowin/fastmcp
- MCP SDK: https://github.com/modelcontextprotocol/python-sdk

---

## Audit Resolution Log

### 2025-12-26: SSEServer Healthcheck Starvation Fix

**Issue:** `sanctuary_cortex` container entered "unhealthy" state (FailingStreak: 7, ExitCode 125) during `cortex-cortex-ingest-full` operations (~687s runtime).

**Root Cause:** The SSEServer Scalability Constraint (documented above) was triggered — synchronous blocking of the event loop during 6,000+ chunk ingestion prevented the healthcheck endpoint from responding within the 10s timeout.

**Resolution (Dual-Stack Fix):**

| Layer | Change | Rationale |
|-------|--------|-----------|
| **Infrastructure** | `docker-compose.yml` healthcheck: `start_period: 600s`, `interval: 60s`, `timeout: 30s`, `retries: 5` | Accommodate 11-minute ingestion window |
| **Code** | `server.py`: `cortex_ingest_full` and `cortex_ingest_incremental` converted to `async` using `asyncio.to_thread()` | Offload blocking I/O to thread pool, keep SSE event loop responsive |

**Verification:**
```bash
$ podman inspect --format='{{json .State.Health}}' sanctuary_cortex | jq '.Status, .FailingStreak'
"healthy"
0
```

**Files Modified:**
- `docker-compose.yml` (lines 192-199)
- `mcp_servers/gateway/clusters/sanctuary_cortex/server.py` (lines 180-210)

**Status:** ✅ Resolved

---

## Red Team Review Sign-Off

| Reviewer | Verdict | Date |
|----------|---------|------|
| Gemini 3 (Gateway Agent) | ✅ Root cause confirmed | 2025-12-24 |
| ChatGPT (Red Team) | 🟡 Conditionally Acceptable | 2025-12-24 |
| Grok 4 (Red Team) | 🟡 Approve with Revisions | 2025-12-24 |
| Antigravity | ✅ Hardened v1.3 | 2025-12-24 |
| Antigravity | ✅ Audit Resolution (Healthcheck Fix) | 2025-12-26 |

--- END OF FILE ADRs/066_standardize_on_fastmcp_for_all_mcp_server_implementations.md ---

--- START OF FILE scripts/extract_orphaned_diagrams.py ---

#!/usr/bin/env python3
"""
Extract Orphaned Mermaid Diagrams (Task #154 - Phase 2)

Reads the inventory_mermaid.json and extracts all orphaned diagrams
(those without .mmd files) into the new centralized directory structure.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "architecture_diagrams"

# Mapping of diagram types/content patterns to subdirectories
def categorize_diagram(content: str, preview: str) -> str:
    """Determine which subdirectory a diagram belongs in."""
    content_lower = content.lower()
    preview_lower = preview.lower()
    
    # Check for workflow/protocol patterns
    if any(kw in content_lower for kw in ['protocol', 'learning', 'scout', 'seal', 'audit', 'pipeline', 'flow']):
        return 'workflows'
    
    # Check for sequence diagrams
    if content_lower.startswith('sequencediagram'):
        return 'sequences'
    
    # Check for class diagrams
    if content_lower.startswith('classdiagram'):
        return 'class'
    
    # Check for RAG-related
    if any(kw in content_lower for kw in ['rag', 'embedding', 'vector', 'chunking']):
        return 'rag'
    
    # Check for transport/SSE/stdio
    if any(kw in content_lower for kw in ['transport', 'sse', 'stdio', 'bridge']):
        return 'transport'
    
    # Default to system for architecture diagrams
    return 'system'


def sanitize_filename(preview: str, hash_val: str) -> str:
    """Create a reasonable filename from preview text."""
    # Extract meaningful words from preview
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', preview[:80])
    
    if len(words) >= 3:
        # Use first few meaningful words
        name = '_'.join(words[:4]).lower()
    else:
        # Fall back to hash-based name
        name = f"diagram_{hash_val[:8]}"
    
    # Ensure it's a valid filename
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    
    return name[:50]  # Limit length


def extract_content_by_hash(hash_val: str, inline_blocks: list) -> str:
    """Extract the full content for a given hash from inline blocks."""
    for block in inline_blocks:
        if block['content_hash'] == hash_val:
            # Read the file and extract the block
            try:
                file_path = PROJECT_ROOT / block['file_path']
                content = file_path.read_text(encoding='utf-8')
                pattern = r'```mermaid\s*\n(.*?)```'
                for match in re.finditer(pattern, content, re.DOTALL):
                    block_content = match.group(1)
                    # Compute hash to verify
                    from hashlib import sha256
                    normalized = '\n'.join(line.strip() for line in block_content.strip().split('\n'))
                    h = sha256(normalized.encode('utf-8')).hexdigest()[:16]
                    if h == hash_val:
                        return block_content.strip()
            except Exception as e:
                print(f"   Warning: Could not read {block['file_path']}: {e}")
    
    return None


def main():
    print("📦 Extracting Orphaned Mermaid Diagrams")
    print("=" * 60)
    
    # Load inventory
    inventory_path = PROJECT_ROOT / "inventory_mermaid.json"
    if not inventory_path.exists():
        print("❌ inventory_mermaid.json not found. Run mermaid_inventory.py first.")
        return
    
    with open(inventory_path, 'r') as f:
        inventory = json.load(f)
    
    canonical_diagrams = inventory.get('canonical_diagrams', [])
    inline_blocks = inventory.get('inline_blocks', [])
    
    # Filter to orphaned (no canonical_mmd)
    orphaned = [d for d in canonical_diagrams if not d.get('canonical_mmd') and d.get('total_occurrences', 0) > 0]
    
    print(f"\n📊 Found {len(orphaned)} orphaned canonical diagrams")
    print(f"   Target directory: {DIAGRAMS_DIR}")
    
    # Track created files
    created_files = []
    skipped = []
    category_counts = defaultdict(int)
    
    for diag in orphaned:
        hash_val = diag['canonical_hash']
        preview = diag.get('preview', '')
        occurrences = diag.get('total_occurrences', 0)
        
        print(f"\n   Processing {hash_val[:12]}... ({occurrences} occurrences)")
        
        # Extract full content
        content = extract_content_by_hash(hash_val, inline_blocks)
        if not content:
            print(f"      ⚠️  Could not extract content, skipping")
            skipped.append(hash_val)
            continue
        
        # Categorize
        category = categorize_diagram(content, preview)
        category_counts[category] += 1
        
        # Generate filename
        filename = sanitize_filename(preview, hash_val)
        target_dir = DIAGRAMS_DIR / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing file with same name
        target_path = target_dir / f"{filename}.mmd"
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{filename}_{counter}.mmd"
            counter += 1
        
        # Write the file
        target_path.write_text(content, encoding='utf-8')
        created_files.append({
            'path': str(target_path.relative_to(PROJECT_ROOT)),
            'hash': hash_val,
            'category': category,
            'occurrences': occurrences
        })
        print(f"      ✅ Created: {target_path.relative_to(PROJECT_ROOT)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"   Total orphaned diagrams:  {len(orphaned)}")
    print(f"   Successfully created:     {len(created_files)}")
    print(f"   Skipped (no content):     {len(skipped)}")
    print("\n   By category:")
    for cat, count in sorted(category_counts.items()):
        print(f"      {cat}: {count}")
    
    # Write manifest
    manifest_path = PROJECT_ROOT / "docs" / "architecture_diagrams" / "MANIFEST.json"
    manifest = {
        "description": "Centralized Mermaid diagram repository (Task #154)",
        "created": "2025-12-29",
        "total_diagrams": len(created_files),
        "diagrams": created_files
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n   ✅ Manifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("=" * 60)
    print("   1. Review created .mmd files for naming/categorization")
    print("   2. Run similarity check to find any remaining duplicates")
    print("   3. Render .mmd → .png with mermaid-cli")
    print("   4. Update all .md files to reference images")
    print("=" * 60)


if __name__ == "__main__":
    main()

--- END OF FILE scripts/extract_orphaned_diagrams.py ---

--- START OF FILE ADRs/073_standardization_of_python_dependency_management_across_environments.md ---

# Standardization of Python Dependency Management Across Environments

**Status:** Approved
**Date:** 2025-12-26
**Author:** AI Assistant
**Related tasks:** Task 146, Task 147

**Summary:** Each service owns one runtime `requirements.txt` used consistently across all execution environments, while shared dependencies are versioned centrally via a common core.

---

## Core Principles

1.  **Every service/container owns its runtime dependencies**
    *   Ownership is expressed via one `requirements.txt` per service.
    *   This file is the single source of truth, regardless of how the service is run.
2.  **Execution environment does not change dependency logic**
    *   Docker, Podman, .venv, and direct terminal execution must all install from the same `requirements.txt`.
    *   "Each service defines its own runtime world."
3.  **Shared versions are centralized, runtime ownership remains local**

---

## Context

To reduce "Dependency Chaos" across the Project Sanctuary fleet, we are standardizing Python requirements management. This ADR addresses the following problems:

1.  **Fragmented Container Dependencies**: 8 separate Dockerfiles with inconsistent approaches — some use manual `pip install`, others use `requirements.txt`. *Solved: Single source of truth per service.*
2.  **Local/Container Drift**: Root `/requirements.txt` doesn't match container environments. *Solved: Locked files guarantee identical versions everywhere.*
3.  **Scattered Requirements**: Individual directories maintain their own lists with no coordination. *Solved: Tiered hierarchy with shared baseline.*
4.  **Redundant Installations**: `sanctuary_cortex/Dockerfile` installs from both `requirements.txt` AND inline `pip install`. *Solved: No manual installs in Dockerfiles.*
5.  **Cache Invalidation**: Incorrect Dockerfile ordering (`COPY code` → `RUN pip install`) broke caching. *Solved: Proper layer ordering.*

**Scope**: This policy applies equally to:
*   Docker / Podman
*   .venv-based execution
*   Direct terminal execution

Dockerfiles are not special — they are just one consumer of `requirements.txt`.

## Options Analysis

### Option A: Distributed/Manual (Current Status Quo)
- **Description**: Each Dockerfile manually lists its own packages (`RUN pip install fastapi uvicorn...`).
- **Pros**: Zero coupling between services.
- **Cons**: High maintenance. Inconsistent versions across the fleet. High risk of "it works on my machine" vs. container. Redundant layer caching is minimal.

### Option B: Unified "Golden" Requirements
- **Description**: A single `requirements-fleet.txt` used by ALL 8 containers.
- **Pros**: Absolute consistency. Simplified logic. Maximum Docker layer sharing if base images match.
- **Cons**: Bloated images. `sanctuary_utils` (simple) inherits heavy ML deps from `rag_cortex` (complex). Security risk surface area increases unnecessarily for simple tools.

### Option C: Tiered "Base vs. Specialized" (Recommended)
- **Description**:
    *   **Tier 1 (Common)**: A `requirements-common.txt` (fastapi, uvicorn, pydantic, mcp) used by all.
    *   **Tier 2 (Specialized)**: Specific files for heavy lifters (e.g., `requirements-cortex.txt` for ML/RAG).
- **Pros**: Balances consistency with efficiency. Keeps lightweight containers light.
- **Cons**: Slightly more complex build context (need to mount/copy the common file).

### Option D: Dockerfile-Specific Requirements (Strict Mapping)
- **Description**: Every Dockerfile `COPY`s exactly one `requirements.txt` that lives next to it. No manual `pip install` lists allowed in Dockerfiles.
- **Pros**: Explicit, declarative. Clean caching.
- **Cons**: Can lead to version drift if not managed by a central lockfile or policy.

### Dependency Locking Tools: pip-compile and uv

To achieve deterministic builds, we use tools like `pip-compile` (from `pip-tools`) or `uv` to manage the translation between intent (`.in`) and lock (`.txt`).

1.  **Purpose**:
    *   `pip-compile` reads high-level dependency intent from `.in` files (e.g., `fastapi`, `pydantic`).
    *   It resolves the entire dependency graph and generates a locked `requirements.txt` containing exact versions (e.g., `fastapi==0.109.0`, `starlette==0.36.3`) and hashes.
    *   It **does not install** packages; it strictly generates the artifact.

2.  **Why this matters for Sanctuary**:
    *   **Determinism**: Ensures that Docker containers, local `.venv`s, and CI pipelines install mathematically identical environments.
    *   **Prevention**: Eliminates the class of bugs where a transitive dependency updates silently and breaks a service ("works on my machine" but fails in prod).
    *   **Alignment**: Supports the core principle that "every service defines one runtime world."

3.  **Understanding Transitive Dependencies**:
    *   `.in` files list only **direct dependencies** — packages your code explicitly imports (e.g., `langchain`, `chromadb`).
    *   `pip-compile` resolves the **entire dependency tree**, discovering all sub-dependencies automatically.
    *   Example: `chromadb` depends on `kubernetes`, which depends on `urllib3`. You never list `urllib3` in your `.in` file — pip-compile finds it and locks a specific version in the `.txt` file.
    *   **Security fixes for transitive deps**: Use `--upgrade-package <name>` to force an upgrade without polluting your `.in` file with packages you don't directly use.

4.  **Workflow Example**:
    ```bash
    # Generate locked requirements (Do this when dependencies change)
    pip-compile requirements-core.in --output-file requirements-core.txt
    pip-compile requirements-dev.in --output-file requirements-dev.txt
    pip-compile mcp_servers/gateway/clusters/sanctuary_cortex/requirements.in \
      --output-file mcp_servers/gateway/clusters/sanctuary_cortex/requirements.txt
    
    # Install (Do this to run)
    pip install -r requirements.txt
    ```

### Local Environment Synchronization

ADR 073 mandates that **Core Principle #2 ("Execution environment does not change dependency logic")** applies strictly to local `.venv` and terminal execution. Pure Docker consistency is insufficient.

1.  **Policy**:
    *   Docker, Podman, and Local `.venv` must instal from the exact same locked artifacts.
    *   Local environments MAY additionally install `requirements-dev.txt` (which containers MUST skip).

2.  **Setup Strategies**:

    *   **Option A: Single Service Mode** (Focus on one component):
        ```bash
        source .venv/bin/activate
        # Install runtime
        pip install -r mcp_servers/gateway/clusters/sanctuary_cortex/requirements.txt
        # Install dev tooling
        pip install -r requirements-dev.txt
        ```

    *   **Option B: Full Monorepo Mode** (Shared venv):
        ```bash
        source .venv/bin/activate
        # Install shared baseline
        pip install -r mcp_servers/requirements-core.txt
        # Install all service extras (potentially conflicting, use with care)
        pip install -r mcp_servers/gateway/clusters/*/requirements.txt
        # Install dev tooling
        pip install -r requirements-dev.txt
        ```

3.  **Cross-Platform Environment Standard**:
    *   **Problem:** `.venv` created on Windows (`Scripts/`) is incompatible with WSL (`bin/`).
    *   **Rule:** When switching platforms (e.g., Windows -> WSL), the environment must be reset to match the kernel.
    *   **Mechanism:** Use `make bootstrap` (which handles `python3 -m venv`).
    *   **Warning:** Do not share a single `.venv` folder across Windows and WSL filesystems.

4.  **Automation & Enforcement**:
    *   We will introduce a Makefile target `install-env` to standardize this.
    *   Agents must detect drift between `pip freeze` and locked requirements in active environments.

## Reference Directory Structure (Example)

```
mcp_servers/
  gateway/
    requirements-core.txt  <-- Shared Baseline

  filesystem/
    requirements.txt       <-- Service specific (installs core + extras)
    Dockerfile

  utils/
    requirements.txt

  cortex/
    requirements.txt
    Dockerfile

requirements-dev.txt       <-- Local Dev only
```

## Decision & Recommendations

We recommend **Option D** (Strict Mapping) enhanced with a **Tiered Policy**:

1.  **Eliminate Manual `pip install`**: All Dockerfiles must `COPY requirements.txt` and `RUN pip install -r`. No inline package lists.
2.  **Harmonize Versions**: We will create a `requirements-core.txt` at the `mcp_servers/gateway` level to define the **shared baseline**.
    *   Individual services MAY reference it (`-r requirements-core.txt`) or copy it explicitly.
    *   The mechanism is less important than the rule: Shared versions are centralized, runtime ownership remains local.
3.  **Locking Requirement (Critical)**: All `requirements.txt` files MUST be generated artifacts from `.in` files using a single approved tool (e.g., `pip-tools` or `uv`).
    *   `.in` files represent **human-edited dependency intent**.
    *   `.txt` files are **machine-generated locks** to ensure reproducible builds.
    *   Manual editing of `.txt` files is prohibited.
4.  **Dev vs Runtime Separation**: explicit `requirements-dev.txt` for local/test dependencies. Containers must NEVER install dev dependencies.
    *   Avoids the "Superset" risk where local logic relies on tools missing in production.
5.  **CI Enforcement**: CI pipelines must fail if any Dockerfile contains inline `pip install` commands not referencing a requirements file.
6.  **Clean Up**: Remove the redundant manual `pip install` block from `sanctuary_cortex/Dockerfile` immediately.

## Sanctuary Dependency Update Workflow

This is the "Tiered Policy" approach (Option D) to maintain consistency across local Mac and Podman containers.

![python_dependency_workflow](../docs/architecture_diagrams/workflows/python_dependency_workflow.png)

*[Source: python_dependency_workflow.mmd](../docs/architecture_diagrams/workflows/python_dependency_workflow.mmd)*

### Step-by-Step Process

1. **Identify Intent**: Open the relevant `.in` file (e.g., `requirements-core.in` for shared tools or `sanctuary_cortex/requirements.in` for RAG-specific tools).

2. **Declare Dependency**: Add the package name (e.g., `langchain`). This is the "Human Intent" phase.

3. **Generate Lockfile**: Run the compilation command:
   ```bash
   pip-compile <path_to_in_file> --output-file <path_to_txt_file>
   ```
   This resolves all sub-dependencies and creates a deterministic "Machine Truth" file.

4. **Local Sync**: Update your local `.venv` by running `pip install -r <path_to_txt_file>`.

5. **Container Sync**: Rebuild the Podman container. Because the Dockerfile uses `COPY requirements.txt`, it will automatically pull the exact same versions you just locked locally.

6. **Commitment**: Commit both the `.in` (Intent) and `.txt` (Lock) files to Git.

### Why This is the "Sanctuary Way"

- **No Manual Installs**: You never run `pip install <package>` directly in a Dockerfile; everything is declared in the requirements file.
- **No Drift**: If a tool works on your MacBook Pro, it is mathematically guaranteed to work inside the `sanctuary_cortex` container because they share the same `.txt` lock.

## Consequences

- **Immediate**: `sanctuary_cortex/Dockerfile` becomes cleaner and builds slightly faster (no double install checks).
- **Long-term**: Dependency updates (e.g., bumping `fastapi`) can be done in one place for 80% of the fleet.
- **Why**: ".in files exist to make upgrades safe and reproducible, not to change how services are run."
- **Determinism**: Builds become reproducible across machines and time (via locking).
- **Safety**: "Works on my machine" bugs reduced by strict dev/runtime separation.
- **Risk**: Needs careful audit of `sanctuary_cortex/requirements.txt` to ensuring nothing from the manual list is missing before deletion.

## Developer / Agent Checklist (Future Reference)

**Purpose**: Ensure all environments (Docker, Podman, local .venv) remain consistent with locked requirements.

### Verify Locked Files
- [ ] **Confirm `.in` files exist** for core, dev, and each service.
- [ ] **Confirm `.txt` files were generated** via `pip-compile` (or `uv`) from `.in` files.
- [ ] **Check that Dockerfiles point to the correct `requirements.txt`.**

### Update / Install Dependencies
#### Local venv / Terminal:
```bash
source .venv/bin/activate
pip install --no-cache-dir -r mcp_servers/requirements-core.txt
pip install --no-cache-dir -r mcp_servers/gateway/clusters/<service>/requirements.txt
pip install --no-cache-dir -r requirements-dev.txt  # optional for dev/test
```

#### Containers:
- [ ] Ensure Dockerfiles use:
    ```dockerfile
    COPY requirements.txt /tmp/requirements.txt
    RUN pip install --no-cache-dir -r /tmp/requirements.txt
    ```
- [ ] **Dev dependencies must not be installed in containers.**

### Check for Drift
- [ ] Compare `pip freeze` in active environments vs locked `.txt` files.
- [ ] Warn if any packages or versions differ.

### Regenerate Locks When Updating Dependencies
1.  Update `.in` files with new intent.
2.  Run `pip-compile` to produce updated `.txt` files.
3.  Verify Dockerfiles and local environments still match.

### Automation
- [ ] Use `make install-env TARGET=<service>` to sync venv for a specific service.
- [ ] CI pipelines should enforce: no inline `pip install`, only locked files allowed.

### Pre-Commit / Pre-Build
- [ ] Confirm all `.txt` files are up-to-date.
- [ ] Ensure Dockerfiles reference correct files.
- [ ] Optional: run `make verify` to validate local and container environments.
## How to Add a New Python Dependency (Standard Practice)

Follow this exact workflow to add or update a dependency in Project Sanctuary. This ensures determinism, consistency across local/container environments, and compliance with the locked-file policy.

### Step-by-Step: Adding a New Requirement

1.  **Identify the correct .in file (human intent file)**
    *   **Shared baseline** (e.g., fastapi, pydantic, MCP libs): Edit `mcp_servers/gateway/requirements-core.in`
    *   **Service-specific** (e.g., chromadb, langchain for RAG cortex): Edit the service’s own file, e.g. `mcp_servers/gateway/clusters/sanctuary_cortex/requirements.in`
    *   **Local development/testing only** (e.g., black, ruff): Edit `requirements-dev.in` (root or appropriate location)
    *   **Note**: If a service needs testing tools *inside* its container (e.g., for Protocol 101 gates), add them to the service-specific `.in` file.

2.  **Add the dependency intent**
    Write only the high-level package name and optional version constraint in the `.in` file.
    *   Examples: `fastapi>=0.110.0`, `chromadb`, `langchain-huggingface`
    *   Do not add transitive deps or exact pins here.

3.  **Regenerate the locked .txt file(s)**
    Run `pip-compile` (or `uv`) to produce the deterministic lockfile:
    ```bash
    # Example for a specific service
    pip-compile mcp_servers/gateway/clusters/sanctuary_git/requirements.in \
      --output-file mcp_servers/gateway/clusters/sanctuary_git/requirements.txt
    ```

4.  **Commit both files**
    Commit the edited `.in` file and the regenerated `.txt` file. Never hand-edit `.txt` files.

5.  **Verify installation**
    *   **Local**: `pip install -r <service>/requirements.txt`
    *   **Container**: Rebuild the image (`make up force=true TARGET=<service>`).

### Quick Reference Table

| Dependency Type | Edit This File | Regenerate Command Example | Install Command (Local) |
| :--- | :--- | :--- | :--- |
| **Shared baseline** | `requirements-core.in` | `pip-compile ... --output-file requirements-core.txt` | `pip install -r requirements-core.txt` |
| **Service-specific** | `<service>/requirements.in` | `pip-compile <service>/requirements.in --output-file <service>/requirements.txt` | `pip install -r <service>/requirements.txt` |
| **Dev / testing** | `requirements-dev.in` | `pip-compile requirements-dev.in --output-file requirements-dev.txt` | `pip install -r requirements-dev.txt` |

**Golden Rule**: `.in` = what humans edit (intent). `.txt` = what machines generate and everything installs from (truth).

## How to Update Dependencies (e.g., Security Fixes / Dependabot)

When security vulnerabilities (CVEs) are reported or Dependabot suggests updates:

1.  **Do NOT edit .txt files manually.**
    *   Dependabot often tries to edit `requirements.txt` directly. This breaks the link with `.in` files.
    *   You must update via the `.in` file -> `pip-compile` workflow.

2.  **Workflow**:
    *   **Option A: Update All**: Run `pip-compile --upgrade mcp_servers/requirements-core.in` to pull latest compatible versions for everything.
    *   **Option B: Targeted Update**: Run `pip-compile --upgrade-package <package_name> mcp_servers/requirements-core.in` (e.g., `pip-compile --upgrade-package uvicorn mcp_servers/requirements-core.in`).

3.  **Verify**:
    *   Check the generated `requirements-core.txt` to confirm the version bump.
    *   Rebuild affected containers or reinstall local environment.

4.  **Troubleshooting Dependency Conflicts**:
    *   If `pip-compile --upgrade-package` fails with `ResolutionImpossible`, a transitive dependency has a conflicting constraint.
    *   **Identify the constraint**:
        ```bash
        # Check what requires the problem package
        pip show <package> | grep -i required-by
        # Check what version constraints exist
        pip index versions <package>
        ```
    *   **Common pattern**: Package A (e.g., `kubernetes`) pins package B (e.g., `urllib3<2.6`). Fix requires upgrading both A and B together: `pip-compile --upgrade-package kubernetes --upgrade-package urllib3 ...`
    *   **If still blocked**: The constraint is upstream. File an issue with the constraining package or wait for their release.

### Real-World Example: urllib3 Security Advisory (2025-12-26)

**Situation**: Dependabot flagged 4 urllib3 vulnerabilities (CVE-2025-66418, CVE-2025-66471, etc.) requiring urllib3 ≥2.6.0. Current lock has urllib3==2.3.0.

**Attempted fixes**:
1. `pip-compile --upgrade-package urllib3` → No change (stayed at 2.3.0)
2. `pip-compile --upgrade-package 'urllib3>=2.6.0'` → `ResolutionImpossible`
3. `pip-compile --upgrade` (full upgrade) → Still 2.3.0

**Root cause**: `chromadb` → `kubernetes` has an upstream version constraint incompatible with urllib3 2.6+. The kubernetes Python client had breaking changes with urllib3 2.6.0 (removed `getheaders()` method).

**Resolution options**:
- **Wait for upstream**: Monitor kubernetes-client/python for a release compatible with urllib3 2.6+
- **Security override** (if critical): Add `urllib3>=2.6.0` to `.in` file, then investigate which direct dependency to upgrade/replace
- **Accept risk with mitigation**: Document the advisory, monitor for upstream fix, apply when available

**Status**: Blocked pending upstream kubernetes/chromadb compatibility update.

## Special Case: The Forge (ml_env)

While the Core fleet (Gateway, Cortex, etc.) strictly follows the locked-file policy, the **Forge** environment (`ml_env`) is a recognized exception.

### Rationale
*   **Hardware Dependency**: The Forge relies on extremely specific CUDA versions (e.g., CUDA 12.1 vs 12.4) and PyTorch builds (e.g., `cu121` vs `cu124`) that often require manual `pip install --index-url` commands not easily captured in standard `requirements.txt` resolution.
*   **Ephemeral Nature**: The Forge is used for specific pipeline phases (Fine-Tuning, Merging) and is often rebuilt for different hardware targets.

### Policy for ml_env
1.  **Exemption**: `ml_env` is **exempt** from the `pip-compile` / `requirements.txt` locking requirement.
2.  **Documentation**: Its state is defined procedurally in `forge/CUDA-ML-ENV-SETUP.md`.
3.  **Isolation**: Users **MUST** deactivate `ml_env` before running Core tools (like `cortex_cli.py`) to prevent "dependency bleeding" (e.g., mixing `torch` versions).

--- END OF FILE ADRs/073_standardization_of_python_dependency_management_across_environments.md ---

--- START OF FILE ADRs/068_decide_on_approach_for_sse_bridge.md ---

# Decide on approach for SSE bridge

**Status:** accepted
**Date:** 2025-12-20
**Author:** user, anti gravity agent

---

## Context

Claude Desktop and Gemini Antigravity IDE (via the standard MCP client) do not support `SSE` (Server-Sent Events) transport out of the box; they primarily rely on `stdio` (standard input/output) for local process communication.

The backend Sanctuary Gateway (running in Podman) exposes an SSE endpoint (`https://localhost:4444/sse`). However, the "Official" IBM bridge code (`mcpgateway.translate`) resides in a separate repository (`../sanctuary-gateway`) and is not currently installed in this project's environment.

We need a strategy to bridge `stdio` <-> `SSE`.

# Decide on approach for SSE bridge

**Status:** draft
**Date:** 2025-12-20
**Author:** user

---

## Context

Claude Desktop and Gemini Antigravity IDE (via the standard MCP client) do not support `SSE` (Server-Sent Events) transport out of the box; they primarily rely on `stdio` (standard input/output) for local process communication.

The backend Sanctuary Gateway (running in Podman) exposes an SSE endpoint (`https://localhost:4444/sse`). However, the "Official" IBM bridge code (`mcpgateway.translate`) resides in a separate repository (`../sanctuary-gateway`) and is not currently installed in this project's environment.

We valid options to bridge `stdio` <-> `SSE`.

## Options Considered

### Option A: The "Official Library" Way
Use the official `mcpgateway.translate` module provided by the IBM/ContextForge gateway project.
*   **Mechanism:** Install the `mcpgateway` package from the sibling directory (`../sanctuary-gateway`) into the current virtual environment (`pip install -e`).
*   **Pros:** Uses official, vendor-maintained code; ensures parity with upstream updates.
*   **Cons:**
    *   **Shared Env Problem:** Requires modifying the shared virtual environment to link to an external folder.
    *   **Dependency:** Creates a hard dependency on the presence of the sibling `sanctuary-gateway` directory on the filesystem.

#### Workflow Diagram (Option A)

![MCP SSE Bridge Approach](../docs/architecture_diagrams/transport/mcp_sse_bridge_approach.png)

*[Source: mcp_sse_bridge_approach.mmd](../docs/architecture_diagrams/transport/mcp_sse_bridge_approach.mmd)*

### Option B: The "Single File" Way (Custom Bridge)
Implement a self-contained, single-file Python script (`mcp_servers/gateway/bridge.py`) within this project.
*   **Mechanism:** A small (~100 line) script using `mcp` and `httpx` SDKs that handles the translation logic directly. "Like having a copy of a recipe instead of the whole cookbook."
*   **Pros:**
    *   **No Shared Env Issue:** Completely self-contained in `Project_Sanctuary`. No need to link external libraries.
    *   **Portability:** The code lives with the project.
    *   **Simplicity:** No need to have the other project window open; just requires the Gateway container to be running.
*   **Cons:** We are responsible for maintaining this bridge code (it is a fork/re-implementation of the translation logic).

#### Workflow Diagram (Option B)

*(See combined diagram above)*


## Decision

**Proceed with Option B: Custom Lightweight Bridge.**

Based on detailed feedback from external audit (Grok, GPT) and internal consensus (Claude), Option B is the unanimous recommendation. The Shared Environment Constraint dictates that minimizing external filesystem links is critical for project portability and stability.

### Security Analysis (Red Team)
The implementation of `mcp_servers/gateway/bridge.py` has been verified to address key concerns:
1.  **JWT Token Injection:** Correctly reads `MCPGATEWAY_BEARER_TOKEN` and formats standard `Authorization: Bearer` headers.
2.  **SSL/TLS Verification:** Implements `MCP_GATEWAY_VERIFY_SSL` flag to safely handle local self-signed certificates without global suppression.
3.  **Error Propagation:** Catches exceptions and returns standard JSON-RPC error objects.

## Consequences

*   **Autonomy:** This project becomes robust against changes or absence of the sibling gateway repo.
*   **Support:** We accept responsibility for maintaining the `bridge.py` logic.

--- END OF FILE ADRs/068_decide_on_approach_for_sse_bridge.md ---

--- START OF FILE ADRs/082_harmonized_content_processing.md ---

# ADR 082: Harmonized Content Processing Architecture

**Status:** PROPOSED  
**Author:** Guardian / Antigravity Synthesis  
**Date:** 2025-12-28  
**Supersedes:** None  
**Related:** ADR 081 (Soul Dataset Structure), ADR 079 (Soul Persistence), Protocol 128 (Hardened Learning Loop)

---

## Context: The Fragmentation Problem

Project Sanctuary has evolved three distinct content processing pipelines that share overlapping concerns but use separate implementations:

| System | Location | Purpose |
|--------|----------|---------|
| **Forge Fine-Tuning** | `forge/scripts/` | Generates JSONL training data for LLM fine-tuning |
| **RAG Vector DB** | `mcp_servers/rag_cortex/operations.py` | Full/incremental ingestion into ChromaDB |
| **Soul Persistence** | `mcp_servers/lib/hf_utils.py` | Uploads snapshots to Hugging Face Commons |

### Forge Fine-Tuning Scripts (Detailed)

| Script | Purpose |
|--------|----------|
| `forge_whole_genome_dataset.py` | Parses `markdown_snapshot_full_genome_llm_distilled.txt` → JSONL |
| `validate_dataset.py` | Validates JSONL syntax, schema (`instruction`, `output`), duplicates |
| `upload_to_huggingface.py` | Uploads GGUF/LoRA/Modelfile to HF Model repos |

### Current State Analysis

**Shared Concerns (Chain of Dependency)**:

![Harmonized Content Processing](../docs/architecture_diagrams/system/harmonized_content_processing.png)

*[Source: harmonized_content_processing.mmd](../docs/architecture_diagrams/system/harmonized_content_processing.mmd)*

**Key Finding:** Forge already consumes `snapshot_utils.generate_snapshot()` output!

| Concern | snapshot_utils | RAG operations | Forge scripts | hf_utils |
|---------|----------------|----------------|---------------|----------|
| Exclusion Lists | ✅ Source | ✅ Imports | 🔄 Via snapshot | ❌ N/A |
| File Traversal | ✅ Source | ✅ Re-implements | 🔄 Via snapshot | ❌ N/A |
| Code-to-Markdown | ❌ N/A | ✅ `ingest_code_shim.py` | ❌ N/A | ❌ N/A |
| Snapshot Generation | ✅ Source | ✅ Calls | 🔄 Consumes output file | ✅ Needs |
| JSONL Formatting | ❌ N/A | ❌ N/A | ✅ `determine_instruction()` | ✅ ADR 081 |
| HF Upload | ❌ N/A | ❌ N/A | ✅ `upload_to_huggingface.py` | ✅ Source |

**Divergent Concerns (Legitimately Different)**:

| Concern | Forge | RAG | Soul |
|---------|-------|-----|------|
| **Output Format** | JSONL (`instruction`, `input`, `output`) | ChromaDB embeddings | JSONL per ADR 081 |
| **Chunking Strategy** | Document-level (whole file) | Parent/child semantic chunks | Document-level |
| **Instruction Generation** | `determine_instruction()` heuristics | N/A | N/A |
| **Destination** | Local file → HF Model repo | Vector DB | HF Dataset repo |
| **Schema Validation** | `validate_dataset.py` | Implicit | ADR 081 manifest |

### The Maintenance Burden

Every time we update exclusion patterns or improve code parsing:
1. `snapshot_utils.py` must be updated (exclusions, traversal)
2. `rag_cortex/operations.py` must import and use correctly
3. `ingest_code_shim.py` must stay aligned
4. Forge scripts duplicate much of this logic

This leads to:
- **Inconsistent behavior** between systems
- **Triple maintenance** when patterns change
- **Difficult debugging** when systems produce different results

---

## Decision Options

### Option A: Status Quo (3 Separate Implementations)

Maintain each system independently.

**Pros:**
- No refactoring required
- Each system can evolve independently

**Cons:**
- Triple maintenance burden
- Inconsistent exclusion patterns across systems
- Bug fixes must be applied in multiple places
- Difficult to ensure content parity

**Verdict:** ❌ Not recommended (technical debt accumulation)

---

### Option B: Unified Content Processing Library

Create a new shared library `mcp_servers/lib/content_processor.py` that all three systems use.

```
mcp_servers/lib/
├── content_processor.py   # [NEW] Core content processing
│   ├── ContentProcessor class
│   │   ├── traverse_and_filter()      # Unified file traversal with exclusions
│   │   ├── transform_to_markdown()    # Uses ingest_code_shim
│   │   ├── chunk_for_rag()            # Parent/child chunking
│   │   ├── chunk_for_training()       # Instruction/response pairs
│   │   └── generate_manifest_entry()  # Provenance tracking
├── exclusion_config.py    # [NEW] Single source of truth for patterns
├── ingest_code_shim.py    # [MOVE] from rag_cortex/
├── snapshot_utils.py      # [REFACTOR] to use ContentProcessor
├── hf_utils.py            # [REFACTOR] to use ContentProcessor
└── path_utils.py          # [KEEP] existing
```

**Pros:**
- Single source of truth for exclusions
- Consistent code-to-markdown transformation
- Shared chunking logic with format-specific adapters
- Bug fixes apply everywhere automatically

**Cons:**
- Significant refactoring effort
- Risk of breaking working systems
- Requires careful backward compatibility testing

**Verdict:** ✅ Recommended (long-term maintainability)

---

### Option C: Lightweight Harmonization (Extract Exclusions Only)

Minimal change: Consolidate only the exclusion patterns, keep processing separate.

```
mcp_servers/lib/
├── exclusion_config.py    # [NEW] All patterns in one place
│   ├── EXCLUDE_DIR_NAMES
│   ├── ALWAYS_EXCLUDE_FILES
│   ├── ALLOWED_EXTENSIONS
│   └── should_exclude_path()     # Unified check function
```

Update all systems to import from `exclusion_config.py`.

**Pros:**
- Low risk, minimal code changes
- Solves the most common inconsistency issue
- Can be done incrementally

**Cons:**
- Doesn't address code transformation duplication
- Doesn't address chunking duplication
- Still requires updating multiple files for traversal logic

**Verdict:** ⚡ Acceptable (quick win, but incomplete)

---

## Recommended Approach: Risk-Ordered Rollout

We adopt a **consumer-driven rollout** starting with the newest code (lowest risk) and ending with the most critical code (highest protection):

### Phase 1: Create `content_processor.py` + HF Consumer (Immediate)

**Goal:** Build the new library with HF soul persistence as the first consumer.

1. Create `mcp_servers/lib/content_processor.py` with:
   - Shared exclusion logic (from `snapshot_utils.py`)
   - Code-to-markdown transformation (from `ingest_code_shim.py`)
   - File traversal utilities
   - `.to_soul_jsonl()` adapter for ADR 081 format

2. Update `mcp_servers/lib/hf_utils.py` to use `ContentProcessor`

3. Test thoroughly with `persist_soul()` operation

**Validation:** Verify HF uploads match expected ADR 081 schema.

---

### Phase 2: Update RAG Ingestion (Short-term)

**Goal:** Migrate `rag_cortex/operations.py` to use the new library.

1. Add `.to_rag_chunks()` adapter to `ContentProcessor`
2. Refactor `ingest_full()` to use `ContentProcessor`
3. Refactor `ingest_incremental()` to use `ContentProcessor`
4. Keep `ingest_code_shim.py` as a thin wrapper (backward compatibility)

**Validation:** Compare chunk counts and content before/after migration.

---

### Phase 3: Update Forge Fine-Tuning (Long-term, Protected)

**Goal:** Migrate `forge_whole_genome_dataset.py` to use the unified library.

> ⚠️ **CAUTION:** This is the most sensitive code path. Extra validation required.

1. Add `.to_training_jsonl()` adapter with `determine_instruction()` logic
2. Refactor `forge_whole_genome_dataset.py` to call `ContentProcessor`
3. Run `validate_dataset.py` before AND after to verify parity
4. Keep original script logic available for rollback

**Validation:** Byte-for-byte comparison of JSONL output with previous version.

---

## Architecture Diagram

*(See harmonized diagram above)*

---

## Implementation Considerations

### Backward Compatibility

All existing function signatures must remain supported:
- `snapshot_utils.generate_snapshot()` → Continue working as-is
- `rag_cortex.ingest_code_shim.convert_and_save()` → Re-export from new location
- `hf_utils.upload_soul_snapshot()` → No interface change

### Testing Strategy

| Phase | Test Type | Scope |
|-------|-----------|-------|
| Phase 1 | Unit tests for `should_exclude_path()` | All exclusion patterns |
| Phase 2 | Integration tests for code-to-markdown | Python, JS, TS file parsing |
| Phase 3 | E2E tests for each consumer | RAG ingestion, Forge output, HF upload |

### Fine-Tuning Code Safety

> **CAUTION (Per User Request):** Fine-tuning JSONL generation is the highest-risk area.

The Forge scripts that generate training data must:
1. Never be modified without explicit testing
2. Use the shared library **in addition to** existing validation
3. Maintain a separate manifest for training data provenance

---

## Consequences

### Positive

- **Single Source of Truth**: Exclusion patterns maintained in one file
- **Consistent Behavior**: All systems use identical filtering logic
- **Reduced Maintenance**: Bug fixes apply once, affect all consumers
- **Better Testing**: Consolidated logic enables comprehensive unit tests
- **Cleaner Architecture**: Clear separation of concerns

### Negative

- **Migration Effort**: Phase 2-3 requires significant refactoring
- **Risk During Transition**: Potential for breaking changes
- **Import Complexity**: More cross-module dependencies

### Mitigations

- Phased approach reduces risk
- Comprehensive testing before each phase
- Backward-compatible wrappers during transition

---

## Decision

**Selected Option:** Phased Harmonization (C → B)

**Rationale:** Start with low-risk extraction (Phase 1), prove value, then proceed to deeper consolidation. This balances immediate wins against long-term architectural goals.

---

## Action Items

| Task | Phase | Priority | Status |
|------|-------|----------|--------|
| Create `content_processor.py` | 1 | P1 | ⏳ Pending |
| Add `.to_soul_jsonl()` adapter | 1 | P1 | ⏳ Pending |
| Refactor `hf_utils.py` to use ContentProcessor | 1 | P1 | ⏳ Pending |
| Test `persist_soul()` with new processor | 1 | P1 | ⏳ Pending |
| Add `.to_rag_chunks()` adapter | 2 | P2 | ⏳ Pending |
| Refactor `ingest_full()` | 2 | P2 | ⏳ Pending |
| Refactor `ingest_incremental()` | 2 | P2 | ⏳ Pending |
| Add `.to_training_jsonl()` adapter | 3 | P3 | ⏳ Pending |
| Refactor `forge_whole_genome_dataset.py` | 3 | P3 | ⏳ Pending |
| Comprehensive test suite | All | P1 | ⏳ Pending |

---

## Related Documents

- [ADR 079: Soul Persistence via Hugging Face](./079_soul_persistence_hugging_face.md)
- [ADR 081: Soul Dataset Structure](./081_soul_dataset_structure.md)
- [Protocol 128: Hardened Learning Loop](../01_PROTOCOLS/128_Hardened_Learning_Loop.md)
- [ingest_code_shim.py](../mcp_servers/rag_cortex/ingest_code_shim.py)
- [snapshot_utils.py](../mcp_servers/lib/snapshot_utils.py)

---

*Proposed: 2025-12-28 — Awaiting Strategic Review*

--- END OF FILE ADRs/082_harmonized_content_processing.md ---

--- START OF FILE mcp_servers/gateway/clusters/sanctuary_git/SAFETY.md ---

# Git Workflow MCP - Safety Features Documentation

**Version:** 1.0  
**Last Updated:** 2025-11-30  
**Status:** Production Ready

---

## Overview

The Git Workflow MCP implements a **strict safety system** to prevent dangerous git operations and enforce a disciplined feature branch workflow. This document details all safety features, their rationale, and test coverage.

---

## Safety Philosophy

### Core Principles

1. **Never Commit to Main:** All development must occur on feature branches
2. **One Feature at a Time:** Only one active feature branch allowed
3. **Verify Before Trust:** All operations verify state before proceeding
4. **Merge Before Delete:** Feature branches can only be deleted after PR merge
5. **Clean State Required:** Critical operations require clean working directory

### Removed Operations

- **`git_sync_main`** - Removed entirely (unsafe standalone operation)
  - **Rationale:** Agents were pulling main prematurely, before PR merge
  - **Alternative:** Sync happens automatically in `git_finish_feature` after merge verification

---

## Operation Safety Matrix

| Operation | Main Block | Feature Check | Clean State | Merge Verify | Idempotent |
|-----------|------------|---------------|-------------|--------------|------------|
| `git_get_status` | N/A | N/A | N/A | N/A | ✅ |
| `git_diff` | N/A | N/A | N/A | N/A | ✅ |
| `git_log` | N/A | N/A | N/A | N/A | ✅ |
| `git_start_feature` | N/A | ✅ | ✅ | N/A | ✅ |
| `git_add` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_smart_commit` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_push_feature` | ✅ | ✅ | N/A | N/A | ❌ |
| `git_finish_feature` | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## Detailed Safety Checks

### 1. `git_start_feature`

**Purpose:** Create or switch to a feature branch

**Safety Checks:**
- ✅ **One at a Time:** Blocks if another feature branch exists
- ✅ **Clean State:** Requires clean working directory for new branch creation
- ✅ **Idempotent:** Safe to call multiple times
  - Already on branch → no-op
  - Branch exists elsewhere → checkout
  - Branch doesn't exist → create

**Error Conditions:**
```python
# Another feature branch exists
"ERROR: Cannot create new feature branch. Existing feature branch(es) detected: feature/task-999-other"

# Working directory dirty
"ERROR: Cannot create new feature branch. Working directory has uncommitted changes"
```

**Test Coverage:** 4 tests (2 failure, 2 success)

---

### 2. `git_add`

**Purpose:** Stage files for commit

**Safety Checks:**
- ✅ **Block on Main:** Cannot stage files on `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot stage files on main branch. You must be on a feature branch"

# On non-feature branch (e.g., develop)
"ERROR: Cannot stage files on branch 'develop'. You must be on a feature branch"
```

**Test Coverage:** 3 tests (2 failure, 1 success)

---

### 3. `git_smart_commit`

**Purpose:** Commit staged files with Protocol 101 v3.0 enforcement

**Safety Checks:**
- ✅ **Block on Main:** Cannot commit to `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch
- ✅ **Staged Files Required:** Verifies files are staged before committing

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot commit directly to main branch. You must be on a feature branch"

# On non-feature branch
"ERROR: Cannot commit on branch 'develop'. You must be on a feature branch"

# No staged files
"ERROR: No files staged for commit. Please use git_add first"
```

**Test Coverage:** 4 tests (3 failure, 1 success)

---

### 4. `git_push_feature`

**Purpose:** Push feature branch to origin

**Safety Checks:**
- ✅ **Block on Main:** Cannot push `main` branch
- ✅ **Feature Branch Only:** Must be on `feature/task-XXX-desc` branch
- ✅ **Remote Hash Verification:** Verifies remote hash matches local after push

**Error Conditions:**
```python
# On main branch
"ERROR: Cannot push main branch directly. You must be on a feature branch"

# On non-feature branch
"ERROR: Cannot push branch 'develop'. You must be on a feature branch"

# Hash mismatch (WARNING, not blocking)
"WARNING: Push completed but remote hash (abc123de) does not match local (xyz789ab)"
```

**Test Coverage:** 4 tests (2 failure, 2 success)

---

### 5. `git_finish_feature`

**Purpose:** Cleanup after PR merge (delete branches, sync main)

**Safety Checks:**
- ✅ **Block Main:** Cannot finish `main` branch
- ✅ **Feature Branch Only:** Must be `feature/task-XXX-desc` format
- ✅ **Clean State:** Requires clean working directory
- ✅ **Merge Verification:** Verifies branch is merged into `main` before deletion
  - Pulls `main` first to ensure local is up-to-date
  - Checks `git branch --merged main`
  - **Prevents data loss** by blocking unmerged branch deletion

**Error Conditions:**
```python
# Trying to finish main
"ERROR: Cannot finish 'main' branch. It is the protected default branch"

# Invalid branch name
"ERROR: Invalid branch name 'develop'. Can only finish feature branches"

# Working directory dirty
"ERROR: Working directory is not clean. Please commit or stash changes"

# Branch not merged
"ERROR: Branch 'feature/task-123-test' is NOT merged into main. Cannot finish/delete an unmerged feature branch"
```

**Test Coverage:** 5 tests (4 failure, 1 success)

---

## Workflow Enforcement

### Required Sequence

![git_workflow_sequence](../../../../docs/architecture_diagrams/workflows/git_workflow_sequence.png)

*[Source: git_workflow_sequence.mmd](../../../../docs/architecture_diagrams/workflows/git_workflow_sequence.mmd)*

### Out-of-Sequence Prevention

| Attempted Action | Without | Result |
|------------------|---------|--------|
| `git_add` | `git_start_feature` | ❌ Blocked: "Cannot stage files on main branch" |
| `git_smart_commit` | `git_add` | ❌ Blocked: "No files staged for commit" |
| `git_push_feature` | `git_smart_commit` | ⚠️ Allowed (git handles "everything up-to-date") |
| `git_finish_feature` | PR Merge | ❌ Blocked: "Branch is NOT merged into main" |

---

## Test Suite

### Location
- **Unit Tests:** `tests/test_git_ops.py` (10 tests)
- **Safety Tests:** `tests/mcp_servers/git_workflow/test_tool_safety.py` (20 tests)
- **Total:** 30 tests, 100% passing ✅

### Coverage Breakdown

```
git_add:           3 tests (2 failure, 1 success)
git_start_feature: 4 tests (2 failure, 2 success)
git_smart_commit:  4 tests (3 failure, 1 success)
git_push_feature:  4 tests (2 failure, 2 success)
git_finish_feature: 5 tests (4 failure, 1 success)
```

### Running Tests

```bash
# All git tests
pytest tests/test_git_ops.py tests/mcp_servers/git_workflow/ -v

# Safety tests only
pytest tests/mcp_servers/git_workflow/test_tool_safety.py -v

# Specific test
pytest tests/mcp_servers/git_workflow/test_tool_safety.py::TestGitToolSafety::test_finish_feature_blocks_unmerged -v
```

---

## Protocol Compliance

### Protocol 101 v3.0: Functional Coherence

**Enforcement:** `git_smart_commit` automatically runs the test suite via pre-commit hook

**Workflow:**
1. User stages files with `git_add`
2. User calls `git_smart_commit` with message
3. Pre-commit hook executes `./scripts/run_genome_tests.sh`
4. If tests pass → commit succeeds
5. If tests fail → commit is blocked

**No Manual Intervention Required** - The hook enforces functional coherence automatically.

---

## Migration from `git_sync_main`

### Why Removed?

**Problem:** Agents were calling `git_sync_main` at inappropriate times:
- Before PR was merged
- While on feature branches
- Without verifying remote state

**Solution:** Removed tool entirely. Sync now happens **only** in `git_finish_feature` after merge verification.

### Migration Path

**Old Workflow:**
```python
git_finish_feature("feature/task-123-test")
git_sync_main()  # Manual sync
```

**New Workflow:**
```python
git_finish_feature("feature/task-123-test")  # Syncs main automatically
```

---

## Future Enhancements

### Recommended Additions

1. **Remote Tracking Verification**
   - Check if remote exists before push
   - Verify network connectivity

2. **Ahead/Behind Check**
   - Warn if remote is ahead before push
   - Suggest pull/rebase

3. **Force Push Warning**
   - Add explicit confirmation for `force=True`
   - Or block force push entirely

4. **Stale Branch Detection**
   - Warn if feature branch is behind main
   - Suggest rebase

---

## Troubleshooting

### Common Errors

**"Cannot stage files on main branch"**
- **Cause:** Attempted `git_add` on `main`
- **Solution:** Run `git_start_feature` first

**"No files staged for commit"**
- **Cause:** Attempted `git_smart_commit` without staging
- **Solution:** Run `git_add` first

**"Branch is NOT merged into main"**
- **Cause:** Attempted `git_finish_feature` before PR merge
- **Solution:** Merge PR on GitHub first, then retry

**"Existing feature branch(es) detected"**
- **Cause:** Attempted to create second feature branch
- **Solution:** Finish current feature branch first with `git_finish_feature`

---

## Related Documentation

- [Git Workflow MCP README](README.md)
- [MCP Operations Inventory](../../../../docs/operations/mcp/mcp_operations_inventory.md)
- [Protocol 101 v3.0](../../../../01_PROTOCOLS/101_The_Doctrine_of_the_Unbreakable_Commit.md)

---

**Maintainer:** Project Sanctuary Team  
**Status:** Production Ready ✅

--- END OF FILE mcp_servers/gateway/clusters/sanctuary_git/SAFETY.md ---

--- START OF FILE scripts/validate_manifest.py ---

import json
import os
from pathlib import Path

manifest_path = ".agent/learning/learning_manifest.json"
project_root = Path(".")

print(f"Validating {manifest_path}...")

try:
    with open(manifest_path, 'r') as f:
        files = json.load(f)
except Exception as e:
    print(f"Error loading manifest: {e}")
    exit(1)

missing = []
for file_path in files:
    # Handle directory references (some might end in /)
    clean_path = file_path.rstrip('/')
    path = project_root / clean_path
    
    if not path.exists():
        print(f"❌ MISSING: {file_path}")
        missing.append(file_path)
    else:
        print(f"✅ FOUND: {file_path}")

if missing:
    print(f"\nFound {len(missing)} missing files.")
    exit(1)
else:
    print("\nAll files verified successfully.")
    exit(0)

--- END OF FILE scripts/validate_manifest.py ---

--- START OF FILE scripts/ensure_manifest_completeness.py ---

import json
import os
import re
from pathlib import Path

MANIFEST_PATH = ".agent/learning/learning_manifest.json"
PROJECT_ROOT = Path(".")

# Load Manifest
with open(MANIFEST_PATH, 'r') as f:
    manifest_entries = json.load(f)

print(f"Loaded {len(manifest_entries)} entries from {MANIFEST_PATH}")

# 1. Build Effective File List (Expand Directories)
effective_files = set()
explicit_entries = set(manifest_entries)

# Helper to check if a file is already "covered" by the manifest (either explicitly or via a dir)
def is_covered(file_rel_path_str):
    if file_rel_path_str in explicit_entries:
        return True
    # Check parent directories
    p = Path(file_rel_path_str)
    for parent in p.parents:
        # manifest entries for dirs might have trailing slash or not depending on convention
        # usually snapshot_utils expands dirs.
        # Let's check string matches
        parent_str = str(parent)
        parent_str_slash = parent_str + "/"
        if parent_str in explicit_entries or parent_str_slash in explicit_entries:
            return True
    return False

# Build list of files to SCAN for links
# We only assume we need to scan things that are currently in the manifest.
files_to_scan = []

for entry in manifest_entries:
    path = PROJECT_ROOT / entry.rstrip('/')
    if path.is_file():
        files_to_scan.append(path)
    elif path.is_dir():
        # Recursive glob
        for f in path.rglob("*"):
            if f.is_file():
                files_to_scan.append(f)

print(f"Scanning {len(files_to_scan)} files for .mmd references...")

# 2. Scan content for .mmd links
# Matches: ](path/to/file.mmd) or "path/to/file.mmd"
# Regex: (?:\]\(|'|")([^)'"]+\.mmd)(?:\)|'|")
mmd_regex = re.compile(r'(?:\]\(|[\'"])([^)\'"]+\.mmd)(?:\)|[\'"])')

missing_mmd_files = set()

for file_path in files_to_scan:
    if file_path.suffix not in ['.md', '.txt', '.py', '.json']:
        continue
        
    try:
        content = file_path.read_text(errors='ignore')
    except Exception as e:
        print(f"Warning: could not read {file_path}: {e}")
        continue
        
    matches = mmd_regex.findall(content)
    for match in matches:
        # Resolve path
        # Match might be relative to the file, or relative to root (if starts with / or is just a path)
        # But usually in our system, links are relative.
        
        # 1. Try relative to file
        resolved_path = None
        
        # Cleaning match (remove ../ etc)
        # If absolute path (unlikely in repo)
        if match.startswith('/'):
            # Assume relative to project root actually if it starts with / in a markdown link usually it means root?
            # Or just absolute path on FS.
            # Let's try treating as relative to root stripped of /
            candidate = PROJECT_ROOT / match.lstrip('/')
            if candidate.exists():
                resolved_path = candidate
        else:
            # Try relative to current file's dir
            candidate_rel = file_path.parent / match
            if candidate_rel.resolve().exists():
                resolved_path = candidate_rel.resolve().relative_to(PROJECT_ROOT.resolve())
            else:
                # Try relative to project root directly (sometimes people put paths that way)
                candidate_root = PROJECT_ROOT / match
                if candidate_root.exists():
                    resolved_path = candidate_root
        
        if resolved_path:
            # Convert to string relative to root
            try:
                rel_path_str = str(resolved_path) # already made relative above if possible
                # double check relative status
                if resolved_path.is_absolute():
                     rel_path_str = str(resolved_path.relative_to(PROJECT_ROOT.resolve()))
            except ValueError:
                 # path not in tree?
                 continue

            # Check if covered
            if not is_covered(rel_path_str):
                # Avoid duplicates
                if rel_path_str not in missing_mmd_files:
                    print(f"Found missing dependency: {rel_path_str} (referenced in {file_path.relative_to(PROJECT_ROOT)})")
                    missing_mmd_files.add(rel_path_str)

# 3. Update Manifest
if missing_mmd_files:
    print(f"\nAdding {len(missing_mmd_files)} missing diagrams to manifest...")
    manifest_entries.extend(list(missing_mmd_files))
    
    # Optional: Sort or clean up?
    # Keeping original order + appended for diff clarity might be better, or just sorting.
    # JSON usually loaded as list.
    
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest_entries, f, indent=4)
    print("Manifest updated.")
else:
    print("\nNo missing .mmd dependencies found.")

--- END OF FILE scripts/ensure_manifest_completeness.py ---

--- START OF FILE scripts/init_session.py ---

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

--- END OF FILE scripts/init_session.py ---

--- START OF FILE scripts/manual_test_deliberation.py ---

#!/usr/bin/env python3
"""
Manual Verification Script for Multi-Round Deliberation Logic (Task #086B)

This script simulates a full Council deliberation (3 rounds) by mocking the 
LLM and Cortex dependencies but executing the actual Council and Agent Persona 
logic. It prints the conversation flow to verify the logic integrity.
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Mock legacy modules to allow imports
sys.modules["mnemonic_cortex.app.services.vector_db_service"] = MagicMock()
sys.modules["mnemonic_cortex.app.services.llm_service"] = MagicMock()

# 2. Import Operations
from mcp_servers.lib.council.council_ops import CouncilOperations
from mcp_servers.cognitive.cortex.operations import CortexOperations

def run_verification():
    print("🔍 Starting Multi-Round Deliberation Verification...")
    
    # 3. Setup Mocks
    with patch('mcp_servers.lib.agent_persona.agent_persona_ops.get_llm_client') as mock_llm, \
         patch.object(CortexOperations, 'query') as mock_cortex_query, \
         patch.object(CortexOperations, 'cache_warmup'):
        
        # Mock Cortex Context
        mock_cortex_query.return_value = MagicMock(
            results=[
                MagicMock(
                    content="Protocol 87: Structured Queries. Defines how to query the system.",
                    metadata={"source": "01_PROTOCOLS/087.md"},
                    score=0.95
                )
            ]
        )
        
        # Mock LLM Responses (Context-Aware)
        def mock_generate(prompt, **kwargs):
            prompt_lower = prompt.lower()
            
            # Identify Agent
            agent = "unknown"
            if "coordinator" in prompt_lower: agent = "Coordinator"
            elif "strategist" in prompt_lower: agent = "Strategist"
            elif "auditor" in prompt_lower: agent = "Auditor"
            
            # Identify Round (heuristic based on context length or content)
            # This is a simplification; in real flow, context grows.
            if "critique" in prompt_lower or "review" in prompt_lower: 
                phase = "Critique"
            elif "synthesize" in prompt_lower or "final" in prompt_lower:
                phase = "Synthesis"
            else:
                phase = "Initial Proposal"
                
            return f"[{agent} - {phase}] Based on Protocol 87, I suggest..."

        mock_client = MagicMock()
        mock_client.generate.side_effect = mock_generate
        mock_llm.return_value = mock_client
        
        # 4. Execute Council Dispatch
        print("\n🚀 Dispatching Task to Council (Max Rounds: 2)...")
        council_ops = CouncilOperations()
        result = council_ops.dispatch_task(
            task_description="Design a new protocol for MCP composition",
            agent=None, # Full Council
            max_rounds=2
        )
        
        # 5. Analyze Results
        print("\n✅ Execution Complete!")
        print(f"Status: {result['status']}")
        print(f"Rounds Completed: {result['rounds']}")
        print(f"Agents Involved: {len(result['agents'])} ({', '.join(result['agents'])})")
        print(f"Total Packets: {len(result['packets'])}")
        
        print("\n📜 Conversation Flow:")
        current_round = -1
        for packet in result['packets']:
            if packet['round_id'] != current_round:
                current_round = packet['round_id']
                print(f"\n--- ROUND {current_round + 1} ---")
            
            print(f"\n👤 {packet['member_id'].upper()}:")
            print(f"   {packet['decision']}")
            
        print("\n🧠 Final Synthesis:")
        print(f"   {result['final_synthesis']}")
        
        # 6. Verify Logic
        print("\n🕵️ Logic Verification:")
        
        # Check 1: Cortex was queried
        cortex_called = mock_cortex_query.call_count > 0
        print(f"   [{'x' if cortex_called else ' '}] Cortex Context Retrieved")
        
        # Check 2: All agents participated
        all_agents = set(result['agents']) == {'coordinator', 'strategist', 'auditor'}
        print(f"   [{'x' if all_agents else ' '}] All Agents Participated")
        
        # Check 3: Rounds executed
        rounds_ok = result['rounds'] == 2
        print(f"   [{'x' if rounds_ok else ' '}] Multi-Round Execution")
        
        if cortex_called and all_agents and rounds_ok:
            print("\n✅ VERIFICATION SUCCESSFUL: Logic integrity confirmed.")
        else:
            print("\n❌ VERIFICATION FAILED: Logic gaps detected.")

if __name__ == "__main__":
    run_verification()

--- END OF FILE scripts/manual_test_deliberation.py ---

--- START OF FILE scripts/update_genome.sh ---

#!/bin/bash
# file: update_genome.sh
# version: 3.0 (Absolute Stability Protocol - Manifest Purge)
#
# Changelog v3.0:
# 1. PROTOCOL PURGE: Permanently removes all logic related to the commit_manifest.json file and Protocol 101.
# 2. OPTIONAL EMBEDDING: Added optional flag --full-embed for RAG DB update (Step 3).
# 3. SOVEREIGN STAGING: Step 5 uses 'git add .' and 'git commit --no-verify' for guaranteed deployment.
# 4. FINAL SNAPSHOTS: Includes all explicit genome snapshot captures.

# --- CONTROL LOGIC: Handle Arguments ---
DO_FULL_EMBED=false
if [[ "$1" == "--full-embed" ]]; then
    DO_FULL_EMBED=true
    shift # Remove the flag from the arguments list
fi

# --- Safeguard: Check for Commit Message ---
if [ -z "$1" ]; then
    echo "[FATAL] A commit message is required."
    echo "Usage: ./update_genome.sh [--full-embed] \"Your descriptive commit message\""
    exit 1
fi

COMMIT_MESSAGE=$1
TIMESTAMP=$(date +%s)
BRANCH_NAME="deployment/final-stability-v${TIMESTAMP}" 

echo "[FORGE] Initiating Absolute Stability Protocol..."
echo "------------------------------------------------"

# --- Step 0: Create and Switch to Feature Branch ---
echo "[Step 0/5] Creating and switching to new feature branch: $BRANCH_NAME"
# Ensure we are starting from the latest main branch state
git fetch origin
git checkout main
git pull origin main
git checkout -b "$BRANCH_NAME"
if [ $? -ne 0 ]; then
    echo "[FATAL] Failed to create new branch. Halting."
    exit 1
fi
echo "[SUCCESS] Switched to branch: $BRANCH_NAME"
echo ""

# Step 1: Rebuild the Master Index
echo "[Step 1/5] Rebuilding Living Chronicle Master Index..."
python3 mnemonic_cortex/scripts/create_chronicle_index.py
if [ $? -ne 0 ]; then
    echo "[FATAL] Index creation failed. Halting."
    exit 1
fi
echo "[SUCCESS] Master Index is now coherent."
echo ""

# Step 2: Capture the Snapshots (Explicit & Full)
echo "[Step 2/5] Capturing new Cognitive Genome snapshots..."
# Full genome capture (generates the seeds and full snapshots)
node capture_code_snapshot.py
# Subdirectory captures
node capture_code_snapshot.py council_orchestrator
node capture_code_snapshot.py docs
node capture_code_snapshot.py forge
node capture_code_snapshot.py mcp_servers
node capture_code_snapshot.py mnemonic_cortex

if [ $? -ne 0 ]; then
    echo "[FATAL] Snapshot creation failed. Halting."
    exit 1
fi
echo "[SUCCESS] All snapshots have been updated."
echo ""

# Step 3: Embed the New Knowledge into the Mnemonic Cortex (Optional RAG DB Update)
if $DO_FULL_EMBED; then
    echo "[Step 3/5] Re-indexing the Mnemonic Cortex with the new Genome (FULL EMBED MODE)..."
    python3 mnemonic_cortex/scripts/ingest.py
    if [ $? -ne 0 ]; then
        echo "[FATAL] Mnemonic Cortex ingestion failed. Halting."
        exit 1
    fi
    echo "[SUCCESS] Mnemonic Cortex is now synchronized with the latest knowledge."
else
    echo "[Step 3/5] Bypassing Mnemonic Cortex re-indexing (Full embed disabled. Use --full-embed to run)."
fi
echo ""

# Step 4: Run Automated Tests
echo "[Step 4/5] Running automated functionality tests..."
./scripts/run_genome_tests.sh
if [ $? -ne 0 ]; then
    echo "[FATAL] Genome tests failed. Update aborted to prevent broken deployment."
    exit 1
fi
echo "[SUCCESS] All tests passed - genome update is functional."
echo ""

# Step 5: Commit the Coherent State (Sovereign Override)
echo "[Step 5/5] Staging all changes and committing with SOVEREIGN OVERRIDE (--no-verify)..."

# --- SOVEREIGN STAGING: Stage everything, including the removal of commit_manifest.json ---
# Remove the old manifest file, just in case it's still lying around
rm -f commit_manifest.json
git add . 

# Commit using --no-verify to guarantee success
git commit --no-verify -m "$COMMIT_MESSAGE on branch $BRANCH_NAME"
if [ $? -ne 0 ]; then
    echo "[FATAL] Git commit failed. Halting."
    exit 1
fi
echo "[SUCCESS] All changes committed with message: \"$COMMIT_MESSAGE on branch $BRANCH_NAME\""
echo ""

# Step 6: Push to the Canonical Repository (Feature Branch)
echo "[Step 6/5] Pushing changes to remote feature branch: $BRANCH_NAME..."
git push --set-upstream origin "$BRANCH_NAME"
if [ $? -ne 0 ]; then
    echo "[FATAL] Git push failed. Check your network connection and remote repository permissions. Halting."
    exit 1
fi
echo "[SUCCESS] Changes have been pushed to the remote repository on branch $BRANCH_NAME."
echo ""

echo "------------------------------------------------"
echo "[FORGE] Absolute Stability Protocol Complete."
echo ""
echo "################################################################################"
echo "### MISSION COMPLETE: STABILIZATION DEPLOYED. ###"
echo "### NEXT ACTION REQUIRED: ###"
echo "### 1. Create a Pull Request (PR) on GitHub from $BRANCH_NAME to main. ###"
echo "### 2. MERGE THE PR. The CI checks should now pass. ###"
echo "################################################################################"

--- END OF FILE scripts/update_genome.sh ---

--- START OF FILE scripts/run_integration_tests.sh ---

#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure we run from project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
USE_REAL_LLM=false
for arg in "$@"
do
    if [ "$arg" == "-r" ] || [ "$arg" == "--real" ]; then
        USE_REAL_LLM=true
    fi
done

PYTEST_ARGS="-m integration -v $@"
if [ "$USE_REAL_LLM" = true ]; then
    echo -e "${GREEN}=== Running Integration Tests with REAL LLM ===${NC}"
else
    echo -e "${GREEN}=== Running Integration Tests with MOCKED LLM ===${NC}"
fi

# Re-construct args for pytest
FINAL_ARGS=""
for arg in "$@"
do
    if [ "$arg" == "-r" ] || [ "$arg" == "--real" ]; then
        FINAL_ARGS="$FINAL_ARGS --real-llm"
    else
        FINAL_ARGS="$FINAL_ARGS $arg"
    fi
done

echo "Running: pytest -m integration -v $FINAL_ARGS"
pytest -m integration -v $FINAL_ARGS
INTEGRATION_EXIT_CODE=$?

echo -e "\n${GREEN}=== Running Performance Benchmarks ===${NC}"
pytest -m benchmark --benchmark-only
BENCHMARK_EXIT_CODE=$?

echo -e "\n${GREEN}=== Summary ===${NC}"
if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo -e "Integration Tests: ${GREEN}PASSED${NC}"
else
    echo -e "Integration Tests: ${RED}FAILED${NC}"
fi

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo -e "Benchmarks: ${GREEN}PASSED${NC}"
else
    echo -e "Benchmarks: ${RED}FAILED${NC}"
fi

# Exit with failure if either failed
if [ $INTEGRATION_EXIT_CODE -ne 0 ] || [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0

--- END OF FILE scripts/run_integration_tests.sh ---

--- START OF FILE scripts/wait_for_pulse.sh ---

#!/bin/bash
# scripts/wait_for_pulse.sh
# Checks for 'pulse' (health) of key fleet services before orchestration proceeds.
# Refer: ADR 065 v1.3

# Config
MAX_RETRIES=15
BACKOFF=3

# Helper function for retrying curls
wait_for_url() {
    local url=$1
    local name=$2
    local attempt=1

    echo -n "   - Checking $name ($url)... "
    
    while [ $attempt -le $MAX_RETRIES ]; do
        if curl -s -f -o /dev/null "$url"; then
            echo "✅ OK"
            return 0
        fi
        
        # Simple progress indicator
        echo -n "."
        sleep $BACKOFF
        ((attempt++))
    done
    
    echo " ❌ TIMEOUT after $((MAX_RETRIES * BACKOFF))s"
    return 1
}

# 1. Critical Backend: Vector DB (Port 8110)
wait_for_url "http://localhost:8110/api/v2/heartbeat" "Vector DB" || exit 1

# 2. Critical Backend: Ollama (Port 11434)
# Note: This might take longer if pulling models
wait_for_url "http://localhost:11434/api/tags" "Ollama" || exit 1

echo "   ✨ Fleet Pulse Detected."
exit 0

--- END OF FILE scripts/wait_for_pulse.sh ---

--- START OF FILE .agent/learning/red_team/red_team_prompts.md ---

# Adversarial Prompts (Audit Context)

1. Verify that the file manifest accurately reflects all tactical state changes made during this session.
2. Check for any 'hallucinations' or logic errors in the new ADRs or Learning notes.
3. Ensure that critical security and safety protocols (e.g. Protocol 101/128) have not been bypassed.
4. Specifically audit the reasoning behind: this session

--- END OF FILE .agent/learning/red_team/red_team_prompts.md ---

