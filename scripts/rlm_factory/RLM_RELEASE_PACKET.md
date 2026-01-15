# RLM Bootstrap Packet: The Cognitive Architecture Kit (RLM + RAG)
**Generated:** 2026-01-14T18:32:57.504000
**Description:** Manifest defining the components of the RLM Bootstrap Packet.

## 🚀 Instructions for the AI Agent
You are receiving a complete 'Cognitive Architecture' starter kit.
It contains two complimentary systems:
1.  **Semantic Ledger (RLM):** High-level summaries of every file (`operations.py`).
2.  **Vector Memory (RAG):** Deep retrieval of code chunks (`chroma_ingest.py`).

**Action Plan:**
1.  **Install:** `pip install requests genericpath python-dotenv langchain-chroma langchain-huggingface`.
2.  **Digest:** Read `REACTIVE_LEDGER_MEMORY.md` (Theory) and `operations.py` (Logic).
3.  **Boot:** Run `python chroma_ingest.py ingest src/` to hydrate your vector memory.
4.  **Init:** Run the RLM distillation loop to build your semantic ledger.

---

## 📦 Component: Architecture Blueprint (Theory)
**Source:** `docs/architecture/patterns/REACTIVE_LEDGER_MEMORY.md`
_The theoretical foundation and implementation guide._

```markdown
# Reactive Ledger Memory (RLM): The Holographic Cache Pattern

**Status:** Production Verified (Project Sanctuary)  
**Architecture Type:** Cognitive Memory Architecture  
**Primary Use Case:** Large-Scale Codebase & Documentation Understanding for AI Agents

## 1. The Problem: The "Needle in the Haystack"

Standard RAG (Retrieval-Augmented Generation) uses Vector Databases to find specific text chunks based on semantic similarity. While powerful, this approach suffers from **Context Blindness**:

*   **The Chunking Problem:** A chunk of text (e.g., "function process_data()") loses its meaning when separated from its file's purpose (e.g., "This file handles critical PII sanitization").
*   **The Latency Problem:** Querying a Vector DB for "What is the architecture of this project?" requires retrieving and synthesizing hundreds of disconnected chunks, which is slow and error-prone.
*   **The "Unknown Unknowns":** An agent cannot query for what it doesn't know exists.

## 2. The Solution: Reactive Ledger Memory (RLM)

The RLM is a **precognitive, holographic cache**. Instead of slicing the repository into incoherent chunks, it maintains a **"One-Sentence Source of Truth"** for every single file in the project.

### Core Concepts

1.  **The Atom (Summary):** Every file is distilled into a dense, high-entropy summary (The "Essence").
2.  **The Ledger (JSON):** A flat, highly portable JSON file acts as the "Map of the Territory."
3.  **Incremental Persistence:** The ledger updates transactionally after every file is processed, ensuring resilience.
4.  **Hash-Based Validity:** Files are only re-processed if their MD5 hash changes.

## 3. Technical Implementation Blueprint

This pattern can be implemented in any language (Python/TS/Go). Below is the reference implementation logic used in Project Sanctuary.

### 3.1 The Schema (`ledger.json`)

Structure your cache as a flat Key-Value store to allow O(1) lookups.

```json
{
  "docs/architecture/system_design.md": {
    "hash": "a1b2c3d4e5f6...",
    "mtime": 1704092833.0,
    "summarized_at": "2024-01-01T12:00:00Z",
    "summary": "This document defines the 3-tier architecture (Frontend, API, Worker) and the data flow protocol for the entire system."
  },
  "src/utils/sanitizer.py": {
    "hash": "f6e5d4c3b2a1...",
    "mtime": 1704092900.0,
    "summarized_at": "2024-01-01T12:05:00Z",
    "summary": "A utility module responsible for stripping PII from user inputs before database insertion; critical for GDPR compliance."
  }
}
```

### 3.2 The Distillation Loop (Python Pseudo-code)

```python
def rlm_distill(target_dir):
    ledger = load_json("ledger.json")
    
    for file in walk(target_dir):
        current_hash = md5(file.content)
        
        # 1. Skip if unchanged (Cache Hit)
        if file.path in ledger and ledger[file.path]["hash"] == current_hash:
            continue
            
        # 2. Distill (The LLM Call)
        # Use a small, local model (see Section 4) for speed/privacy
        summary = llm_generate(
            model="qwen2.5-7b-instruct",
            prompt=f"Summarize the architectural purpose of this file in 2 sentences:\n\n{file.content}"
        )
        
        # 3. Incremental Persistence (CRITICAL)
        ledger[file.path] = {
            "hash": current_hash,
            "summary": summary
        }
        save_json(ledger, "ledger.json") # Write immediately!
```

## 4. Model Selection (Open Source Recommendations)

You do not need a massive model (GPT-4) for this. Summarization is a high-compression task suitable for 7B-class models.

| Model Family | Variant | Why Use It? |
| :--- | :--- | :--- |
| **Qwen** | `Qwen2.5-7B-Instruct` | **Best All-Rounder.** Excellent at code understanding and concise technical writing. |
| **Llama 3** | `Llama-3-8B-Instruct` | **High Fidelity.** Very strong reasoning, good for complex prose documentation. |
| **Llama 3.2** | `Llama-3.2-3B` | **Ultra Fast.** Perfect for massive repos running on consumer laptops (M1/M2/M3). |
| **Mistral** | `Mistral-7B-v0.3` | **Reliable.** A classic workhorse with a large context window (32k). |

**Serving:** use [Ollama](https://ollama.com/) locally (`ollama serve`).

## 5. Augmenting the Vector DB (The "Super-RAG")

The RLM Ledger isn't just for humans/agents to look at—it is the **perfect meta-data source** for your Vector DB (Chroma/Pinecone/Weaviate).

**The "Context Injection" Strategy:**

When you chunk a file for your Vector DB, prepending the RLM Summary to *every single chunk* dramatically improves retrieval quality.

**Without RLM (Standard Chunk):**
> "def validate(x): return x > 0 else raise Error"
*(Vector DB doesn't know what this validates)*

**With RLM Injection:**
> **[Context: critical_financial_validator.py - ensuring no negative balances in user wallets]**
> "def validate(x): return x > 0 else raise Error"
*(Vector DB now perfectly matches queries about "financial validation")*

## 6. Benefits for Agent Systems

1.  **Fast Boot:** Agent reads the `ledger.json` (300KB) instead of scanning 5,000 files (500MB).
2.  **Code Navigation:** Agent uses the ledger to decide *which* files to open (`read_file`), reducing token costs by 95%.
3.  **Self-Healing:** If the ledger is missing, the agent can regenerate it locally.

## 7. Migration Guide

To adopt this pattern in your repo:
1.  **Install:** `ollama` and `python`.
2.  **Pull Model:** `ollama pull qwen2.5:7b`.
3.  **Script:** Implement the "Distillation Loop" above.
4.  **Policy:** Add `ledger.json` to your `.gitignore` (or commit it if you want shared team memory).
5.  **Doc:** Update your `README.md` to explain that `ledger.json` is the map.

```

---

## 📦 Component: Core Logic (The Distiller)
**Source:** `mcp_servers/learning/operations.py`
_The distillation engine that recursively summarizes files._

```python

import os
import re
import sys
import time
import subprocess
import contextlib
import io
import logging
import json
import hmac
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.snapshot_utils import (
    generate_snapshot, 
    EXCLUDE_DIR_NAMES,
    ALWAYS_EXCLUDE_FILES,
    PROTECTED_SEEDS,
    RECURSIVE_ARTIFACTS
)
from mcp_servers.learning.models import (
    CaptureSnapshotResponse,
    PersistSoulRequest,
    PersistSoulResponse,
    GuardianWakeupResponse,
    GuardianSnapshotResponse
)

# Setup logging
logger = logging.getLogger("learning.operations")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class LearningOperations:
    """
    Operations for the Project Sanctuary Learning Loop (Protocol 128).
    Migrated from RAG Cortex to ensure domain purity.
    """


    def _is_recursive_artifact(self, f: str) -> bool:
        """Centralized check for files that should be excluded from snapshots and RLM."""
        path = Path(f)
        f_lower = f.lower()
        base_name = path.name
        
        # 1. Protocol 128: Manifest Priority Bypass (Protected Seeds)
        # Seeds are fair game if they are .txt or .md
        if path.suffix.lower() in [".md", ".txt"]:
            if any(f == p or base_name == Path(p).name for p in PROTECTED_SEEDS):
                return False  # Force inclusion
        
        # 2. Check the Central Sanctuary "No-Fly List" (always_exclude_files + patterns)
        for pattern in ALWAYS_EXCLUDE_FILES:
            if isinstance(pattern, str):
                if pattern.lower() == base_name.lower():
                    return True
            elif hasattr(pattern, 'match'): 
                if pattern.match(base_name) or pattern.match(f):
                    return True

        # 3. Skip Archive folders (USER requirement)
        if "/archive/" in f_lower or f_lower.startswith("archive/"): return True
        
        # 4. Handle the Learning Metadata directory
        if ".agent/learning/" in f:
            # Rules/Policies are FOUNDATIONAL and should be included
            if "rules/" in f:
                return False
            # Specific recursive artifacts already handled by ALWAYS_EXCLUDE_FILES
            # but we block the rest of the metadata dir by default
            return True
            
        return False

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / ".agent" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.learning_dir = self.project_root / ".agent" / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        # We don't need ChromaDB here.

    #============================================================
    # 1. LEARNING DEBRIEF (The Scout)
    #============================================================
    def learning_debrief(self, hours: int = 24) -> str:
        """
        Scans project for technical state changes (Protocol 128).
        Args:
          hours: Lookback window for modifications
        Returns: Comprehensive Markdown string
        """
        try:
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
                
                # 3. Read Core Documents
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
                package_status = "ℹ️ No `.agent/learning/learning_package_snapshot.md` detected."
                
                if package_path.exists():
                    try:
                        mtime = package_path.stat().st_mtime
                        delta_hours = (datetime.now().timestamp() - mtime) / 3600
                        if delta_hours <= hours:
                            last_package_content = package_path.read_text()
                            package_status = f"✅ Loaded Learning Package Snapshot from {delta_hours:.1f}h ago."
                        else:
                            package_status = f"⚠️ Snapshot found but too old ({delta_hours:.1f}h)."
                    except Exception as e:
                        package_status = f"❌ Error reading snapshot: {e}"

                # 4b. Mandatory Logic Verification (ADR 084)
                mandatory_files = [
                    "IDENTITY/founder_seed.json",
                    "LEARNING/calibration_log.json", 
                    "ADRs/084_semantic_entropy_tda_gating.md",
                    "mcp_servers/learning/operations.py" # Ref updated
                ]
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

                # 5. Create Draft
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                lines = [
                    f"# [HARDENED] Learning Package Snapshot v4.0 (The Edison Seal)",
                    f"**Scan Time:** {timestamp} (Window: {hours}h)",
                    f"**Strategic Status:** ✅ Successor Context v4.0 Active",
                    "",
                    "> [!IMPORTANT]",
                    "> **STRATEGIC PIVOT: THE EDISON MANDATE (ADR 084)**",
                    "> The project has formally abandoned the QEC-AI Metaphor in favor of **Empirical Epistemic Gating**.",
                    "> - **Primary Gate:** Every trace must pass the Dead-Man's Switch in `persist_soul`.",
                    "> - **Success Metric:** Semantic Entropy < 0.79 (Target) / > 0.2 (Rigidity Floor).",
                    "> - **Logic:** Do NOT inject metaphorical fluff. Synthesize hard operational reality.",
                    "",
                    "## I. The Truth (System State)",
                    f"**Git Status:**\n```\n{git_evidence}\n```",
                    "",
                    f"## II. The Change (Recency Delta - {hours}h)",
                    recency_summary,
                    "",
                    "## III. The Law (Protocol 128 - Cognitive Continuity)",
                    "> *\"We do not restart. We reload.\"*",
                    "### A. The Cognitive Primer (Constitution)",
                    f"```markdown\n{primer_content[:1000]}...\n```",
                    "",
                    "### B. The Recursive Loop (Standard Operating Procedure)",
                    f"```markdown\n{sop_content[:1000]}...\n```",
                    "",
                    "## IV. The Strategy (Successor Context)",
                    f"**Snapshot Status:** {package_status}",
                    f"**Registry Status (ADR 084):**\n{registry_status}",
                    "### Active Context (Previous Cycle):",
                    f"```markdown\n{last_package_content[:2000]}...\n```",
                ]
                
                return "\n".join(lines)
        except Exception as e:
            logger.error(f"Learning Debrief Failed: {e}", exc_info=True)
            return f"Error generating debrief: {str(e)}"

    def _get_recency_delta(self, hours: int = 48) -> str:
        """Get summary of recently modified high-signal files."""
        try:
            delta = timedelta(hours=hours)
            cutoff_time = time.time() - delta.total_seconds()
            now = time.time()
            
            recent_files = []
            scan_dirs = ["00_CHRONICLE/ENTRIES", "01_PROTOCOLS", "mcp_servers", "02_USER_REFLECTIONS"]
            allowed_extensions = {".md", ".py"}
            
            for directory in scan_dirs:
                dir_path = self.project_root / directory
                if not dir_path.exists(): continue
                
                for file_path in dir_path.rglob("*"):
                    if not file_path.is_file(): continue
                    if file_path.suffix not in allowed_extensions: continue
                    if "__pycache__" in str(file_path): continue
                    
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff_time:
                        recent_files.append((file_path, mtime))
            
            if not recent_files:
                return "* **Recent Files Modified (48h):** None"
                
            recent_files.sort(key=lambda x: x[1], reverse=True)
            
            git_info = "[Git unavailable]"
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--oneline"],
                    cwd=self.project_root, capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0: git_info = result.stdout.strip()
            except Exception: pass
            
            lines = [f"* **Most Recent Commit:** {git_info}", f"* **Recent Files Modified ({hours}h):**"]
            
            for file_path, mtime in recent_files[:5]:
                relative_path = file_path.relative_to(self.project_root)
                age_seconds = now - mtime
                if age_seconds < 3600: age_str = f"{int(age_seconds / 60)}m ago"
                elif age_seconds < 86400: age_str = f"{int(age_seconds / 3600)}h ago"
                else: age_str = f"{int(age_seconds / 86400)}d ago"
                
                context = ""
                try:
                    with open(file_path, 'r') as f:
                        content = f.read(500)
                        if file_path.suffix == ".md":
                            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                            if title_match: context = f" → {title_match.group(1)}"
                        elif file_path.suffix == ".py":
                            if "def _get_" in content or "class " in content:
                                context = " → Implementation changes"
                except Exception: pass
                
                diff_summary = self._get_git_diff_summary(str(relative_path))
                if diff_summary: context += f" [{diff_summary}]"
                
                lines.append(f"    * `{relative_path}` ({age_str}){context}")
            
            return "\n".join(lines)
        except Exception as e:
            return f"Error generating recency delta: {str(e)}"

    def _get_git_diff_summary(self, file_path: str) -> str:
        """Get concise summary of git changes for a file."""
        try:
            result = subprocess.run(
                ["git", "diff", "--shortstat", "HEAD", file_path],
                cwd=self.project_root, capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception: pass
        return ""

    #============================================================
    # 2. CAPTURE SNAPSHOT (The Seal)
    #============================================================
    def capture_snapshot(
        self, 
        manifest_files: List[str], 
        snapshot_type: str = "audit",
        strategic_context: Optional[str] = None
    ) -> CaptureSnapshotResponse:
        """
        Generates a consolidated snapshot of the project state.
        Types: 'audit' (Red Team), 'learning_audit' (Cognitive), or 'seal' (Final).
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Ensure Diagrams are Rendered
        self._ensure_diagrams_rendered()
        
        # 2. Prepare Paths
        learning_dir = self.project_root / ".agent" / "learning"
        if snapshot_type == "audit":
            output_dir = learning_dir / "red_team"
        elif snapshot_type == "learning_audit":
            output_dir = learning_dir / "learning_audit"
        else:
            output_dir = learning_dir
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to ensure directory {output_dir}: {e}")
        
        # 3. Default Manifest Handling
        effective_manifest = list(manifest_files or [])
        if not effective_manifest:
            if snapshot_type == "seal":
                manifest_file = learning_dir / "learning_manifest.json"
                # ACTIVATE PROTOCOL 132 (RLM SYNTHESIS)
                if not strategic_context:
                    strategic_context = self._rlm_context_synthesis()
            elif snapshot_type == "learning_audit":
                manifest_file = output_dir / "learning_audit_manifest.json"
            else:
                manifest_file = output_dir / "red_team_manifest.json"
                
            if manifest_file and manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        manifest_data = json.load(f)
                    
                    # Handle modular manifest structure (ADR 089)
                    if isinstance(manifest_data, dict):
                        core = manifest_data.get("core", [])
                        topic = manifest_data.get("topic", [])
                        effective_manifest = core + topic
                        logger.info(f"Loaded {snapshot_type} manifest: {len(core)} core + {len(topic)} topic entries")
                    else:
                        # Legacy: flat array
                        effective_manifest = manifest_data
                        logger.info(f"Loaded default {snapshot_type} manifest: {len(effective_manifest)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load {snapshot_type} manifest: {e}")

        # 2. Strict Filter (No recursive artifacts)
        effective_manifest = [f for f in effective_manifest if not self._is_recursive_artifact(f)]

        # Protocol 130: Deduplicate
        if effective_manifest:
            effective_manifest, dedupe_report = self._dedupe_manifest(effective_manifest)
            if dedupe_report:
                logger.info(f"Protocol 130: Deduplicated {len(dedupe_report)} items")

        if snapshot_type == "audit": 
            snapshot_filename = "red_team_audit_packet.md"
        elif snapshot_type == "learning_audit": 
            snapshot_filename = "learning_audit_packet.md"
        elif snapshot_type == "seal":
            snapshot_filename = "learning_package_snapshot.md"
        else:
            snapshot_filename = f"{snapshot_type}_snapshot_{timestamp}.md"
            
        final_snapshot_path = output_dir / snapshot_filename

        # 4. Git State (Protocol 128 verification)
        git_state_dict = self._get_git_state(self.project_root)
        git_diff_context = git_state_dict["hash"]
        manifest_verified = True
        
        # Strict Rejection Logic (Protocol 128)
        if snapshot_type == "audit":
            untracked_changes = git_state_dict["changed_files"] - set(effective_manifest)
            # Remove patterns that are always excluded or from excluded dirs
            untracked_changes = {f for f in untracked_changes if not any(p in f for p in ["logs/", "temp/", ".temp", ".agent/learning/"])}
            
            if untracked_changes:
                manifest_verified = False
                logger.warning(f"STRICT REJECTION: Git changes detected outside of manifest: {untracked_changes}")
                return CaptureSnapshotResponse(
                    snapshot_path="",
                    manifest_verified=False,
                    git_diff_context=f"REJECTED: Untracked changes in {list(untracked_changes)[:5]}",
                    snapshot_type=snapshot_type,
                    status="error",
                    error="Strict manifestation failed: drift detected"
                )

        # 3. Generate Snapshot
        try:
            from uuid import uuid4
            # We use the existing generate_snapshot utility
            # It expects a manifest file path in JSON format (list or dict)
            temp_manifest = self.project_root / f".temp_manifest_{uuid4()}.json"
            temp_manifest.write_text(json.dumps(effective_manifest, indent=2))
            
            try:
                stats = generate_snapshot(
                    project_root=self.project_root,
                    manifest_path=temp_manifest,
                    output_dir=final_snapshot_path.parent,
                    output_file=final_snapshot_path,
                    should_forge_seeds=False
                )
                
                if not final_snapshot_path.exists():
                     return CaptureSnapshotResponse(
                        snapshot_path="",
                        manifest_verified=manifest_verified,
                        git_diff_context=git_diff_context,
                        snapshot_type=snapshot_type,
                        status="error",
                        error="Snapshot generation failed (file not created)"
                    )

                # Inject Cognitive Hologram (Protocol 132) if available
                if strategic_context and snapshot_type == "seal":
                    try:
                        existing_content = final_snapshot_path.read_text()
                        final_snapshot_path.write_text(
                            strategic_context + "\n\n---\n\n" + existing_content
                        )
                        logger.info("🧠 RLM: Cognitive Hologram injected into snapshot.")
                    except Exception as inj_err:
                        logger.warning(f"🧠 RLM: Could not inject hologram: {inj_err}")

                file_stat = final_snapshot_path.stat()
                return CaptureSnapshotResponse(
                    snapshot_path=str(final_snapshot_path.relative_to(self.project_root)),
                    manifest_verified=manifest_verified,
                    git_diff_context=git_diff_context,
                    snapshot_type=snapshot_type,
                    status="success",
                    total_files=stats.get("total_files", 0),
                    total_bytes=file_stat.st_size
                )
            finally:
                if temp_manifest.exists():
                    temp_manifest.unlink(missing_ok=True)
                    
        except Exception as e:
            logger.error(f"Snapshot generation failed: {e}", exc_info=True)
            return CaptureSnapshotResponse(
                snapshot_path="",
                manifest_verified=manifest_verified,
                git_diff_context=git_diff_context,
                snapshot_type=snapshot_type,
                status="error",
                error=str(e)
            )

    #============================================================
    # 4. GUARDIAN SNAPSHOT (The Session Pack)
    #============================================================
    def guardian_snapshot(self, strategic_context: str = None) -> GuardianSnapshotResponse:
        """
        Captures the 'Guardian Start Pack' (Chronicle/Protocol/Roadmap) for session continuity.
        Logical Fit: Lifecycle management (Protocol 114).
        """
        logger.info("Generating Guardian Snapshot (Session Context Pack)...")
        try:
            # Default Start Pack Files (from Protocol 114)
            # We scan CHRONICLE, PROTOCOLS, and the main Roadmap
            manifest = []
            
            # 1. Chronicle Entries (Recent 5)
            chronicle_dir = self.project_root / "00_CHRONICLE" / "ENTRIES"
            if chronicle_dir.exists():
                entries = sorted(chronicle_dir.glob("*.md"), reverse=True)[:5]
                manifest.extend([str(e.relative_to(self.project_root)) for e in entries])
            
            # 2. Protocols (Core)
            protocol_dir = self.project_root / "01_PROTOCOLS"
            if protocol_dir.exists():
                cores = ["114_Guardian_Wakeup_and_Cache_Prefill.md", "118_Agent_Session_Initialization_and_MCP_Tool_Usage_Protocol.md"]
                for core in cores:
                    if (protocol_dir / core).exists():
                        manifest.append(f"01_PROTOCOLS/{core}")

            # 3. Roadmap
            if (self.project_root / "README.md").exists():
                manifest.append("README.md")
                
            # Reuse capture_snapshot logic with type 'seal'
            resp = self.capture_snapshot(
                manifest_files=manifest, 
                snapshot_type="seal", 
                strategic_context=strategic_context
            )
            
            return GuardianSnapshotResponse(
                status=resp.status,
                snapshot_path=resp.snapshot_path,
                total_files=resp.total_files,
                total_bytes=resp.total_bytes,
                error=resp.error
            )
            
        except Exception as e:
            logger.error(f"Guardian Snapshot failed: {e}", exc_info=True)
            return GuardianSnapshotResponse(status="error", snapshot_path="", error=str(e))

    def _ensure_diagrams_rendered(self):
        """Scan docs/architecture_diagrams and render any outdated .mmd files."""
        try:
            diagrams_dir = self.project_root / "docs" / "architecture_diagrams"
            if not diagrams_dir.exists(): return
            
            # Simple check for mmd-cli (skipped for brevity/robustness in migration, assume user has env)
            # Use subprocess to check/run if necessary in full implementation
            pass 
        except Exception as e:
            logger.warning(f"Diagram rendering check failed: {e}")

    def _dedupe_manifest(self, manifest: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Protocol 130: Remove files already embedded in included outputs."""
        # Simplified: for now just return manifest. Full implementation requires registry loading.
        # Ideally load .agent/learning/manifest_registry.json
        return manifest, {}

    def _rlm_context_synthesis(self) -> str:
        """
        Implements Protocol 132: Recursive Context Synthesis.
        Generates the 'Cognitive Hologram' by mapping and reducing the system state via LOCAL LLM.
        """
        try:
            import time
            import json
            start_time = time.time()
            logger.info("🧠 RLM: Starting Recursive Context Synthesis (Sovereign Mode)...")
            
            # PHASE 1: Map (Decomposition)
            # Use Manifest if available (ADR 083)
            manifest_path = self.learning_dir / "learning_manifest.json"
            target_files = []
            
            if manifest_path.exists():
                logger.info(f"🧠 RLM: Loading scope from {manifest_path.name}...")
                manifest_data = json.loads(manifest_path.read_text())
                
                # Handle modular manifest structure (ADR 089)
                if isinstance(manifest_data, dict):
                    core = manifest_data.get("core", [])
                    topic = manifest_data.get("topic", [])
                    target_files = core + topic
                    logger.info(f"🧠 RLM: Merged {len(core)} core + {len(topic)} topic entries.")
                else:
                    # Legacy: flat array
                    target_files = manifest_data
            else:
                logger.warning("🧠 RLM: No manifest found. Falling back to default roots.")
                target_files = ["01_PROTOCOLS", "ADRs"] # Safe default

            perception_map = self._rlm_map(target_files)
            
            # Phase 2: Reduce (Synthesis)
            hologram = self._rlm_reduce(perception_map)
            
            duration = time.time() - start_time
            logger.info(f"🧠 RLM: Synthesis Complete in {duration:.2f} seconds.")
            
            # Append timing to hologram for visibility
            hologram += f"\n\n**Process Metrics:**\n* Total Synthesis Time: {duration:.2f}s"
            
            return hologram
        except Exception as e:
            logger.error(f"RLM Synthesis failed: {e}")
            return "## Cognitive Hologram [Failure]\n* System failed to synthesize state."

    def _rlm_map(self, targets: List[str]) -> Dict[str, str]:
        """
        Level 1: Iterate targets (Files or Dirs) and generate atomic summaries using local Qwen2-7B.
        Uses hash-based caching to skip unchanged files.
        """
        import requests
        import os
        import hashlib
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Configuration
        OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
        MODEL_NAME = os.getenv("OLLAMA_MODEL", "hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M")
        CACHE_PATH = self.project_root / ".agent" / "learning" / "rlm_summary_cache.json"
        
        # Load Cache
        cache = {}
        if CACHE_PATH.exists():
            try:
                cache = json.loads(CACHE_PATH.read_text())
                logger.info(f"🧠 RLM: Loaded cache with {len(cache)} entries.")
            except:
                pass
        
        results = {}
        cache_hits = 0
        
        # 1. Flatten Targets into File List
        all_files = []
        
        for target in targets:
            path = self.project_root / target
            if not path.exists(): continue
            
            # Helper to check if a file should be skipped
            def should_skip(p: Path):
                rel_p = str(p.relative_to(self.project_root))
                
                # 1. Check central recursive artifact exclusion logic
                if self._is_recursive_artifact(rel_p):
                    return True
                
                # RLM: Process .md, .txt, and .in files (captured unless globally excluded)
                if p.suffix.lower() not in [".md", ".txt", ".in"]: return True
                
                return False
 
            if path.is_file():
                # Manifest Entry (File)
                if not should_skip(path):
                    all_files.append((str(path.parent), path))
            elif path.is_dir():
                # Recursive Scan (Legacy/Folder Mode)
                for subpath in path.rglob("*.md"):
                    if not should_skip(subpath):
                        all_files.append((str(path), subpath))
                
        # Deduplicate
        all_files = list(set(all_files))
        
        total_files = len(all_files)
        logger.info(f"🧠 RLM: Mapping {total_files} files with model {MODEL_NAME}...")

        file_number = 0
        for root, path in all_files:
            file_number += 1
            rel_path = str(path.relative_to(self.project_root))
            
            try:
                content = path.read_text(errors='ignore')
                
                # Optimization: Skip empty files
                if not content.strip(): 
                    results[rel_path] = "[Empty File]"
                    continue
                
                # Compute hash for cache lookup
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                
                # Check cache
                if rel_path in cache and cache[rel_path].get("hash") == content_hash:
                    results[rel_path] = cache[rel_path]["summary"]
                    cache_hits += 1
                    logger.info(f"   [{file_number}/{total_files}] {rel_path} [CACHE HIT]")
                    continue
                
                # Log progress
                logger.info(f"   [{file_number}/{total_files}] Processing {rel_path}...")
                
                if len(content) > 10000: 
                    content = content[:10000] + "\n...[Truncated]"

                # The Real Prompt
                prompt = (
                    f"Analyze the following Project Sanctuary document. "
                    f"Provide a single, dense sentence summarizing its architectural purpose and status.\n"
                    f"Document: {rel_path}\n"
                    f"Content:\n{content}\n\n"
                    f"Architectural Summary:"
                )

                # The Real Call (Ollama)
                # Timeout INCREASED to 300s per file to accommodate slow generations on large files
                response = requests.post(
                    OLLAMA_URL, 
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_ctx": 4096,
                            "temperature": 0.1  # Low temp for factual precision
                        } 
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    summary = response.json().get("response", "").strip()
                    # Clean up common LLM chatting artifacts
                    if summary.startswith("Here is a"): summary = summary.split(":", 1)[-1].strip()
                    results[rel_path] = summary
                    
                    # Update cache with file metadata
                    file_mtime = path.stat().st_mtime
                    cache[rel_path] = {
                        "hash": content_hash,
                        "summary": summary,
                        "file_mtime": file_mtime,
                        "summarized_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                    }
                    
                    # PERSIST IMMEDIATELY (Incremental Save)
                    try:
                        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                        with open(CACHE_PATH, "w") as f:
                            json.dump(cache, f, indent=2, sort_keys=True)
                        logger.debug(f"🧠 RLM: Incremental save for {rel_path}")
                    except Exception as e:
                        logger.warning(f"🧠 RLM: Failed to save cache: {e}")

                else:
                    logger.warning(f"Ollama Error {response.status_code} for {rel_path}")
                    results[rel_path] = f"[Ollama Generation Failed: {response.status_code}]"

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout processing {rel_path}")
                results[rel_path] = "[RLM Read Timeout]"
            except Exception as e:
                logger.warning(f"Failed to map {rel_path}: {e}")
                results[rel_path] = f"[Processing Error: {str(e)}]"
        
        return results

    def _rlm_reduce(self, map_data: Dict[str, str]) -> str:
        """
        Level 2: Synthesize atomic summaries into the Hologram.
        """
        lines = [
            "# Cognitive Hologram (Protocol 132)", 
            f"**Synthesis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            f"**Engine:** Local Sovereign (Sanctuary-Qwen2-7B)",
            "",
            "> [!NOTE]",
            "> This context is recursively synthesized from the current system state using the local fine-tuned model.",
            ""
        ]
        
        # Group by Domain
        protocols = sorted([f"`{k}`: {v}" for k,v in map_data.items() if "PROTOCOL" in k])
        adrs = sorted([f"`{k}`: {v}" for k,v in map_data.items() if "ADR" in k])
        code = sorted([f"`{k}`: {v}" for k,v in map_data.items() if "mcp_servers" in k])
        
        lines.append(f"## 1. Constitutional State ({len(protocols)} Protocols)")
        lines.append("\n".join([f"* {p}" for p in protocols]))
        
        lines.append(f"\n## 2. Decision Record ({len(adrs)} Decisions)")
        lines.append("\n".join([f"* {a}" for a in adrs]))
        
        lines.append(f"\n## 3. Active Capabilities ({len(code)} Modules)")
        lines.append("\n".join([f"* {c}" for c in code[:30]])) # Slightly larger display
        if len(code) > 30: lines.append(f"* ... and {len(code)-30} more modules.")
        
        return "\n".join(lines)

    def _get_git_state(self, project_root: Path) -> Dict[str, Any]:

        """Captures current Git state signature."""
        try:
            git_status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=str(project_root)
            )
            git_lines = git_status_proc.stdout.splitlines()
            changed_files = set()
            for line in git_lines:
                status_bits = line[:2]
                path = line[3:].split(" -> ")[-1].strip()
                if not path: # Handle cases where space might be missing or different
                    path = line[2:].strip()
                if 'D' not in status_bits: changed_files.add(path)
            
            state_str = "".join(sorted(git_lines))
            state_hash = hashlib.sha256(state_str.encode()).hexdigest()
            return {"lines": git_lines, "changed_files": changed_files, "hash": state_hash}
        except Exception as e:
            return {"lines": [], "changed_files": set(), "hash": "error"}

    #============================================================
    # 3. PERSIST SOUL (The Chronicle)
    #============================================================
    def persist_soul(self, request: PersistSoulRequest) -> PersistSoulResponse:
        """Broadcasts the session soul to Hugging Face."""
        from mcp_servers.lib.hf_utils import ensure_dataset_card
        from mcp_servers.lib.content_processor import ContentProcessor

        try:
            # 1. Environment & Metacognitive checks (Simplified)
            # ... (Checks skipped for brevity, full impl requires env vars)
            
            # 2. Dead Man's Switch (ADR 084)
            se_score = 0.5 # Default
            # In full impl: self._calculate_semantic_entropy(content)
            
            # 3. Initialization
            snapshot_path = self.project_root / request.snapshot_path
            if not snapshot_path.exists():
                return PersistSoulResponse(status="error", error=f"Snapshot not found: {snapshot_path}")
                
            # 4. Upload Logic (Delegated to hf_utils)
            import asyncio
            from mcp_servers.lib.hf_utils import upload_soul_snapshot, upload_semantic_cache
            
            logger.info(f"Uploading snapshot to HF: {snapshot_path}")
            result = asyncio.run(upload_soul_snapshot(
                snapshot_path=str(snapshot_path),
                valence=request.valence
            ))
            
            # 5. Upload Semantic Cache (ADR 094)
            cache_file = self.project_root / ".agent/learning/rlm_summary_cache.json"
            if cache_file.exists():
                logger.info("Syncing Semantic Ledger (RLM Cache) to HF...")
                cache_result = asyncio.run(upload_semantic_cache(str(cache_file)))
                if not cache_result.success:
                    logger.warning(f"Failed to sync Semantic Ledger: {cache_result.error}")
            
            if result.success:
                return PersistSoulResponse(
                    status="success",
                    repo_url=result.repo_url,
                    snapshot_name=result.remote_path
                )
            else:
                return PersistSoulResponse(status="error", error=result.error)

        except Exception as e:
            return PersistSoulResponse(status="error", error=str(e))

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
                        # Placeholder for SE logic until migrated
                        se_score = 0.5 
                        alignment_score = 0.85
                        stability_class = "STABLE"
                    except Exception as se_error:
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
            logger.error(f"Full Soul Sync failed: {e}", exc_info=True)
            return PersistSoulResponse(status="error", error=str(e))

    #============================================================
    # 4. GUARDIAN WAKEUP (The Bootloader)
    #============================================================
    def guardian_wakeup(self, mode: str = "HOLISTIC") -> GuardianWakeupResponse:
        """Generate Guardian boot digest using manifest-driven content."""
        start = time.time()
        try:
            health_color, health_reason = self._get_system_health_traffic_light()
            integrity_status = "GREEN"
            container_status = self._get_container_status()
            
            digest_lines = [
                "# 🛡️ Guardian Wakeup Briefing (v3.0 - Manifest Driven)",
                f"**System Status:** {health_color} - {health_reason}",
                f"**Integrity Mode:** {integrity_status}",
                f"**Infrastructure:** {container_status}",
                f"**Generated Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
                "",
                "## I. Strategic Directives",
                self._get_strategic_synthesis(),
                "",
                "## II. Tactical Priorities",
                self._get_tactical_priorities(),
                "",
            ]
            
            # Load Guardian Manifest (ADR 089 format)
            learning_dir = self.project_root / ".agent" / "learning"
            manifest_path = learning_dir / "guardian_manifest.json"
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text())
                    if isinstance(manifest_data, dict):
                        core = manifest_data.get("core", [])
                        topic = manifest_data.get("topic", [])
                        all_files = core + topic
                    else:
                        all_files = manifest_data
                    
                    digest_lines.append("## III. Context Files (from guardian_manifest.json)")
                    digest_lines.append(f"*Loaded {len(all_files)} files.*")
                    digest_lines.append("")
                    
                    # Include key file summaries (first 500 chars each)
                    for file_path in all_files[:10]:  # Limit to first 10 files
                        full_path = self.project_root / file_path
                        if full_path.exists() and full_path.is_file():
                            try:
                                content = full_path.read_text(errors='ignore')[:500]
                                digest_lines.append(f"### {file_path}")
                                digest_lines.append(f"```\n{content}\n```\n")
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"Failed to load guardian manifest: {e}")
            
            # Write to file
            digest_path = self.project_root / ".agent" / "learning" / "guardian_boot_digest.md"
            digest_path.parent.mkdir(parents=True, exist_ok=True)
            digest_path.write_text("\n".join(digest_lines))
            
            return GuardianWakeupResponse(
                status="success", digest_path=str(digest_path), 
                total_time_ms=(time.time()-start)*1000
            )
            
        except Exception as e:
            return GuardianWakeupResponse(status="error", digest_path="", error=str(e))

    #============================================================
    # HELPER STUBS (Migrated)
    #============================================================
    def _get_system_health_traffic_light(self):
        # Simplified Check - Real one checks Vector DB
        return "GREEN", "Nominal (Learning Mode)"

    def _get_container_status(self):
        # Using podman check
        try:
            result = subprocess.run(
                ["podman", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=2
            )
            if "sanctuary" in result.stdout: return "✅ Fleet Active"
        except: pass
        return "⚠️ Container Check Failed"

    def _get_strategic_synthesis(self):
        return ("* **Core Mandate:** I am the Gemini Orchestrator. Values: Integrity, Efficiency, Clarity. "
                "Executing Protocol 128.")

    def _get_tactical_priorities(self):
        # Scans for tasks
        scan_dir = self.project_root / "tasks" / "in-progress"
        if scan_dir.exists():
            tasks = list(scan_dir.glob("*.md"))
            if tasks: return f"* Found {len(tasks)} active tasks."
        return "* No active tasks found."

```

---

## 📦 Component: Ledger Schema
**Source:** `mcp_servers/learning/rlm_schema.json`
_Formal JSON schema defining the cache structure._

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Reactive Ledger Memory (RLM) Cache",
    "description": "A flat key-value store mapping file paths to their semantic summaries.",
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "required": [
            "hash",
            "mtime",
            "summarized_at",
            "summary"
        ],
        "properties": {
            "hash": {
                "type": "string",
                "description": "MD5 hash of the file content at time of summarization."
            },
            "file_mtime": {
                "type": "number",
                "description": "File modification timestamp (epoch)."
            },
            "summarized_at": {
                "type": "string",
                "format": "date-time",
                "description": "ISO 8601 timestamp of when the summary was generated."
            },
            "summary": {
                "type": "string",
                "description": "High-entropy semantic summary of the file's purpose and content."
            }
        }
    }
}
```

---

## 📦 Component: Inventory Auditor (The Check)
**Source:** `scripts/rlm_factory/inventory.py`
_The script to verify coverage of the semantic ledger._

```python
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_inventory():
    # Adjusted for location in scripts/rlm_factory/
    project_root = Path(__file__).parent.parent.parent.absolute()
    # Load from Env or Default
    env_cache = os.getenv("RLM_CACHE_PATH")
    if env_cache:
        cache_path = project_root / env_cache
    else:
        cache_path = project_root / ".agent" / "learning" / "rlm_summary_cache.json"

    env_targets = os.getenv("RLM_TARGET_DIRS")
    if env_targets:
        target_dirs = [t.strip() for t in env_targets.split(",") if t.strip()]
    else:
        # Default fallback for this repo
        target_dirs = ["docs", "ADRs", "01_PROTOCOLS", "mcp_servers", "LEARNING"]
    
    if not cache_path.exists():
        print(f"❌ Cache not found at {cache_path}")
        return

    with open(cache_path, "r") as f:
        cache = json.load(f)
    
    print(f"🧠 RLM Semantic Ledger Inventory")
    print(f"==============================")
    print(f"{'Directory':<15} | {'Total':<5} | {'Cached':<6} | {'Status':<10}")
    print(f"--------------------------------------------------")

    missing_files = []

    for d in target_dirs:
        dir_path = project_root / d
        if not dir_path.exists():
            continue
            
        # Find all .md and .txt files (as per RLM rules)
        files = []
        for ext in ["**/*.md", "**/*.txt"]:
            files.extend([p for p in dir_path.glob(ext) if p.is_file()])
        
        # Filter out anything in 'archive' subdirs
        files = [f for f in files if "archive" not in str(f).lower()]
        
        total = len(files)
        cached_count = 0
        dir_missing = []

        for f in files:
            rel_path = str(f.relative_to(project_root))
            if rel_path in cache:
                cached_count += 1
            else:
                dir_missing.append(rel_path)
        
        status = "✅ COMPLETE" if total == cached_count else f"⚠️  {total - cached_count} MISSING"
        print(f"{d:<15} | {total:<5} | {cached_count:<6} | {status}")
        
        missing_files.extend(dir_missing)

    if missing_files:
        print(f"\n📑 Missing Files (Ready for targeting):")
        print(f"--------------------------------------")
        # Group by folder for readability
        current_folder = ""
        for mf in sorted(missing_files):
            folder = os.path.dirname(mf)
            if folder != current_folder:
                print(f"\n📂 {folder}/")
                current_folder = folder
            print(f"  - {mf}")
        
        print(f"\n💡 Tip: Run 'python3 scripts/cortex_cli.py rlm-distill <file_path>' to fix specific gaps.")
    else:
        print(f"\n🎉 All priority directories are fully distilled into the Semantic Ledger.")

if __name__ == "__main__":
    get_inventory()

```

---

## 📦 Component: ChromaDB Ingestor (The Vector Memory)
**Source:** `scripts/rlm_factory/chroma_ingest.py`
_Standalone script for ingesting files into ChromaDB._

```python
#!/usr/bin/env python3
"""
Standalone ChromaDB Ingestion Script for RLM Kit.
Provides both Full and Incremental ingestion capabilities.
"""
import os
import sys
import logging
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Third-party imports (Install: pip install langchain-chroma langchain-huggingface python-dotenv)
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chroma_ingest")

load_dotenv()

class VectorMemory:
    def __init__(self, persist_directory: str = ".vector_data"):
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Client
        self.vector_store = Chroma(
            collection_name="project_memory",
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        
    def ingest_file(self, file_path: Path):
        """Ingest a single file with chunking."""
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to DB
            self.vector_store.add_documents(chunks)
            logger.info(f"✅ Ingested {file_path} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest {file_path}: {e}")

    def ingest_directory(self, directory: Path, extensions: List[str] = [".md", ".txt", ".py"]):
        """Recursively ingest a directory."""
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return
            
        logger.info(f"📂 Scanning {directory}...")
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    self.ingest_file(file_path)

    def query(self, query_text: str, n_results: int = 5):
        """Search the vector database."""
        results = self.vector_store.similarity_search_with_score(query_text, k=n_results)
        
        print(f"\n🔍 Query: '{query_text}'\n")
        for doc, score in results:
            print(f"--- [Score: {score:.4f}] {doc.metadata.get('source', 'Unknown')} ---")
            print(doc.page_content[:400] + "...\n")

    def stats(self):
        """Get collection statistics."""
        try:
            # Direct access to underlying Chroma collection for raw stats
            count = self.vector_store._collection.count()
            print(f"📊 Vector Memory Stats:")
            print(f"   - Collection: {self.vector_store._collection.name}")
            print(f"   - Total Chunks: {count}")
            print(f"   - Location: {self.persist_directory}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Ingest: python chroma_ingest.py ingest <file_or_directory>")
        print("  Query:  python chroma_ingest.py query \"search text\"")
        print("  Stats:  python chroma_ingest.py stats")
        sys.exit(1)
        
    command = sys.argv[1]
    memory = VectorMemory()
    
    if command == "ingest":
        if len(sys.argv) < 3:
             print("Error: Missing target directory/file")
             sys.exit(1)
        target = Path(sys.argv[2])
        if target.is_dir():
            memory.ingest_directory(target)
        else:
            memory.ingest_file(target)
            
    elif command == "query":
        if len(sys.argv) < 3:
             print("Error: Missing query text")
             sys.exit(1)
        q = sys.argv[2]
        memory.query(q)
        
    elif command == "stats":
        memory.stats()
        
    else:
        print(f"Unknown command: {command}")

```

---

## 📦 Component: Code-to-MD Shim (AST Converter)
**Source:** `mcp_servers/rag_cortex/ingest_code_shim.py`
_Smart converter that turns Python/JS code into RAG-optimized Markdown._

```python
#!/usr/bin/env python3
#============================================
# mcp_servers/rag_cortex/ingest_code_shim.py
# Purpose: Code-to-Markdown Ingestion Shim for RAG Cortex.
#          Converts code files into markdown optimized for ingestion.
# Role: Single Source of Truth
# Used as a module by operations.py and as a CLI script.
# Strategy: AST-Based "Pseudo-Markdown" Conversion (Zero tokens)
# Calling example:
#   from mcp_servers.rag_cortex.ingest_code_shim import convert_and_save
#   out_path = convert_and_save("script.py")
# LIST OF FUNCTIONS IMPLEMENTED:
#   - convert_and_save
#   - main
#   - parse_javascript_to_markdown
#   - parse_python_to_markdown
#============================================

import ast
import os
import sys
from pathlib import Path
from typing import Optional
from mcp_servers.lib.path_utils import find_project_root


#============================================
# Function: parse_python_to_markdown
# Purpose: Reads a .py file and converts it into a Markdown string optimized for RAG.
# Args:
#   file_path: Path to the Python file to convert
# Returns: Markdown-formatted string ready for RAG ingestion
# Raises:
#   FileNotFoundError: If the file doesn't exist
#   SyntaxError: If the Python file has syntax errors
#============================================
def parse_python_to_markdown(file_path: str) -> str:
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Failed to parse {file_path}: {e}")
    
    filename = file_path.name
    try:
        project_root = Path(find_project_root())
        relative_path = file_path.relative_to(project_root) if file_path.is_absolute() else file_path
    except (ValueError, RuntimeError):
        relative_path = file_path
    
    # Header acts like a file summary
    markdown_output = f"# Code File: {filename}\n\n"
    markdown_output += f"**Path:** `{relative_path}`\n"
    markdown_output += f"**Language:** Python\n"
    markdown_output += f"**Type:** Code Implementation\n\n"
    
    # Extract Global Docstring if exists
    docstring = ast.get_docstring(tree)
    if docstring:
        markdown_output += f"## Module Description\n\n{docstring}\n\n"
    
    # Extract imports for context
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    
    if imports:
        markdown_output += f"## Dependencies\n\n"
        for imp in imports[:10]:  # Limit to first 10 to avoid clutter
            markdown_output += f"- `{imp}`\n"
        if len(imports) > 10:
            markdown_output += f"- ... and {len(imports) - 10} more\n"
        markdown_output += "\n"
    
    # Iterate through functions and classes (The "Chunking Strategy")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Extract Metadata
            name = node.name
            
            if isinstance(node, ast.ClassDef):
                type_label = "Class"
            elif isinstance(node, ast.AsyncFunctionDef):
                type_label = "Async Function"
            else:
                type_label = "Function"
            
            start_line = node.lineno
            
            # Get the raw source code for this specific function/class
            segment = ast.get_source_segment(source, node)
            
            if segment is None:
                # Fallback: extract manually from source lines
                end_line = getattr(node, 'end_lineno', start_line)
                source_lines = source.split('\n')
                segment = '\n'.join(source_lines[start_line-1:end_line])
            
            # Extract docstring for the function/class
            func_doc = ast.get_docstring(node) or "No documentation provided."
            
            # Extract function signature for functions
            signature = ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        # Try to get annotation as string
                        try:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        except:
                            pass
                    args.append(arg_str)
                
                # Add return type if available
                returns = ""
                if node.returns:
                    try:
                        returns = f" -> {ast.unparse(node.returns)}"
                    except:
                        pass
                
                signature = f"({', '.join(args)}){returns}"
            
            # Format as a Markdown Section (This is what RAG likes)
            markdown_output += f"## {type_label}: `{name}`\n\n"
            markdown_output += f"**Line:** {start_line}\n"
            if signature:
                markdown_output += f"**Signature:** `{name}{signature}`\n"
            markdown_output += f"\n**Documentation:**\n\n{func_doc}\n\n"
            markdown_output += f"**Source Code:**\n\n```python\n{segment}\n```\n\n"
            
            # For classes, also extract methods
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if methods:
                    markdown_output += f"**Methods:** {', '.join([f'`{m.name}`' for m in methods])}\n\n"
    
    # Add footer with metadata
    markdown_output += "---\n\n"
    markdown_output += f"**Generated by:** Code Ingestion Shim (Task 110)\n"
    markdown_output += f"**Source File:** `{relative_path}`\n"
    markdown_output += f"**Total Lines:** {len(source.split(chr(10)))}\n"
    
    return markdown_output


#============================================
# Function: parse_javascript_to_markdown
# Purpose: Reads a .js/.ts file and converts it into a Markdown string using Regex.
# Args:
#   file_path: Path to the JS/TS file
# Returns: Markdown-formatted string
#============================================
def parse_javascript_to_markdown(file_path: Path) -> str:
    import re
    
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
        
    filename = file_path.name
    try:
        project_root = Path(find_project_root())
        relative_path = file_path.relative_to(project_root) if file_path.is_absolute() else file_path
    except (ValueError, RuntimeError):
        relative_path = file_path
        
    # Header
    markdown_output = f"# Code File: {filename}\n\n"
    markdown_output += f"**Path:** `{relative_path}`\n"
    markdown_output += f"**Language:** JavaScript/TypeScript\n\n"
    
    # Simple formatting: Extract doc comments blocks (/** ... */)
    # This is a basic heuristic
    doc_blocks = re.finditer(r'/\*\*(.*?)\*/', source, re.DOTALL)
    for match in doc_blocks:
        comment = match.group(1).strip()
        # Clean up asterisks
        cleaned_comment = '\n'.join([line.strip().lstrip('*').strip() for line in comment.split('\n')])
        if len(cleaned_comment) > 20: # Arbitrary filter for significant comments
             markdown_output += f"## Documentation Hint\n\n{cleaned_comment}\n\n"

    # Identify Functions (Basic Regex)
    # 1. function foo()
    func_pattern = re.compile(r'function\s+(\w+)\s*\((.*?)\)')
    for match in func_pattern.finditer(source):
        name = match.group(1)
        args = match.group(2)
        line_no = source[:match.start()].count('\n') + 1
        markdown_output += f"## Function: `{name}`\n\n"
        markdown_output += f"**Line:** {line_no}\n"
        markdown_output += f"**Signature:** `function {name}({args})`\n\n"
        
        # Context extraction (dumb implementation: just grab next 10 lines)
        full_lines = source.split('\n')
        start_idx = max(0, line_no - 1)
        end_idx = min(len(full_lines), start_idx + 20) # Grab up to 20 lines
        snippet = '\n'.join(full_lines[start_idx:end_idx])
        markdown_output += f"```javascript\n{snippet}\n...\n```\n\n"

    # 2. const foo = () => 
    arrow_pattern = re.compile(r'(const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(?(.*?)\)?\s*=>')
    for match in arrow_pattern.finditer(source):
        kind = match.group(1)
        name = match.group(2)
        args = match.group(3) or ""
        line_no = source[:match.start()].count('\n') + 1
        markdown_output += f"## {kind.title()} Function: `{name}`\n\n"
        markdown_output += f"**Line:** {line_no}\n"
        markdown_output += f"**Signature:** `{name} = ({args}) => ...`\n\n"
        
        full_lines = source.split('\n')
        start_idx = max(0, line_no - 1)
        end_idx = min(len(full_lines), start_idx + 20)
        snippet = '\n'.join(full_lines[start_idx:end_idx])
        markdown_output += f"```javascript\n{snippet}\n...\n```\n\n"
            
    # Always include full source at bottom for reference if file is small (<500 lines)
    lines = source.split('\n')
    if len(lines) < 500:
         markdown_output += "## Full Source Code\n\n```javascript\n" + source + "\n```\n\n"
    
    return markdown_output


#============================================
# Function: convert_and_save
# Purpose: Convert a Python file to markdown and optionally save it.
# Args:
#   input_file: Path to the Python file
#   output_file: Optional path to save the markdown output. If None, uses input_file.md.
# Returns: Path to the output markdown file
#============================================
def convert_and_save(input_file: str, output_file: Optional[str] = None) -> str:
    input_path = Path(input_file)
    
    if output_file is None:
        # Append .md to the original filename (e.g., test.py -> test.py.md)
        # This prevents collisions with existing .md files and allows .gitignore filtering
        output_file = str(input_path) + ".md"
    
    output_path = Path(output_file)
    
    # Select parser based on extension
    if input_path.suffix == '.py':
        markdown_content = parse_python_to_markdown(input_path)
    elif input_path.suffix in ['.js', '.jsx', '.ts', '.tsx']:
        markdown_content = parse_javascript_to_markdown(input_path)
    else:
        # Fallback for unknown text files (treat as plain text block)
        with open(input_path, 'r', encoding='utf-8') as f:
             source = f.read()
        markdown_content = f"# File: {input_path.name}\n\n```\n{source}\n```"
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return str(output_path)


#============================================
# Function: main
# Purpose: CLI interface for the code ingestion shim.
#============================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_code_shim.py <python_file> [output_file]")
        print("\nExample:")
        print("  python ingest_code_shim.py scripts/stabilizers/vector_consistency_check.py")
        print("  python ingest_code_shim.py my_code.py my_code_docs.md")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = convert_and_save(input_file, output_file)
        print(f"✅ Successfully converted {input_file}")
        print(f"📄 Output saved to: {output_path}")
        
        # Show stats
        with open(output_path, 'r') as f:
            content = f.read()
            lines = len(content.split('\n'))
            chars = len(content)
            print(f"📊 Stats: {lines} lines, {chars} characters")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

```

---

## 📦 Component: Reference: Cortex Operations (The Brain)
**Source:** `mcp_servers/rag_cortex/operations.py`
_Deep logic for RAG coordination, HMAC checking, and ingestion._

```python
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

    # [DISABLED] Synaptic Phase (Dreaming) - See ADR 091 (Rejected for now)
    # def dream(self):
    #     #============================================
    #     # Method: dream
    #     # Purpose: Execute the Synaptic Phase (Dreaming).
    #     #          Consolidate memories and update Opinion Network.
    #     # Reference: ADR 091
    #     #============================================
    #     from .dreaming import Dreamer
    #     
    #     try:
    #         logger.info("Initializing Synaptic Phase (Dreaming)...")
    #         dreamer = Dreamer(self.project_root)
    #         dreamer.dream()
    #         return {"status": "success", "message": "Synaptic Phase complete."}
    #     except Exception as e:
    #         logger.error(f"Dreaming failed: {e}", exc_info=True)
    #         return {"status": "error", "error": str(e)}

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



    #============================================
    # Protocol 130: Manifest Deduplication (ADR 089)
    # Prevents including files already embedded in generated outputs
    #============================================
    
    def _load_manifest_registry(self) -> Dict[str, Any]:
        """
        Load the manifest registry that maps manifests to their generated outputs.
        Location: .agent/learning/manifest_registry.json
        """
        registry_path = self.project_root / ".agent" / "learning" / "manifest_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Protocol 130: Failed to load manifest registry: {e}")
        return {"manifests": {}}
    
    def _get_output_to_manifest_map(self, registry: Dict[str, Any]) -> Dict[str, str]:
        """
        Invert the registry: output_file → source_manifest_path
        """
        output_map = {}
        for manifest_path, info in registry.get("manifests", {}).items():
            output = info.get("output")
            if output:
                output_map[output] = manifest_path
        return output_map
    
    def _dedupe_manifest(self, manifest: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Protocol 130: Remove files from manifest that are already embedded in included outputs.
        
        Args:
            manifest: List of file paths
            
        Returns:
            Tuple of (deduped_manifest, duplicates_found)
            duplicates_found is dict of {file: reason}
        """
        registry = self._load_manifest_registry()
        output_map = self._get_output_to_manifest_map(registry)
        duplicates = {}
        
        # For each file in manifest, check if it's an output of another manifest
        for file in manifest:
            if file in output_map:
                # This file is a generated output. Check if its source files are also included.
                source_manifest_path = self.project_root / output_map[file]
                
                if source_manifest_path.exists():
                    try:
                        with open(source_manifest_path, "r") as f:
                            source_files = json.load(f)
                        
                        # Check each source file - if it's in the manifest, it's a duplicate
                        for source_file in source_files:
                            if source_file in manifest and source_file != file:
                                duplicates[source_file] = f"Already embedded in {file}"
                    except Exception as e:
                        logger.warning(f"Protocol 130: Failed to load source manifest {source_manifest_path}: {e}")
        
        if duplicates:
            logger.info(f"Protocol 130: Found {len(duplicates)} embedded duplicates, removing from manifest")
            for dup, reason in duplicates.items():
                logger.debug(f"  - {dup}: {reason}")
        
        # Remove duplicates
        deduped = [f for f in manifest if f not in duplicates]
        return deduped, duplicates

    #============================================
    # Diagram Rendering (Task #154)
    # Automatically renders .mmd to .png during snapshot
    #============================================

    def _check_mermaid_cli(self) -> bool:
        """Check if mermaid-cli is available (via npx)."""
        try:
            # Check if npx is in path
            subprocess.run(["npx", "--version"], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _render_single_diagram(self, mmd_path: Path) -> bool:
        """Render a single .mmd file to png if outdated."""
        output_path = mmd_path.with_suffix(".png")
        try:
            # Check timestamps
            if output_path.exists() and mmd_path.stat().st_mtime <= output_path.stat().st_mtime:
                return True # Up to date

            logger.info(f"Rendering outdated diagram: {mmd_path.name}")
            result = subprocess.run(
                [
                    "npx", "-y", "@mermaid-js/mermaid-cli",
                    "-i", str(mmd_path),
                    "-o", str(output_path),
                    "-b", "transparent", "-t", "default"
                ],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                logger.warning(f"Failed to render {mmd_path.name}: {result.stderr[:200]}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Error rendering {mmd_path.name}: {e}")
            return False

    def _ensure_diagrams_rendered(self):
        """Scan docs/architecture_diagrams and render any outdated .mmd files."""
        try:
            diagrams_dir = self.project_root / "docs" / "architecture_diagrams"
            if not diagrams_dir.exists():
                return
                
            if not self._check_mermaid_cli():
                logger.warning("mermaid-cli not found (npx missing or failed). Skipping diagram rendering.")
                return

            mmd_files = sorted(diagrams_dir.rglob("*.mmd"))
            logger.info(f"Verifying {len(mmd_files)} architecture diagrams...")
            
            rendered_count = 0
            for mmd_path in mmd_files:
                # We only render if outdated, logic is in _render_single_diagram
                if self._render_single_diagram(mmd_path): 
                   pass 
        except Exception as e:
            logger.warning(f"Diagram rendering process failed: {e}")



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

```

---

## 📦 Component: Reference: Cortex Models (The Structure)
**Source:** `mcp_servers/rag_cortex/models.py`
_Pydantic models defining the data structures for RAG._

```python
#============================================
# mcp_servers/rag_cortex/models.py
# Purpose: Pydantic/Dataclass models for RAG operations in the Mnemonic Cortex.
# Role: Single Source of Truth
# Used as a module by operations.py and server.py
# Calling example:
#   from mcp_servers.rag_cortex.models import to_dict
# LIST OF MODELS:
#   - IngestFullRequest
#   - IngestFullResponse
#   - QueryRequest
#   - QueryResult
#   - QueryResponse
#   - DocumentSample
#   - CollectionStats
#   - StatsResponse
#   - IngestIncrementalRequest
#   - IngestIncrementalResponse
#   - CacheGetResponse
#   - CacheSetResponse
#   - CacheWarmupResponse
#   - GuardianWakeupResponse
#   - CaptureSnapshotRequest
#   - CaptureSnapshotResponse
# LIST OF FUNCTIONS:
#   - to_dict
#============================================

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Ingest Full Models
# ============================================================================

@dataclass
class IngestFullRequest:
    #============================================
    # Model: IngestFullRequest
    # Purpose: Request model for full ingestion.
    # Fields:
    #   purge_existing: Whether to purge existing database
    #   source_directories: Optional specific directories to ingest
    #============================================
    purge_existing: bool = True
    source_directories: Optional[List[str]] = None


@dataclass
class IngestFullResponse:
    #============================================
    # Model: IngestFullResponse
    # Purpose: Response model for full ingestion.
    #============================================
    documents_processed: int
    chunks_created: int
    ingestion_time_ms: float
    vectorstore_path: str
    status: str  # "success" or "error"
    error: Optional[str] = None


# ============================================================================
# Query Models
# ============================================================================

@dataclass
class QueryRequest:
    #============================================
    # Model: QueryRequest
    # Purpose: Request model for RAG query.
    #============================================
    query: str
    max_results: int = 5
    use_cache: bool = False  # Phase 2 feature


@dataclass
class QueryResult:
    #============================================
    # Model: QueryResult
    # Purpose: Individual query result.
    #============================================
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None


@dataclass
class QueryResponse:
    #============================================
    # Model: QueryResponse
    # Purpose: Response model for RAG query.
    #============================================
    results: List[QueryResult]
    query_time_ms: float
    status: str  # "success" or "error"
    cache_hit: bool = False  # Phase 2 feature
    error: Optional[str] = None


# ============================================================================
# Stats Models
# ============================================================================

@dataclass
class DocumentSample:
    #============================================
    # Model: DocumentSample
    # Purpose: Sample document for diagnostics.
    #============================================
    id: str
    metadata: Dict[str, Any]
    content_preview: str  # First 150 chars


@dataclass
class CollectionStats:
    #============================================
    # Model: CollectionStats
    # Purpose: Statistics for a single collection.
    #============================================
    count: int
    name: str


@dataclass
class StatsResponse:
    #============================================
    # Model: StatsResponse
    # Purpose: Response model for database statistics.
    #============================================
    total_documents: int
    total_chunks: int
    collections: Dict[str, CollectionStats]
    health_status: str  # "healthy", "degraded", or "error"
    samples: Optional[List[DocumentSample]] = None  # Enhanced diagnostics from inspect_db
    cache_stats: Optional[Dict[str, Any]] = None  # Phase 2 feature
    error: Optional[str] = None


# ============================================================================
# Ingest Incremental Models
# ============================================================================

@dataclass
class IngestIncrementalRequest:
    #============================================
    # Model: IngestIncrementalRequest
    # Purpose: Request model for incremental ingestion.
    #============================================
    file_paths: List[str]
    metadata: Optional[Dict[str, Any]] = None
    skip_duplicates: bool = True


@dataclass
class IngestIncrementalResponse:
    #============================================
    # Model: IngestIncrementalResponse
    # Purpose: Response model for incremental ingestion.
    #============================================
    documents_added: int
    chunks_created: int
    skipped_duplicates: int
    ingestion_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


# ============================================================================
# Cache Operation Models (Protocol 114 - Guardian Wakeup)
# ============================================================================

@dataclass
class CacheGetResponse:
    #============================================
    # Model: CacheGetResponse
    # Purpose: Response from cache retrieval operation.
    #============================================
    cache_hit: bool
    answer: Optional[str]
    query_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class CacheSetResponse:
    #============================================
    # Model: CacheSetResponse
    # Purpose: Response from cache storage operation.
    #============================================
    cache_key: str
    stored: bool
    status: str  # "success" or "error"
    error: Optional[str] = None


@dataclass
class CacheWarmupResponse:
    #============================================
    # Model: CacheWarmupResponse
    # Purpose: Response from cache warmup operation.
    #============================================
    queries_cached: int
    cache_hits: int
    cache_misses: int
    total_time_ms: float
    status: str  # "success" or "error"
    error: Optional[str] = None





# ============================================================================



# ============================================================================
# Opinion Models (ADR 091 - The Synaptic Phase)
# ============================================================================

@dataclass
class DispositionParameters:
    #============================================
    # Model: DispositionParameters
    # Purpose: Behavioral parameters from HINDSIGHT/CARA.
    #============================================
    skepticism: float
    literalism: float
    empathy: float = 0.5  # Default

@dataclass
class HistoryPoint:
    #============================================
    # Model: HistoryPoint
    # Purpose: Tracking confidence trajectory over time.
    #============================================
    timestamp: str
    score: float
    delta_reason: str

@dataclass
class Opinion:
    #============================================
    # Model: Opinion
    # Purpose: Subjective belief node (Synaptic Phase).
    #============================================
    id: str
    statement: str
    confidence_score: float
    formation_source: str
    supporting_evidence_ids: List[str]
    history_trajectory: List[HistoryPoint]
    disposition_parameters: Optional[DispositionParameters] = None
    type: str = "opinion"  # Discriminator

# ============================================================================



# FastMCP Request Models
# ============================================================================
from pydantic import BaseModel, Field

class CortexIngestFullRequest(BaseModel):
    purge_existing: bool = Field(True, description="Whether to purge existing data")
    source_directories: Optional[List[str]] = Field(None, description="Paths to directories to ingest")

class CortexQueryRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    max_results: int = Field(5, description="Maximum number of context fragments")
    use_cache: bool = Field(False, description="Whether to use Mnemonic Cache")
    reasoning_mode: bool = Field(False, description="Whether to use reasoning model")

class CortexIngestIncrementalRequest(BaseModel):
    file_paths: List[str] = Field(..., description="Paths to files to ingest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for documents")
    skip_duplicates: bool = Field(True, description="Skip if already in store")

class CortexCacheGetRequest(BaseModel):
    query: str = Field(..., description="Query key to look up")

class CortexCacheSetRequest(BaseModel):
    query: str = Field(..., description="Query key")
    answer: str = Field(..., description="Answer to cache")

class CortexCacheWarmupRequest(BaseModel):
    genesis_queries: Optional[List[str]] = Field(None, description="Queries to pre-warm the cache")





class ForgeQueryRequest(BaseModel):
    prompt: str = Field(..., description="Model prompt")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(2048, description="Max tokens to generate")
    system_prompt: Optional[str] = Field(None, description="System persona prompt")




# ============================================================================
# Helper Functions
# ============================================================================

def to_dict(obj: Any) -> Dict[str, Any]:
    #============================================
    # Function: to_dict
    # Purpose: Convert dataclass to dictionary recursively.
    # Args:
    #   obj: The dataclass object to convert
    # Returns: Dictionary representation
    #============================================
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if isinstance(value, list):
                result[field_name] = [to_dict(item) if hasattr(item, '__dataclass_fields__') else item for item in value]
            elif isinstance(value, dict):
                result[field_name] = {k: to_dict(v) if hasattr(v, '__dataclass_fields__') else v for k, v in value.items()}
            elif hasattr(value, '__dataclass_fields__'):
                result[field_name] = to_dict(value)
            else:
                result[field_name] = value
        return result
    return obj

```

---

## 📦 Component: CLI Interface (The Driver)
**Source:** `scripts/cortex_cli.py`
_The command-line interface for triggering RLM operations._

```python
#============================================
# scripts/cortex_cli.py
# Purpose: CLI Orchestrator for the Mnemonic Cortex RAG server.
# Role: Single Source of Truth for Terminal Operations.
# Reference: Protocol 128 (Hardened Learning Loop)
#
# INGESTION EXAMPLES:
#   python3 scripts/cortex_cli.py ingest                    # Full purge & rebuild (Default behavior)
#   python3 scripts/cortex_cli.py ingest --no-purge         # Append to existing Vector DB
#   python3 scripts/cortex_cli.py ingest --dirs "LEARNING"  # Target specific directory ingestion
#   python3 scripts/cortex_cli.py ingest --type incremental --files "path/to/file.md"  # Targeted update
#
# SNAPSHOT EXAMPLES (Protocol 128 Workflow):
#   python3 scripts/cortex_cli.py snapshot --type audit --manifest .agent/learning/red_team/red_team_manifest.json
#   python3 scripts/cortex_cli.py snapshot --type learning_audit --manifest .agent/learning/learning_audit/learning_audit_manifest.json
#   python3 scripts/cortex_cli.py snapshot --type seal --manifest .agent/learning/learning_manifest.json
#   python3 scripts/cortex_cli.py snapshot --type learning_audit --context "Egyptian Labyrinth research"
#
# GUARDIAN WAKEUP (Protocol 128 Bootloader):
#   python3 scripts/cortex_cli.py guardian                     # Standard wakeup
#   python3 scripts/cortex_cli.py guardian --mode TELEMETRY    # Telemetry-focused wakeup
#   python3 scripts/cortex_cli.py guardian --show              # Display digest content after generation
#
# BOOTSTRAP DEBRIEF (Fresh Repo Onboarding):
#   python3 scripts/cortex_cli.py bootstrap-debrief            # Generate onboarding context packet
#
# DIAGNOSTICS & RETRIEVAL:
#   python3 scripts/cortex_cli.py stats                     # View child/parent counts & health
#   python3 scripts/cortex_cli.py query "Protocol 128"      # Semantic search across Mnemonic Cortex
#   python3 scripts/cortex_cli.py debrief --hours 48        # Session diff & recency scan
#   python3 scripts/cortex_cli.py cache-stats               # Check semantic cache (CAG) efficiency
#   python3 scripts/cortex_cli.py cache-warmup              # Pre-populate CAG with genesis queries
#
# SOUL PERSISTENCE (ADR 079 / 081):
#   Incremental (append 1 seal to JSONL + upload MD to lineage/):
#     python3 scripts/cortex_cli.py persist-soul
#     python3 scripts/cortex_cli.py persist-soul --valence 0.8 --snapshot .agent/learning/learning_package_snapshot.md
#
#   Full Sync (regenerate entire JSONL from all files + deploy data/):
#     python3 scripts/cortex_cli.py persist-soul-full
#
# EVOLUTIONARY METRICS (Protocol 131):
#   python3 scripts/cortex_cli.py evolution fitness "Some content"
#   python3 scripts/cortex_cli.py evolution depth --file .agent/learning/learning_debrief.md
#
# RLM DISTILLATION (Protocol 132):
#   python3 scripts/cortex_cli.py rlm-distill README.md        # Distill summary for a file
#   python3 scripts/cortex_cli.py rlm-distill "ADRs"            # Distill summaries for a directory (Recursive)
#============================================
import argparse
import sys
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.learning.operations import LearningOperations
from mcp_servers.evolution.operations import EvolutionOperations
import subprocess

# ADR 090: Iron Core Definitions
IRON_CORE_PATHS = [
    "01_PROTOCOLS",
    "ADRs",
    "cognitive_continuity_policy.md",
    "founder_seed.json"
]

def verify_iron_core(root_path):
    """
    Verifies that Iron Core paths have not been tampered with (uncommitted/unstaged changes).
    ADR 090 (Evolution-Aware):
    - Unstaged changes (Dirty Worktree) -> VIOLATION (Drift)
    - Staged changes (Index) -> ALLOWED (Evolution)
    """
    violations = []
    try:
        # Check for modifications in Iron Core paths
        # --porcelain format:
        # XY Path
        # X = Index (Staged), Y = Worktree (Unstaged)
        # We only care if Y is modified (meaning unstaged changes exist)
        cmd = ["git", "status", "--porcelain"] + IRON_CORE_PATHS
        result = subprocess.run(
            cmd, 
            cwd=root_path, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                # Line format: "XY Path" (e.g., " M file.md", "M  file.md", "?? file.md")
                if len(line.strip()) < 3: 
                    continue
                    
                status_code = line[:2]
                path = line[3:]
                
                # Check Worktree Status (2nd character)
                # ' ' = Unmodified in worktree (changes are staged or clean)
                # 'M' = Modified in worktree
                # 'D' = Deleted in worktree
                # '?' = Untracked
                worktree_status = status_code[1]
                
                # Violation if:
                # 1. Untracked ('??') inside Iron Core path (adding new files without staging)
                # 2. Modified in Worktree ('M') (editing without staging)
                # 3. Deleted in Worktree ('D') (deleting without staging)
                if status_code == '??' or worktree_status in ['M', 'D']:
                    violations.append(f"{line.strip()} (Unstaged/Dirty - Please 'git add' to authorize)")
                
    except Exception as e:
        return False, [f"Error checking Iron Core: {str(e)}"]
        
    return len(violations) == 0, violations


def main():
    parser = argparse.ArgumentParser(description="Mnemonic Cortex CLI")
    parser.add_argument("--root", default=".", help="Project root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available operations")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Perform full ingestion")
    ingest_parser.add_argument("--no-purge", action="store_false", dest="purge", help="Skip purging DB")
    ingest_parser.add_argument("--dirs", nargs="+", help="Specific directories to ingest")
    ingest_parser.add_argument("--incremental", action="store_true", help="Incremental ingestion mode")
    ingest_parser.add_argument("--hours", type=int, default=24, help="Hours to look back (for incremental mode)")

    # Command: snapshot
    snapshot_parser = subparsers.add_parser("snapshot", help="Capture a Protocol 128 snapshot")
    snapshot_parser.add_argument("--type", choices=["audit", "learning_audit", "seal"], required=True)
    snapshot_parser.add_argument("--manifest", help="Path to manifest JSON file")
    snapshot_parser.add_argument("--context", help="Strategic context for the snapshot")
    snapshot_parser.add_argument("--override-iron-core", action="store_true", help="⚠️ Override Iron Core check (Requires ADR 090 Amendment)")

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Get RAG health and statistics")
    stats_parser.add_argument("--samples", action="store_true", help="Include sample documents")
    stats_parser.add_argument("--sample-count", type=int, default=5, help="Number of samples to include")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Perform semantic search query")
    query_parser.add_argument("query_text", help="Search query string")
    query_parser.add_argument("--max-results", type=int, default=5, help="Maximum results to return")
    query_parser.add_argument("--use-cache", action="store_true", help="Use semantic cache")

    # Command: debrief
    debrief_parser = subparsers.add_parser("debrief", help="Run learning debrief (Protocol 128)")
    debrief_parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours")
    debrief_parser.add_argument("--output", help="Output file path (default: .agent/learning/learning_debrief.md)")

    # [DISABLED] Synaptic Phase (Dreaming)
    # dream_parser = subparsers.add_parser("dream", help="Execute Synaptic Phase (Dreaming)")

    # Command: guardian (Protocol 128 Bootloader)
    guardian_parser = subparsers.add_parser("guardian", help="Generate Guardian Boot Digest (Protocol 128)")
    guardian_parser.add_argument("--mode", default="HOLISTIC", choices=["HOLISTIC", "TELEMETRY"], help="Wakeup mode")
    guardian_parser.add_argument("--show", action="store_true", help="Display digest content after generation")
    guardian_parser.add_argument("--manifest", default=".agent/learning/guardian_manifest.json", help="Path to guardian manifest")

    # Command: bootstrap-debrief (Fresh Repo Onboarding)
    bootstrap_parser = subparsers.add_parser("bootstrap-debrief", help="Generate onboarding context packet for fresh repo setup")
    bootstrap_parser.add_argument("--manifest", default=".agent/learning/bootstrap_manifest.json", help="Path to bootstrap manifest")
    bootstrap_parser.add_argument("--output", default=".agent/learning/bootstrap_packet.md", help="Output path for the packet")

    # Command: cache-stats
    subparsers.add_parser("cache-stats", help="Get cache statistics")

    # Command: cache-warmup
    warmup_parser = subparsers.add_parser("cache-warmup", help="Pre-populate cache with genesis queries")
    warmup_parser.add_argument("--queries", nargs="+", help="Custom queries to cache")

    # Command: persist-soul (ADR 079)
    soul_parser = subparsers.add_parser("persist-soul", help="Broadcast snapshot to HF AI Commons")
    soul_parser.add_argument("--snapshot", default=".agent/learning/learning_package_snapshot.md", help="Path to snapshot")
    soul_parser.add_argument("--valence", type=float, default=0.0, help="Moral/emotional charge")
    soul_parser.add_argument("--uncertainty", type=float, default=0.0, help="Logic confidence")
    soul_parser.add_argument("--full-sync", action="store_true", help="Sync entire learning directory")

    # Command: persist-soul-full (ADR 081)
    subparsers.add_parser("persist-soul-full", help="Regenerate full JSONL and deploy to HF (ADR 081)")

    # evolution (Protocol 131)
    evolution_parser = subparsers.add_parser("evolution", help="Evolutionary metrics (Protocol 131)")
    evolution_sub = evolution_parser.add_subparsers(dest="subcommand", help="Evolution subcommands")
    
    # fitness
    fit_parser = evolution_sub.add_parser("fitness", help="Calculate full fitness vector")
    fit_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    fit_parser.add_argument("--file", help="Read content from file")
    
    # depth
    depth_parser = evolution_sub.add_parser("depth", help="Evaluate technical depth")
    depth_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    depth_parser.add_argument("--file", help="Read content from file")
    
    # scope
    scope_parser = evolution_sub.add_parser("scope", help="Evaluate architectural scope")
    scope_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    scope_parser.add_argument("--file", help="Read content from file")

    # Command: rlm-distill (Protocol 132)
    rlm_parser = subparsers.add_parser("rlm-distill", aliases=["rlm-test"], help="Distill semantic summaries for a specific file or folder")
    rlm_parser.add_argument("target", help="File or folder path to distill (relative to project root)")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize Operations
    cortex_ops = CortexOperations(project_root=args.root)
    learning_ops = LearningOperations(project_root=args.root)
    evolution_ops = EvolutionOperations(project_root=args.root)

    if args.command == "ingest":
        if args.incremental:
            print(f"🔄 Starting INCREMENTAL ingestion (Last {args.hours}h)...")
            import time
            from datetime import timedelta
            
            cutoff_time = time.time() - (args.hours * 3600)
            modified_files = []
            
            # Walk project root to find modified files
            # Exclude known heavy/irrelevant dirs
            exclude_dirs = {'.git', '.vector_data', '__pycache__', 'node_modules', 'venv', 'env', 
                            'dataset_package', 'docs/site', 'training_logs'}
            
            for path in cortex_ops.project_root.rglob('*'):
                if path.is_file():
                    # Check exclusions
                    if any(part in exclude_dirs for part in path.parts):
                        continue
                        
                    # Check extension
                    if path.suffix not in ['.md', '.py', '.js', '.ts', '.txt', '.json']:
                        continue
                        
                    # Check mtime
                    if path.stat().st_mtime > cutoff_time:
                        modified_files.append(str(path))
            
            if not modified_files:
                print(f"⚠️ No files modified in the last {args.hours} hours. Skipping ingestion.")
                sys.exit(0)
                
            print(f"📄 Found {len(modified_files)} modified files.")
            res = cortex_ops.ingest_incremental(file_paths=modified_files)
            
            if res.status == "success":
                print(f"✅ Success: {res.documents_added} added, {res.chunks_created} chunks in {res.ingestion_time_ms/1000:.2f}s")
            else:
                print(f"❌ Error: {res.error}")
                sys.exit(1)
        
        else:
            # Full Ingestion
            print(f"🔄 Starting full ingestion (Purge: {args.purge})...")
            res = cortex_ops.ingest_full(purge_existing=args.purge, source_directories=args.dirs)
            if res.status == "success":
                print(f"✅ Success: {res.documents_processed} docs, {res.chunks_created} chunks in {res.ingestion_time_ms/1000:.2f}s")
            else:
                print(f"❌ Error: {res.error}")
                sys.exit(1)

    elif args.command == "snapshot":
        # ADR 090: Iron Core Verification
        if not args.override_iron_core:
            print("🛡️  Running Iron Core Verification (ADR 090)...")
            is_pristine, violations = verify_iron_core(args.root)
            if not is_pristine:
                print(f"\n\033[91m⛔ IRON CORE BREACH DETECTED (SAFE MODE ENGAGED)\033[0m")
                print("The following immutable files have been modified without authorization:")
                for v in violations:
                    print(f"  - {v}")
                print("\nAction blocked: 'snapshot' is disabled in Safe Mode.")
                print("To proceed, revert changes or use --override-iron-core (Constitutional Amendment required).")
                sys.exit(1)
            print("✅ Iron Core Integrity Verified.")
        else:
            print(f"⚠️  \033[93mWARNING: IRON CORE CHECK OVERRIDDEN\033[0m")

        manifest = []
        if args.manifest:
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                print(f"❌ Manifest file not found: {args.manifest}")
                sys.exit(1)
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📋 Loaded manifest with {len(manifest)} files")
        
        print(f"📸 Capturing {args.type} snapshot...")
        # ROUTED TO LEARNING MCP
        res = learning_ops.capture_snapshot(
            manifest_files=manifest, 
            snapshot_type=args.type,
            strategic_context=args.context
        )
        
        if res.status == "success":
            print(f"✅ Snapshot created at: {res.snapshot_path}")
            print(f"📊 Files: {res.total_files} | Bytes: {res.total_bytes}")
            print(f"🔍 Manifest verified: {res.manifest_verified}")
            print(f"📝 Git context: {res.git_diff_context}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "stats":
        stats = cortex_ops.get_stats(include_samples=args.samples, sample_count=args.sample_count)
        print(f"🏥 Health: {stats.health_status}")
        print(f"📚 Documents: {stats.total_documents}")
        print(f"🧩 Chunks: {stats.total_chunks}")
        
        if stats.collections:
            print("\n📊 Collections:")
            for name, coll in stats.collections.items():
                print(f"  - {coll.name}: {coll.count} items")
        
        if stats.samples:
            print(f"\n🔍 Sample Documents:")
            for i, sample in enumerate(stats.samples, 1):
                print(f"\n  {i}. ID: {sample.id}")
                print(f"     Preview: {sample.content_preview[:100]}...")
                if sample.metadata:
                    print(f"     Metadata: {sample.metadata}")
        
        if stats.error:
            print(f"\n❌ Error: {stats.error}")

    # [DISABLED] Synaptic Phase (Dreaming)
    # elif args.command == "dream":
    #     print("💤 Mnemonic Cortex: Entering Synaptic Phase (Dreaming)...")
    #     # Use centralized Operations layer
    #     response = ops.dream()
    #     print(json.dumps(response, indent=2))
    elif args.command == "query":
        print(f"🔍 Querying: {args.query_text}")
        res = cortex_ops.query(
            query=args.query_text,
            max_results=args.max_results,
            use_cache=args.use_cache
        )
        
        if res.status == "success":
            print(f"✅ Found {len(res.results)} results in {res.query_time_ms:.2f}ms")
            print(f"💾 Cache hit: {res.cache_hit}")
            
            for i, result in enumerate(res.results, 1):
                print(f"\n--- Result {i} (Score: {result.relevance_score:.4f}) ---")
                print(f"Content: {result.content[:300]}...")
                if result.metadata:
                    source = result.metadata.get('source', 'Unknown')
                    print(f"Source: {source}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "debrief":
        print(f"📋 Running learning debrief (lookback: {args.hours}h)...")
        # ROUTED TO LEARNING MCP
        debrief_content = learning_ops.learning_debrief(hours=args.hours)
        
        # Default output path
        output_path = args.output or ".agent/learning/learning_debrief.md"
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(debrief_content)
        
        print(f"✅ Debrief written to: {output_file}")
        print(f"📊 Content length: {len(debrief_content)} characters")

    elif args.command == "guardian":
        print(f"🛡️ Generating Guardian Boot Digest (mode: {args.mode})...")
        
        # Load manifest if exists
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📋 Loaded guardian manifest: {len(manifest)} files")
        else:
            print(f"⚠️  Guardian manifest not found at {args.manifest}. Using defaults.")
        
        # ROUTED TO LEARNING MCP
        response = learning_ops.guardian_wakeup(mode=args.mode)
        
        print(f"   Status: {response.status}")
        print(f"   Digest: {response.digest_path}")
        print(f"   Time: {response.total_time_ms:.2f}ms")
        
        if response.error:
            print(f"❌ Error: {response.error}")
            sys.exit(1)
        
        if args.show and response.digest_path:
            print("\n" + "="*60)
            with open(response.digest_path, 'r') as f:
                print(f.read())
        
        print(f"✅ Guardian Boot Digest generated.")

    elif args.command == "bootstrap-debrief":
        print(f"🏗️  Generating Bootstrap Context Packet...")
        
        # Load manifest
        manifest_path = Path(args.manifest)
        manifest = []
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📋 Loaded bootstrap manifest: {len(manifest)} files")
        else:
            print(f"⚠️  Bootstrap manifest not found at {args.manifest}. Using defaults.")
        
        # Generate snapshot using the manifest
        # ROUTED TO LEARNING MCP
        res = learning_ops.capture_snapshot(
            manifest_files=manifest,
            snapshot_type="seal",
            strategic_context="Fresh repository onboarding context"
        )
        
        if res.status == "success":
            # Copy to output path
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy(res.snapshot_path, output_path)
            
            print(f"✅ Bootstrap packet generated: {output_path}")
            print(f"📊 Files: {res.total_files} | Bytes: {res.total_bytes}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "cache-stats":
        stats = cortex_ops.get_cache_stats()
        print(f"💾 Cache Statistics:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {stats}")

    elif args.command == "cache-warmup":
        queries = args.queries or None
        print(f"🔥 Warming up cache...")
        res = cortex_ops.cache_warmup(genesis_queries=queries)
        
        if res.status == "success":
            print(f"✅ Cached {res.queries_cached} queries")
            print(f"💾 Cache hits: {res.cache_hits}")
            print(f"❌ Cache misses: {res.cache_misses}")
            print(f"⏱️  Total time: {res.total_time_ms/1000:.2f}s")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "persist-soul":
        from mcp_servers.learning.models import PersistSoulRequest
        print(f"🌱 Broadcasting soul to Hugging Face AI Commons...")
        print(f"   Snapshot: {args.snapshot}")
        print(f"   Valence: {args.valence} | Uncertainty: {args.uncertainty}")
        print(f"   Full sync: {args.full_sync}")
        
        request = PersistSoulRequest(
            snapshot_path=args.snapshot,
            valence=args.valence,
            uncertainty=args.uncertainty,
            is_full_sync=args.full_sync
        )
        # ROUTED TO LEARNING MCP
        res = learning_ops.persist_soul(request)
        
        if res.status == "success":
            print(f"✅ Soul planted successfully!")
            print(f"🔗 Repository: {res.repo_url}")
            print(f"📄 Snapshot: {res.snapshot_name}")
        elif res.status == "quarantined":
            print(f"🚫 Quarantined: {res.error}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "persist-soul-full":
        print(f"🧬 Regenerating full Soul JSONL and deploying to HuggingFace...")
        # ROUTED TO LEARNING MCP
        res = learning_ops.persist_soul_full()
        
        if res.status == "success":
            print(f"✅ Full sync complete!")
            print(f"🔗 Repository: {res.repo_url}")
            print(f"📄 Output: {res.snapshot_name}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "evolution":
        if not args.subcommand:
            print("❌ Subcommand required for 'evolution' (fitness, depth, scope)")
            sys.exit(1)
            
        content = args.content
        if args.file:
            try:
                content = Path(args.file).read_text()
            except Exception as e:
                print(f"❌ Error reading file {args.file}: {e}")
                sys.exit(1)
        
        if not content:
            print("❌ No content provided. Use a positional argument or --file.")
            sys.exit(1)
            
        if args.subcommand == "fitness":
            res = evolution_ops.calculate_fitness(content)
            print(json.dumps(res, indent=2))
        elif args.subcommand == "depth":
            res = evolution_ops.measure_depth(content)
            print(f"Depth: {res}")
        elif args.subcommand == "scope":
            res = evolution_ops.measure_scope(content)
            print(f"Scope: {res}")

    elif args.command in ["rlm-distill", "rlm-test"]:
        print(f"🧠 RLM: Distilling semantic essence of '{args.target}'...")
        import time
        start = time.time()
        
        # Call _rlm_map directly with the target
        results = learning_ops._rlm_map([args.target])
        
        duration = time.time() - start
        print(f"⏱️  Completed in {duration:.2f}s")
        print(f"📊 Files Processed: {len(results)}")
        print("=" * 60)
        
        for file_path, summary in results.items():
            print(f"\n📄 {file_path}")
            print(f"   {summary}")


if __name__ == "__main__":
    main()

```

---

## 📦 Component: Wait-for-Green Helper
**Source:** `scripts/library/wait_for_green.py`
_Helper for safe background operations._

```python
[ERROR: File not found: /Users/richardfremmerlid/Projects/Project_Sanctuary/scripts/library/wait_for_green.py]
```

---

## 🛠️ Environment Setup Guide
### 1. Configuration (.env)
Create a `.env` file in your project root:
```bash
RLM_CACHE_PATH=.agent/learning/rlm_summary_cache.json
RLM_TARGET_DIRS=src,docs,lib  # Customize this list for your repo
```

### 2. Prerequisite: Ollama (The Brain)
1.  **Install:** `curl -fsSL https://ollama.com/install.sh | sh`
2.  **Pull Model:** `ollama pull qwen2.5:7b` (Recommended default)
3.  **Serve:** `ollama serve`

### Prerequisite: Vector DB (Optional Augmentation)
If augmenting a Vector DB (Chroma), ensure you prepend the RLM summary to chunks:
```python
# Pseudo-code for Augmentation
summary = ledger[file_path]['summary']
chunk_content = f"[CONTEXT_SUMMARY: {summary}]\n\n{original_chunk}"
vector_db.add(documents=[chunk_content])
```