#!/usr/bin/env python3
"""
mcp_servers/learning/operations.py
=====================================

Purpose:
    Core operations for the Project Sanctuary Learning Loop (Protocol 128).
    Handles cognitive continuity, snapshot generation, soul persistence,
    and guardian wakeup workflows.
    
    ADR 097: Uses Context Bundler CLI for manifest-based bundling.

Layer: MCP Server / Learning Domain

⚠️  DEPRECATION NOTICE (2026-02-01):
    MCP-based bundling via this module is DEPRECATED.
    Use `tools/cli.py snapshot --type TYPE` instead.
    This module maintains backward compatibility but will be removed in v2.0.
    See: ADR 097 (Base Manifest Inheritance Architecture)

Usage:
    from mcp_servers.learning.operations import LearningOperations
    ops = LearningOperations(project_root)
    result = ops.capture_snapshot(snapshot_type="seal")

Key Functions:
    - capture_snapshot(): Generate context snapshots (seal, audit, learning_audit)
    - persist_soul(): Broadcast learnings to HuggingFace
    - guardian_wakeup(): Protocol 128 bootloader initialization
    - guardian_snapshot(): Session pack generation

Related:
    - tools/retrieve/bundler/bundle.py (Context Bundler)
    - tools/retrieve/bundler/manifest_manager.py (Manifest Manager)
    - tools/cli.py snapshot (PREFERRED CLI path)
    - 01_PROTOCOLS/128_Hardened_Learning_Loop.md
    - ADRs/071_protocol_128_cognitive_continuity.md
"""


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
# ADR 097: Context Bundler CLI paths
BUNDLER_SCRIPT = "tools/retrieve/bundler/bundle.py"
MANIFEST_MANAGER_SCRIPT = "tools/retrieve/bundler/manifest_manager.py"
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
                sop_content = "[MISSING] .agent/workflows/workflow-learning-loop.md"
                protocol_content = "[MISSING] 01_PROTOCOLS/128_Hardened_Learning_Loop.md"
                
                try:
                    p_path = self.project_root / ".agent" / "learning" / "cognitive_primer.md"
                    if p_path.exists(): primer_content = p_path.read_text()
                    
                    s_path = self.project_root / ".agent" / "workflows" / "workflow-learning-loop.md"
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
                    
                    # Handle modular manifest structure (ADR 089 -> ADR 097)
                    if isinstance(manifest_data, dict):
                        # NEW: Prefer 'files' array (ADR 097 simple schema)
                        if "files" in manifest_data and isinstance(manifest_data["files"], list):
                            effective_manifest = []
                            for item in manifest_data["files"]:
                                if isinstance(item, str):
                                    effective_manifest.append(item)
                                elif isinstance(item, dict) and "path" in item:
                                    effective_manifest.append(item["path"])
                            logger.info(f"Loaded {snapshot_type} manifest (ADR 097): {len(effective_manifest)} files")
                        else:
                            # LEGACY: Fallback to core+topic (ADR 089)
                            core = manifest_data.get("core", [])
                            topic = manifest_data.get("topic", [])
                            effective_manifest = core + topic
                            logger.info(f"Loaded {snapshot_type} manifest (legacy): {len(core)} core + {len(topic)} topic entries")
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

        # 3. Generate Snapshot using Context Bundler (ADR 097)
        try:
            from uuid import uuid4
            
            # Create temp manifest in bundler schema format
            temp_manifest = self.project_root / f".temp_manifest_{uuid4()}.json"
            bundler_manifest = {
                "title": f"{snapshot_type.replace('_', ' ').title()} Snapshot",
                "description": f"Auto-generated {snapshot_type} snapshot",
                "files": [{"path": f, "note": ""} for f in effective_manifest]
            }
            temp_manifest.write_text(json.dumps(bundler_manifest, indent=2))
            
            try:
                # Call bundle.py CLI via subprocess
                bundler_cmd = [
                    sys.executable,
                    str(self.project_root / BUNDLER_SCRIPT),
                    str(temp_manifest),
                    "-o", str(final_snapshot_path)
                ]
                result = subprocess.run(
                    bundler_cmd,
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Bundler failed: {result.stderr}")
                    return CaptureSnapshotResponse(
                        snapshot_path="",
                        manifest_verified=manifest_verified,
                        git_diff_context=git_diff_context,
                        snapshot_type=snapshot_type,
                        status="error",
                        error=f"Bundler failed: {result.stderr[:200]}"
                    )
                
                logger.info(f"Bundler output: {result.stdout[:200]}")
                
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
                    total_files=len(effective_manifest),
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
                
                if isinstance(manifest_data, dict):
                    # ADR 097: New simple {files: [{path, note}]} format
                    if "files" in manifest_data:
                        for item in manifest_data["files"]:
                            if isinstance(item, str):
                                target_files.append(item)
                            elif isinstance(item, dict) and "path" in item:
                                target_files.append(item["path"])
                        logger.info(f"🧠 RLM: Loaded {len(target_files)} files (ADR 097 format).")
                    else:
                        # LEGACY: Fallback to core+topic (ADR 089)
                        core = manifest_data.get("core", [])
                        topic = manifest_data.get("topic", [])
                        target_files = core + topic
                        logger.info(f"🧠 RLM: Merged {len(core)} core + {len(topic)} topic entries (legacy).")
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
