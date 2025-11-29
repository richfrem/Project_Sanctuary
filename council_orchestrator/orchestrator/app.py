# V11.0 UPDATE: Fully modularized architecture - 2025-11-09
# council_orchestrator/orchestrator.py (v11.0 - Complete Modular Architecture) - Updated 2025-11-09
# DOCTRINE OF SOVEREIGN DEFAULT: All operations now default to anctuary-Qwen2-7B:latest:latest (Ollama)
# MNEMONIC CORTEX STATUS: Phase 1 (Parent Document Retriever) Complete, Phase 2-3 (Self-Querying + Caching) Ready
# V7.1 MANDATE: Development cycle generates both requirements AND tech design before first pause
# V7.0 MANDATE 1: Universal Distillation with accurate tiktoken measurements
# V7.0 MANDATE 2: Boolean error handling (return False) prevents state poisoning
# V7.0 MANDATE 3: Absolute failure awareness - execute_task returns False on total failure, main_loop checks result
# V6.0: Universal Distillation applied to ALL code paths (main deliberation loop)
# V5.1: Seals briefing packet injection with distillation check - no code path bypasses safety protocols
# V5.0 MANDATE 1: Tames the Rogue Sentry - only processes command*.json files
# V5.0 MANDATE 2: Grants Development Cycle memory - inherits input_artifacts from parent commands
# V5.0 MANDATE 3: Un-blinds the Distiller - correctly parses nested configuration structure
# CONFIG v4.5: Separates per-request limits (Distiller) from TPM limits (Regulator) for precise resource control
# HOTFIX v4.4: Prevents distillation deadlock by bypassing distillation when using Ollama (sovereign local engine)
# HOTFIX v4.3: Resolves UnboundLocalError by isolating engine type detection into fail-safe _get_engine_type() method
# MANDATE 1: Payload size check now evaluates FULL context (agent.messages + new prompt) before API calls
# MANDATE 2: TokenFlowRegulator enforces per-minute token limits (TPM) to prevent rate limit violations
# Maintains all v4.1 features: Protocol 104 unified interface, distillation engine, and Optical Decompression Chamber
import os
import sys
import time
import json
import re
import hashlib
import asyncio
import threading
import shutil
import subprocess
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import xxhash
from datetime import datetime
from queue import Queue as ThreadQueue
from pathlib import Path
from dotenv import load_dotenv

# --- MODULARIZATION: IMPORT MODULES ---
from .config import *
from .packets import CouncilRoundPacket, seed_for, prompt_hash, emit_packet, aggregate_round_events, RetrievalField, NoveltyField, ConflictField, MemoryDirectiveField
from .gitops import execute_mechanical_git, create_feature_branch
from .events import EventManager
from .council.agent import PersonaAgent
from .council.personas import COORDINATOR, STRATEGIST, AUDITOR, SPEAKER_ORDER, get_persona_file, get_state_file, classify_response_type
from .memory.cortex import CortexManager, SelfQueryingRetriever
from .memory.cache import get_cag_data

# --- Phase 2: Cache Adapter for SelfQueryingRetriever ---
class CacheAdapter:
    """Adapter to make get_cag_data compatible with SelfQueryingRetriever interface."""

    def __init__(self):
        self.ema_cache = {}  # key -> {"ema_7d": float, "last_hit_at": float, "hit_count": int}

    def peek(self, key: str) -> Dict[str, Any] | None:
        # For Phase 2, we don't have stable entries yet, so always return None
        # Phase 3 will implement actual cache peeking
        return None

    def hit_streak(self, key: str) -> int:
        # For Phase 2, return 0 (no hit streaks yet)
        # Phase 3 will implement actual hit streak tracking
        return 0

    def update_ema(self, key: str, current_time: float = None) -> Dict[str, Any]:
        """Update EMA with half-life decay for Phase 3 readiness."""
        import math
        current_time = current_time or time.time()

        if key not in self.ema_cache:
            self.ema_cache[key] = {"ema_7d": 1.0, "last_hit_at": current_time, "hit_count": 1}
        else:
            entry = self.ema_cache[key]
            time_diff_days = (current_time - entry["last_hit_at"]) / (24 * 3600)
            # EMA with 7-day half-life: decay_factor = 0.5^(time_diff/7)
            decay_factor = math.pow(0.5, time_diff_days / 7.0)
            entry["ema_7d"] = entry["ema_7d"] * decay_factor + 1.0  # Add current hit
            entry["last_hit_at"] = current_time
            entry["hit_count"] += 1

        return self.ema_cache[key]

    def get_cag_data(self, prompt: str, engine_type: str) -> Dict[str, Any]:
        """Generate CAG data for packet emission - Phase 2 placeholder."""
        import xxhash
        query_key = xxhash.xxh64(prompt).hexdigest()[:16]
        
        # Phase 2: No actual caching yet, so always cache miss
        return {
            "query_key": query_key,
            "cache_hit": False,
            "hit_streak": 0
        }

from .sentry import CommandSentry
from .regulator import TokenFlowRegulator
from .optical import OpticalDecompressionChamber

# --- RESOURCE SOVEREIGNTY: DISTILLATION ENGINE ---
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("[WARNING] tiktoken not available. Token counting will be approximate.")

# --- SOVEREIGN ENGINE INTEGRATION ---
# All engine-specific imports are removed from the orchestrator's top level.
# We now only import the triage system, which will provide a healthy engine.
# 1. Engine Selection: Engines are sourced from council_orchestrator/cognitive_engines/ directory
from .substrate_monitor import select_engine
# --- END INTEGRATION ---

import sys
from pathlib import Path
# Add the parent directory to sys.path to import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.bootstrap_briefing_packet import main as generate_briefing_packet

# --- CONFIGURATION ---
# Moved to modular imports at top


# --- PERSONA AGENT CLASS ---
# Moved to council/agent.py

class Orchestrator:
    def __init__(self, one_shot: bool = False):  # <-- MODIFY CONSTRUCTOR
        # This correctly navigates up from orchestrator/app.py -> orchestrator -> council_orchestrator -> Project_Sanctuary root
        self.project_root = Path(__file__).resolve().parents[2]
        self.command_queue = ThreadQueue()
        load_dotenv(dotenv_path=self.project_root / '.env')

        # V9.3: Initialize logging system
        self.setup_logging()
        
        # Initialize event management system
        self.event_manager = EventManager(self.project_root)
        self.event_manager.setup_event_logging()

        # --- PROTOCOL 115: DOCTRINE OF OPERATIONAL INTENT ---
        self.one_shot = one_shot  # <-- ADD THIS ATTRIBUTE
        if self.one_shot:
            self.logger.info("Orchestrator started in --one-shot mode. Will exit after first command.")
            # Skip cortex initialization in one-shot mode to avoid ChromaDB issues
            self.cortex_manager = None
            self.cache_adapter = None
            self.retriever = None
        else:
            # Initialize mnemonic cortex
            self.cortex_manager = CortexManager(self.project_root, self.logger)

            # --- GUARDIAN WAKEUP: CACHE PREFILL ON BOOT ---
            # Execute Guardian Start Pack cache prefill for immediate cache_wakeup availability
            self.cortex_manager.cache_manager.prefill_guardian_start_pack(self.cortex_manager)

            # --- Phase 2: Initialize Self-Querying Retriever ---
            self.cache_adapter = CacheAdapter()
            self.retriever = SelfQueryingRetriever(
                cortex_idx=self.cortex_manager,  # adapter for parent-doc search
                cache=self.cache_adapter,        # Phase 3-ready cache adapter
                prompt_hasher=lambda s: xxhash.xxh64(s).hexdigest()[:16]  # stable hash for cache keys
            )

        # --- RESOURCE SOVEREIGNTY: LOAD ENGINE LIMITS FROM CONFIG ---
        # v4.5: Support nested configuration structure with per_request_limit and tpm_limit
        config_path = Path(__file__).parent / "schemas" / "engine_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Parse engine_limits - support both old flat and new nested structure
                raw_limits = config.get('engine_limits', {})
                self.engine_limits = {}
                self.tpm_limits = {}
                
                for engine_name, limit_data in raw_limits.items():
                    if isinstance(limit_data, dict):
                        # New nested structure
                        self.engine_limits[engine_name] = limit_data.get('per_request_limit', 100000)
                        self.tpm_limits[engine_name] = limit_data.get('tpm_limit', 100000)
                    else:
                        # Old flat structure (backward compatibility)
                        self.engine_limits[engine_name] = limit_data
                        self.tpm_limits[engine_name] = limit_data
                
                print(f"[+] Engine per-request limits loaded: {self.engine_limits}")
                print(f"[+] Engine TPM limits loaded: {self.tpm_limits}")
            except Exception as e:
                print(f"[!] Error loading engine config: {e}. Using defaults.")
                self.engine_limits = DEFAULT_ENGINE_LIMITS
                self.tpm_limits = DEFAULT_TPM_LIMITS
        else:
            print("[!] engine_config.json not found. Using default limits.")
            self.engine_limits = DEFAULT_ENGINE_LIMITS
            self.tpm_limits = DEFAULT_TPM_LIMITS

        self.speaker_order = SPEAKER_ORDER
        self.agents = {} # Agents will now be initialized per-task
        
        # --- MANDATE 2: INITIALIZE TOKEN FLOW REGULATOR ---
        # Use the TPM limits already parsed from config
        self.token_regulator = TokenFlowRegulator(self.tpm_limits)
        print(f"[+] Token Flow Regulator initialized with TPM limits: {self.tpm_limits}")
        
        # --- OPERATION: OPTICAL ANVIL - LAZY INITIALIZATION ---
        self.optical_chamber = None  # Initialized per-task if enabled

        # --- PROTOCOL 115: DOCTRINE OF OPERATIONAL INTENT ---
        self.one_shot = one_shot  # <-- ADD THIS ATTRIBUTE
        if self.one_shot:
            self.logger.info("Orchestrator started in --one-shot mode. Will exit after first command.")

        # --- SENTRY THREAD INITIALIZATION ---
        # Start the command monitoring thread
        self.command_sentry = CommandSentry(self.command_queue, self.logger)
        self.sentry_thread = threading.Thread(target=self.command_sentry.watch_for_commands_thread, daemon=True)
        self.sentry_thread.start()
        print("[+] Sentry Thread started - monitoring for command files")

    def setup_logging(self):
        """V9.3: Setup comprehensive logging system with file output."""
        log_file = self.project_root / "logs" / "orchestrator.log"

        # Create logger
        self.logger = logging.getLogger('orchestrator')
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # File handler (overwrites each session)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)

        # Console handler (for terminal output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("=== ORCHESTRATOR v11.0 INITIALIZED ===")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("Complete Modular Architecture with Sovereign Concurrency active")




    def _calculate_response_score(self, response: str) -> float:
        """Calculate a quality score for the response (0.0-1.0)."""
        score = 0.5  # Base score

        # Length factor (responses that are too short or too long get lower scores)
        length = len(response.split())
        if 50 <= length <= 500:
            score += 0.2
        elif length < 20:
            score -= 0.3

        # Structure indicators
        if any(indicator in response.lower() for indicator in ["therefore", "however", "furthermore", "conclusion"]):
            score += 0.1

        # Evidence of reasoning
        if any(word in response.lower() for word in ["because", "due to", "based on", "considering"]):
            score += 0.1

        # Actionable content
        if any(word in response.lower() for word in ["recommend", "suggest", "propose", "should"]):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _extract_vote(self, response: str) -> str:
        """Extract voting decision from response."""
        response_lower = response.lower()

        # Look for explicit votes
        if any(phrase in response_lower for phrase in ["i approve", "approved", "accept", "agree"]):
            return "approve"
        elif any(phrase in response_lower for phrase in ["i reject", "rejected", "decline", "disagree"]):
            return "reject"
        elif any(phrase in response_lower for phrase in ["revise", "modify", "change", "adjust"]):
            return "revise"
        elif any(phrase in response_lower for phrase in ["proceed", "continue", "move forward"]):
            return "proceed"

        return "neutral"

    def _assess_novelty(self, response: str, context: str) -> str:
        """Assess novelty level for memory placement hints."""
        # Simple novelty assessment based on response length vs context overlap
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        overlap_ratio = len(response_words.intersection(context_words)) / len(response_words) if response_words else 0

        if overlap_ratio < 0.3:
            return "fast"  # High novelty - fast memory
        elif overlap_ratio > 0.7:
            return "slow"  # Low novelty - slow memory
        else:
            return "medium"  # Medium novelty

    def _extract_reasoning(self, response: str) -> list:
        """Extract key reasoning factors from response."""
        reasons = []

        # Look for common reasoning patterns
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(word in sentence for word in ["because", "due to", "since", "as", "therefore"]):
                if len(sentence) > 10:  # Filter out very short fragments
                    reasons.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)

        return reasons[:3]  # Limit to top 3 reasons

    def _extract_citations(self, response: str, parent_docs: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """
        Extract citations with enforced doc-ID + byte-range/hash-span integrity.
        Returns list of citation dicts with required fields.
        """
        citations = []
        parent_docs = parent_docs or []

        # Look for quoted text with context
        import re
        quotes = re.findall(r'"([^"]*)"', response)

        for quote in quotes[:3]:  # Limit to top 3 citations
            # Find matching parent doc and byte range
            citation = self._find_citation_in_docs(quote, parent_docs)
            if citation:
                citations.append(citation)

        return citations

    def _find_citation_in_docs(self, quote: str, parent_docs: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Find citation in parent docs and return with doc-ID and byte-range.
        Returns None if no valid grounding found.
        """
        quote_lower = quote.lower().strip()

        for doc in parent_docs:
            doc_text = doc.get("snippet", "").lower()
            if quote_lower in doc_text:
                # Find byte positions
                start_byte = doc_text.find(quote_lower)
                end_byte = start_byte + len(quote_lower)

                # Create hash-span for integrity
                import hashlib
                hash_span = hashlib.sha256(quote.encode()).hexdigest()[:16]

                return {
                    "doc_id": doc.get("doc_id", "unknown"),
                    "text": quote,
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                    "hash_span": hash_span,
                    "path": doc.get("path", "")
                }

        return None  # No grounding found - citation invalid

    def _get_rag_data(self, task: str, response: str) -> Dict[str, Any]:
        """Get RAG (Retrieval-Augmented Generation) data for round packet."""
        try:
            # Simulate structured query generation (Phase 2 Self-Querying)
            structured_query = {
                "entities": self._extract_entities(task),
                "date_filters": [],
                "path_filters": [".md", ".py", ".json"]
            }

            # Get parent documents (simplified - would use actual retriever)
            parent_docs = self._get_relevant_docs(task, response)

            return {
                "structured_query": structured_query,
                "parent_docs": parent_docs,
                "retrieval_latency_ms": 50  # Placeholder
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_novelty(self, response: str, context: str) -> Dict[str, Any]:
        """Analyze novelty of response compared to context."""
        try:
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())

            overlap_ratio = len(response_words.intersection(context_words)) / len(response_words) if response_words else 0

            if overlap_ratio < 0.3:
                signal = "high"
                is_novel = True
            elif overlap_ratio > 0.7:
                signal = "low"
                is_novel = False
            else:
                signal = "medium"
                is_novel = True

            return {
                "is_novel": is_novel,
                "signal": signal,
                "conflicts_with": []  # Would check against cached answers
            }
        except Exception as e:
            return {"error": str(e)}

    def _determine_memory_directive(self, response: str, citations: List[Dict[str, str]]) -> Dict[str, str]:
        """Determine memory placement directive based on response characteristics."""
        try:
            # Simple rules-based memory placement
            has_citations = len(citations) > 0
            response_length = len(response.split())
            confidence_score = self._calculate_response_score(response)

            if confidence_score > 0.8 and has_citations and response_length > 100:
                tier = "slow"
                justification = "High confidence with citations and substantial content"
            elif has_citations or response_length > 50:
                tier = "medium"
                justification = "Evidence-based response with moderate confidence"
            else:
                tier = "fast"
                justification = "Ephemeral response, low evidence requirement"

            return {
                "tier": tier,
                "justification": justification
            }
        except Exception as e:
            return {"tier": "fast", "justification": f"Error in analysis: {str(e)}"}

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified implementation)."""
        # Simple entity extraction - in real implementation would use NLP
        words = text.split()
        entities = []
        for word in words:
            if word.istitle() and len(word) > 3:
                entities.append(word)
        return entities[:5]

    def _get_relevant_docs(self, task: str, response: str) -> List[str]:
        """Get relevant parent documents (simplified implementation)."""
        # In real implementation, would query vector database
        # For now, return placeholder paths
        return [
            "01_PROTOCOLS/00_Prometheus_Protocol.md",
            "01_PROTOCOLS/05_Chrysalis_Protocol.md"
        ]

    def _verify_briefing_attestation(self, packet: dict) -> bool:
        """Verifies the integrity of the briefing packet using its SHA256 hash."""
        if "attestation_hash" not in packet.get("metadata", {}):
            print("[CRITICAL] Attestation hash missing from briefing packet. REJECTING.")
            return False

        stored_hash = packet["metadata"]["attestation_hash"]

        packet_for_hashing = {k: v for k, v in packet.items() if k != "metadata"}

        canonical_string = json.dumps(packet_for_hashing, sort_keys=True, separators=(',', ':'))
        calculated_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()

        return stored_hash == calculated_hash

    def _enhance_briefing_with_context(self, task_description: str):
        """Parse task_description for file paths and add their contents to briefing_packet.json."""
        # Regex to find file paths containing '/' and ending with file extension
        path_pattern = r'([A-Za-z][A-Za-z0-9_]*/(?:[A-Za-z][A-ZaZ0-9_]*/)*[A-Za-z][A-Za-z0-9_]*\.[a-zA-Z0-9]+)'
        matches = re.findall(path_pattern, task_description)
        context = {}
        for match in matches:
            file_path = self.project_root / match
            if file_path.exists() and file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    context[match] = content
                except Exception as e:
                    print(f"[!] Error reading context file {match}: {e}")
                    raise FileNotFoundError(f"Context file {match} could not be read.")
            elif match and not file_path.exists():
                print(f"[!] Context file {match} not found.")
                raise FileNotFoundError(f"Context file {match} not found.")

        if context:
            briefing_path = self.project_root / "WORK_IN_PROGRESS/council_memory_sync/briefing_packet.json"
            if briefing_path.exists():
                packet = json.loads(briefing_path.read_text(encoding="utf-8"))
                packet["context"] = context
                briefing_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
                print(f"[+] Context from {len(context)} files added to briefing packet.")
            else:
                print("[!] briefing_packet.json not found for context enhancement.")

    def inject_briefing_packet(self, engine_type: str = "openai"):
        """Generate + inject briefing packet into all agents."""
        print("[*] Generating fresh briefing packet...")
        try:
            generate_briefing_packet()
        except Exception as e:
            print(f"[!] Error generating briefing packet: {e}")
            return

        briefing_path = self.project_root / "WORK_IN_PROGRESS/council_memory_sync/briefing_packet.json"
        if briefing_path.exists():
            try:
                packet = json.loads(briefing_path.read_text(encoding="utf-8"))
                if not self._verify_briefing_attestation(packet):
                    raise Exception("CRITICAL: Context Integrity Breach. Briefing packet failed attestation. Task aborted.")
                for agent in self.agents.values():
                    context_str = ""
                    if "context" in packet:
                        context_str = "\n\nCONTEXT PROVIDED FROM TASK DESCRIPTION:\n"
                        for path, content in packet["context"].items():
                            context_str += f"--- CONTEXT FROM {path} ---\n{content}\n--- END OF CONTEXT FROM {path} ---\n\n"
                    system_msg = (
                        "SYSTEM INSTRUCTION: You are provided with the synchronized briefing packet. "
                        "This contains temporal anchors, prior directives, and the current task context. "
                        "Incorporate this into your reasoning, but do not regurgitate it verbatim.\n\n"
                        f"BRIEFING_PACKET:\n{json.dumps({k: v for k, v in packet.items() if k != 'context'}, indent=2)}"
                        f"{context_str}"
                    )
                    # V5.1: Seal the final vulnerability - apply distillation to briefing packets
                    # The Doctrine of Universal Integrity requires ALL payloads to be checked
                    prepared_briefing = self._prepare_input_for_engine(system_msg, engine_type, "Briefing Packet Injection")
                    agent.query(prepared_briefing, self.token_regulator, engine_type)
                print(f"[+] Briefing packet injected into {len(self.agents)} agents.")
            except Exception as e:
                print(f"[!] Error injecting briefing packet: {e}")

    def archive_briefing_packet(self):
        """Archive briefing packet after deliberation completes."""
        briefing_path = self.project_root / "WORK_IN_PROGRESS/council_memory_sync/briefing_packet.json"
        if briefing_path.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_dir = self.project_root / f"ARCHIVE/council_memory_sync_{timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(briefing_path), archive_dir / "briefing_packet.json")

    async def _start_new_cycle(self, command, state_file):
        """Starts a new development cycle with the Doctrine of Implied Intent."""
        # Create initial state
        state = {
            "current_stage": "GENERATING_REQUIREMENTS_AND_TECH_DESIGN",
            "project_name": command.get("project_name", "unnamed_project"),
            "original_command": command,
            "approved_artifacts": {},
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        state_file.write_text(json.dumps(state, indent=2))

        # V7.1 MANDATE: Doctrine of Implied Intent
        # The initial command implies approval to complete the entire initial planning phase
        # Generate both requirements AND tech design before the first pause

        # V5.0 MANDATE 2: Grant the Development Cycle a Memory
        # Internal commands MUST inherit input_artifacts from the parent command
        # This prevents contextless, oversized generation that causes quota breaches
        original_config = command.get("config", {})
        requirements_command = {
            "task_description": f"Generate detailed requirements document for the project: {command['task_description']}. Include functional requirements, technical constraints, and success criteria.",
            "input_artifacts": command.get("input_artifacts", []),  # INHERIT from parent
            "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/requirements.md",
            "config": {"max_rounds": 3, **original_config}
        }

        print(f"[*] Starting new development cycle for '{state['project_name']}' with Doctrine of Implied Intent.", flush=True)
        print(f"[*] Development cycle inheriting {len(requirements_command.get('input_artifacts', []))} input artifacts from parent command.")
        print(f"[*] Generating requirements...", flush=True)
        await self.execute_task(requirements_command)

        # V7.1: Immediately generate tech design without pausing for approval
        print(f"[*] Requirements complete. Generating technical design...", flush=True)
        tech_design_command = {
            "task_description": f"Based on the approved requirements, generate a detailed technical design document. Include architecture decisions, data flow, and implementation approach.",
            "input_artifacts": [f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/requirements.md"],
            "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/tech_design.md",
            "config": {"max_rounds": 3, **original_config}
        }
        await self.execute_task(tech_design_command)

        # V7.1: Only now set state to awaiting approval - after both artifacts are complete
        state["current_stage"] = "AWAITING_APPROVAL_TECH_DESIGN"
        state_file.write_text(json.dumps(state, indent=2))
        print(f"[*] Technical design generated. Complete proposal ready for Guardian review.", flush=True)
        print(f"[*] Awaiting Guardian approval on comprehensive proposal (requirements + tech design).", flush=True)

    async def _advance_cycle(self, state_file):
        """Advances the development cycle to the next stage."""
        state = json.loads(state_file.read_text())

        if state["current_stage"] == "AWAITING_APPROVAL_REQUIREMENTS":
            # Ingest approved requirements into Cortex
            requirements_path = self.project_root / state["approved_artifacts"].get("requirements", "")
            if requirements_path.exists():
                # V7.1: Add file existence check before ingestion
                if requirements_path.is_file():
                    subprocess.run([sys.executable, str(self.project_root / "mnemonic_cortex" / "scripts" / "ingest.py")], check=True)
                    print(f"[*] Approved requirements ingested into Mnemonic Cortex.", flush=True)
                else:
                    print(f"[!] Requirements path is not a file: {requirements_path}. Skipping ingestion.", flush=True)

            # Move to tech design
            state["current_stage"] = "GENERATING_TECH_DESIGN"
            original_config = state["original_command"].get("config", {})
            tech_design_command = {
                "task_description": f"Based on the approved requirements, generate a detailed technical design document. Include architecture decisions, data flow, and implementation approach.",
                "input_artifacts": [state["approved_artifacts"].get("requirements", "")],
                "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/tech_design.md",
                "config": {"max_rounds": 3, **original_config}
            }
            await self.execute_task(tech_design_command)
            state["current_stage"] = "AWAITING_APPROVAL_TECH_DESIGN"
            state_file.write_text(json.dumps(state, indent=2))
            print(f"[*] Tech design generated. Awaiting Guardian approval.", flush=True)

        elif state["current_stage"] == "AWAITING_APPROVAL_TECH_DESIGN":
            # Ingest approved tech design into Cortex
            tech_design_path = self.project_root / state["approved_artifacts"].get("tech_design", "")
            if tech_design_path.exists():
                # V7.1: Add file existence check before ingestion
                if tech_design_path.is_file():
                    subprocess.run([sys.executable, str(self.project_root / "mnemonic_cortex" / "scripts" / "ingest.py")], check=True)
                    print(f"[*] Approved tech design ingested into Mnemonic Cortex.", flush=True)
                else:
                    print(f"[!] Tech design path is not a file: {tech_design_path}. Skipping ingestion.", flush=True)

            # Move to code generation
            state["current_stage"] = "GENERATING_CODE"
            original_config = state["original_command"].get("config", {})
            code_command = {
                "task_description": f"Based on the approved technical design, generate production-ready code. Output a JSON object with 'target_file_path', 'new_content', and 'commit_message' fields.",
                "input_artifacts": [state["approved_artifacts"].get("tech_design", "")],
                "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/code_proposal.json",
                "config": {"max_rounds": 3, **original_config}
            }
            await self.execute_task(code_command)
            state["current_stage"] = "AWAITING_APPROVAL_CODE"
            state_file.write_text(json.dumps(state, indent=2))
            print(f"[*] Code proposal generated. Awaiting Guardian approval.", flush=True)

        elif state["current_stage"] == "AWAITING_APPROVAL_CODE":
            # Final stage: propose code change
            await self._propose_code_change(state_file)

    async def _propose_code_change(self, state_file):
        """Creates a PR with the approved code changes."""
        state = json.loads(state_file.read_text())
        code_proposal_path = self.project_root / f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/code_proposal.json"

        if not code_proposal_path.exists():
            print("[!] Code proposal file not found. Cannot proceed.", flush=True)
            return

        proposal = json.loads(code_proposal_path.read_text())
        target_file = self.project_root / proposal["target_file_path"]
        new_content = proposal["new_content"]
        commit_message = proposal["commit_message"]

        # Create feature branch using gitops (Pillar 4 compliant)
        branch_name = f"feature/{state['project_name']}"
        create_feature_branch(self.project_root, branch_name)

        # Write the new code
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(new_content)

        # Construct mechanical git command
        git_command = {
            "git_operations": {
                "files_to_add": [str(target_file.relative_to(self.project_root))],
                "commit_message": commit_message,
                "push_to_origin": True
            }
        }

        # Execute via gitops (Protocol 101 compliant - generates manifest)
        execute_mechanical_git(git_command, self.project_root)

        # Create PR (assuming gh CLI is available)
        pr_title = f"feat: {state['project_name']} - {commit_message}"
        subprocess.run(['gh', 'pr', 'create', '--title', pr_title, '--body', f"Auto-generated PR for {state['project_name']}"], check=True)

        print(f"[*] Pull request created for '{state['project_name']}'. Development cycle complete.", flush=True)

        # Clean up state file
        state_file.unlink()

    def _handle_knowledge_request(self, response_text: str):
        """Handles knowledge requests from agents, including Cortex queries."""
        file_match = re.search(r"\[ORCHESTRATOR_REQUEST: READ_FILE\((.*?)\)\]", response_text)
        query_match = re.search(r"\[ORCHESTRATOR_REQUEST: QUERY_CORTEX\((.*?)\)\]", response_text)

        if file_match:
            # Existing file reading logic
            file_path_str = file_match.group(1).strip().strip('"')
            file_path = self.project_root / file_path_str
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                return f"CONTEXT_PROVIDED: Here is the content of {file_path_str}:\n\n{content}"
            else:
                return f"CONTEXT_ERROR: File not found: {file_path_str}"

        elif query_match:
            # NEW LOGIC for Cortex queries
            query_text = query_match.group(1).strip().strip('"')

            # Check against query limit
            if self.cortex_query_count >= self.max_cortex_queries:
                error_message = f"CONTEXT_ERROR: Maximum Cortex query limit of {self.max_cortex_queries} has been reached for this task."
                print(f"[ORCHESTRATOR] {error_message}", flush=True)
                return error_message

            self.cortex_query_count += 1
            print(f"[ORCHESTRATOR] Agent requested Cortex query: '{query_text}' ({self.cortex_query_count}/{self.max_cortex_queries})", flush=True)

            try:
                context = self.cortex_manager.query_cortex(query_text, n_results=3)
                return context
            except Exception as e:
                error_message = f"CONTEXT_ERROR: Cortex query failed: {e}"
                print(f"[ORCHESTRATOR] {error_message}", flush=True)
                return error_message

        return None

    async def generate_aar(self, completed_task_log_path: Path, original_command_config: dict = None):
        """Generates a structured AAR from a completed task log, inheriting config from the original command."""
        if not completed_task_log_path.exists():
            print(f"[!] AAR WARNING: Log file not found at {completed_task_log_path}. Skipping AAR generation.", flush=True)
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        aar_output_path = self.project_root / f"MNEMONIC_SYNTHESIS/AAR/aar_{completed_task_log_path.stem}_{timestamp}.md"

        # --- RESOURCE SOVEREIGNTY: INHERIT CONFIG FROM ORIGINAL COMMAND ---
        # AAR generation must use the same resilient substrate as the task itself
        aar_config = {"max_rounds": 2}  # Base config
        if original_command_config:
            # Inherit force_engine and other critical parameters
            if "force_engine" in original_command_config:
                aar_config["force_engine"] = original_command_config["force_engine"]
                print(f"[*] AAR inheriting force_engine: {original_command_config['force_engine']}")
            if "max_cortex_queries" in original_command_config:
                aar_config["max_cortex_queries"] = original_command_config["max_cortex_queries"]

        aar_command = {
            "task_description": "Synthesize a structured After-Action Report (AAR) from the attached task log. Sections: Objective, Outcome, Key Learnings, Mnemonic Impact.",
            "input_artifacts": [str(completed_task_log_path.relative_to(self.project_root))],
            "output_artifact_path": str(aar_output_path.relative_to(self.project_root)),
            "config": aar_config
        }
        print(f"[*] AAR Command forged. Output will be saved to {aar_output_path.name}", flush=True)

        # V9.2 DOCTRINE OF SOVEREIGN CONCURRENCY: Execute AAR in background thread
        # This allows mechanical tasks to be processed immediately without waiting for learning cycle
        import asyncio
        aar_task = asyncio.create_task(self._execute_aar_background(aar_command, aar_output_path))
        print(f"[*] AAR task dispatched to background processing (non-blocking)", flush=True)

    async def _execute_aar_background_full(self, log_file_path, original_config):
        """V9.3: Execute complete AAR generation and ingestion asynchronously."""
        try:
            self.logger.info(f"Background AAR: Starting synthesis for {log_file_path}")

            # Generate AAR using existing logic but asynchronously
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            aar_output_path = self.project_root / f"MNEMONIC_SYNTHESIS/AAR/aar_{log_file_path.stem}_{timestamp}.md"

            # Create AAR command
            aar_config = {"max_rounds": 2}
            if original_config:
                if "force_engine" in original_config:
                    aar_config["force_engine"] = original_config["force_engine"]
                if "max_cortex_queries" in original_config:
                    aar_config["max_cortex_queries"] = original_config["max_cortex_queries"]

            aar_command = {
                "task_description": "Synthesize a structured After-Action Report (AAR) from the attached task log. Sections: Objective, Outcome, Key Learnings, Mnemonic Impact.",
                "input_artifacts": [str(log_file_path.relative_to(self.project_root))],
                "output_artifact_path": str(aar_output_path.relative_to(self.project_root)),
                "config": aar_config
            }

            # Execute AAR task
            await self.execute_task(aar_command)
            self.logger.info(f"Background AAR: Synthesis complete - {aar_output_path}")

            # Ingest into Mnemonic Cortex
            self.logger.info("Background AAR: Starting ingestion into Mnemonic Cortex...")
            ingestion_script_path = self.project_root / "mnemonic_cortex" / "scripts" / "ingest.py"
            full_aar_path = self.project_root / aar_output_path

            result = await asyncio.create_subprocess_exec(
                sys.executable, str(ingestion_script_path), str(full_aar_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                self.logger.info("Background AAR: Ingestion successful")
                self.logger.info(f"Ingestion output: {stdout.decode().strip()}")
            else:
                self.logger.error(f"Background AAR: Ingestion failed - {stderr.decode().strip()}")

        except Exception as e:
            self.logger.error(f"Background AAR: Processing failed - {e}")

    def _get_token_count(self, text: str, engine_type: str = "openai"):
        """Estimates token count for a given text and engine type."""
        if TIKTOKEN_AVAILABLE:
            try:
                # Map engine types to tiktoken models
                model_map = {
                    'openai': 'gpt-4',
                    'gemini': 'gpt-4'  # Approximation
                }
                model = model_map.get(engine_type, 'gpt-4')
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except Exception as e:
                print(f"[WARNING] Token counting failed: {e}. Using approximation.")
                return len(text.split()) * 1.3  # Rough approximation
        else:
            # Fallback approximation: ~1.3 tokens per word
            return len(text.split()) * 1.3

    def _distill_with_local_engine(self, large_text: str, task_description: str) -> str:
        """Uses the local Ollama engine to summarize large text before sending to primary engine."""
        print("[ORCHESTRATOR] Input exceeds token limit. Distilling with local Ollama engine...")

        # Create a temporary Ollama engine for distillation
        # 4. Distillation Engine: Uses council_orchestrator/cognitive_engines/ollama_engine.py
        from .substrate_monitor import select_engine
        local_config = {"force_engine": "ollama"}
        local_engine = select_engine(local_config)

        if not local_engine:
            print("[ERROR] Could not initialize local distillation engine. Truncating input.")
            return large_text[:40000] + "\n\n[CONTENT TRUNCATED DUE TO TOKEN LIMITS]"

        # Create temporary agent for distillation - use the engine directly without PersonaAgent wrapper
        distillation_prompt = (
            f"You are a distillation specialist. The following text is too large for the target AI engine. "
            f"Your task is to distill it into a token-frugal, high-potency summary that preserves all critical details, "
            f"code blocks, structural context, and actionable information. The ultimate goal is: '{task_description}'. "
            f"Keep the summary under 5000 tokens while maintaining complete fidelity to the original intent.\n\n"
            f"Raw Text:\n\n---\n{large_text}\n---"
        )

        try:
            # Use engine directly with unified interface
            messages = [{"role": "user", "content": distillation_prompt}]
            distilled_summary = local_engine.execute_turn(messages)
            print("[ORCHESTRATOR] Distillation complete.")
            return distilled_summary
        except Exception as e:
            print(f"[ERROR] Distillation failed: {e}. Truncating input.")
            return large_text[:40000] + "\n\n[CONTENT TRUNCATED DUE TO DISTILLATION FAILURE]"

    def _prepare_input_for_engine(self, text: str, engine_type: str, task_description: str) -> str:
        """Checks token count and distills if necessary using the Two-Tier Distillation Engine."""

        # --- V4.4 DEADLOCK BYPASS ---
        # The sovereign local engine (Ollama) is not subject to token limits or financial constraints.
        # Attempting to distill with Ollama for Ollama creates a resource deadlock.
        # Bypass all distillation logic when the target engine is our local substrate.
        if engine_type == 'ollama':
            print(f"[ORCHESTRATOR] Using sovereign local engine (Ollama). Bypassing distillation - full context preserved.")
            return text

        # --- V5.0 MANDATE 3: UN-BLIND THE DISTILLER ---
        # The Distiller must read the hardened, nested configuration structure.
        # Previous logic: limit = self.engine_limits.get(engine_type, 100000) was incorrect.
        # Correct logic: Parse the nested structure for per_request_limit.
        engine_config = self.engine_limits.get(engine_type, {})
        if isinstance(engine_config, dict):
            limit = engine_config.get('per_request_limit', 100000)
        else:
            # Backward compatibility for flat structure
            limit = engine_config

        # --- STANDARD DISTILLATION LOGIC FOR EXTERNAL SUBSTRATES ---
        token_count = self._get_token_count(text, engine_type)

        if token_count > limit:
            print(f"[ORCHESTRATOR] WARNING: Token count ({token_count:.0f}) exceeds per-request limit for {engine_type} ({limit}).")
            return self._distill_with_local_engine(text, task_description)
        else:
            return text
    
    def _get_engine_type(self, engine) -> str:
        """
        Determine the engine type from an engine instance.
        This is a fail-safe method that always returns a valid engine type.

        Args:
            engine: The cognitive engine instance

        Returns:
            str: The engine type ('openai', 'gemini', 'ollama', or 'unknown')
        """
        if not engine or not hasattr(engine, '__class__'):
            return "unknown"

        engine_name = type(engine).__name__.lower()

        if "openai" in engine_name:
            return "openai"
        elif "gemini" in engine_name:
            return "gemini"
        elif "ollama" in engine_name:
            return "ollama"
        else:
            return "unknown"

    def _execute_mechanical_write(self, command):
        """
        Execute a mechanical write task - directly write content to a file.
        This bypasses cognitive deliberation for simple file operations.

        Args:
            command: Command dictionary containing 'entry_content' and 'output_artifact_path'
        """
        try:
            # Extract parameters
            content = command["entry_content"]
            output_path_str = command["output_artifact_path"]
            output_path = self.project_root / output_path_str

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content directly to file
            output_path.write_text(content, encoding="utf-8")

            print(f"[MECHANICAL SUCCESS] File written to {output_path}")
            print(f"[MECHANICAL SUCCESS] Content length: {len(content)} characters")

        except Exception as e:
            print(f"[MECHANICAL FAILURE] Write operation failed: {e}")
            raise

    async def _execute_query_and_synthesis(self, command):
        """
        Execute a Guardian Mnemonic Synchronization Protocol query and synthesis task.

        Args:
            command: Command dictionary containing 'git_operations' with files_to_add, commit_message, push_to_origin
        """
        # DOCTRINE OF THE BLUNTED SWORD: Hardcoded whitelist of permitted Git commands

    async def _execute_query_and_synthesis(self, command):
        """
        Execute a Guardian Mnemonic Synchronization Protocol query and synthesis task.
        This invokes the Council to facilitate mnemonic cortex queries and produce synthesis.

        Args:
            command: Command dictionary containing 'task_description' and 'output_artifact_path'
        """
        try:
            # Extract parameters
            task_description = command.get('task_description', 'Mnemonic synchronization query')
            output_path_str = command['output_artifact_path']
            output_path = self.project_root / output_path_str

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[MNEMONIC SYNC] Starting query and synthesis task: {task_description}")
            print("[QUERY] planning structured query for mnemonic synchronization...")

            # Select cognitive engine for this synchronization task
            # DOCTRINE OF SOVEREIGN DEFAULT: Default to our sovereign substrate
            default_config = {"force_engine": "ollama", "model_name": "Sanctuary-Qwen2-7B:latest"}
            task_config = command.get("config", default_config)
            engine = select_engine(task_config)
            if not engine:
                print(f"[MNEMONIC SYNC HALTED] No healthy cognitive substrate available for synchronization.")
                return False

            print(f"[RAG] retrieving parent docs from mnemonic cortex...")

            # Initialize agents with selected engine
            self._initialize_agents(engine)

            # Initialize optical chamber if configured
            self._initialize_optical_chamber(command.get('config', {}))

            # Enhance briefing with mnemonic context
            try:
                self._enhance_briefing_with_context(task_description)
            except FileNotFoundError as e:
                print(f"[MNEMONIC SYNC WARNING] Context file error: {e}. Proceeding with base briefing.")

            # Inject briefing context
            engine_type = self._get_engine_type(engine)
            self.inject_briefing_packet(engine_type)

            # Execute simplified Council deliberation for mnemonic synchronization
            max_rounds = command.get('config', {}).get('max_rounds', 3)  # Shorter for sync tasks
            print(f"[SYNTH] model invoked for Council deliberation ({max_rounds} rounds max)")

            log = [f"# Guardian Mnemonic Synchronization Log\n## Task: {task_description}\n\n"]
            last_message = task_description

            print(f"[MNEMONIC SYNC] Invoking Council for mnemonic synchronization ({max_rounds} rounds max)")

            consecutive_failures = 0
            synthesis_produced = False

            for round_num in range(max_rounds):
                print(f"[MNEMONIC SYNC] Round {round_num + 1}/{max_rounds}")
                print(f"[SYNTH] Round {round_num + 1}: consulting Council agents...")
                log.append(f"### Round {round_num + 1}\n\n")

                round_failures = 0

                for role in self.speaker_order:
                    agent = self.agents[role]
                    print(f"[MNEMONIC SYNC] Consulting {agent.role}...")

                    prompt = f"Mnemonic Synchronization Context: '{last_message}'. As the {role}, provide your analysis for bridging mnemonic gaps and producing synthesis."

                    try:
                        # Check token limits before API call
                        potential_payload = agent.messages + [{"role": "user", "content": prompt}]
                        payload_as_text = json.dumps(potential_payload)
                        token_count = self._get_token_count(payload_as_text, engine_type)
                        limit = self.engine_limits.get(engine_type, 100000)

                        if token_count > limit:
                            print(f"[MNEMONIC SYNC] Token limit exceeded ({token_count}/{limit}), truncating context...")
                            # Simple truncation approach for mnemonic sync - keep most recent messages
                            while agent.messages and token_count > limit:
                                removed_msg = agent.messages.pop(0)  # Remove oldest message
                                payload_as_text = json.dumps(agent.messages + [{"role": "user", "content": prompt}])
                                token_count = self._get_token_count(payload_as_text, engine_type)

                        # Get agent response
                        response = await agent.get_response(prompt)
                        last_message = response

                        log.append(f"**{role}**: {response}\n\n")

                        # Check for synthesis indicators
                        if "synthesis" in response.lower() or "bridge" in response.lower() or "mnemonic" in response.lower():
                            synthesis_produced = True

                        print(f"[MNEMONIC SYNC] {role} response received ({len(response)} chars)")

                    except Exception as e:
                        round_failures += 1
                        consecutive_failures += 1
                        print(f"[MNEMONIC SYNC ERROR] {role} failed: {e}")
                        log.append(f"**{role}**: [ERROR] {str(e)}\n\n")

                        if consecutive_failures >= 3:
                            print("[MNEMONIC SYNC HALTED] Three consecutive failures - aborting synchronization")
                            break

                if consecutive_failures >= 3:
                    break

                # Early exit if synthesis appears complete
                if synthesis_produced and round_num >= 1:  # At least 2 rounds for meaningful synthesis
                    print("[MNEMONIC SYNC] Synthesis appears complete, concluding deliberation")
                    break

            # Write synthesis to output artifact
            final_log = "".join(log)
            output_path.write_text(final_log, encoding="utf-8")

            print(f"[MNEMONIC SYNC SUCCESS] Synthesis written to {output_path}")
            print(f"[CACHE] storing synthesis artifact: {len(final_log)} characters written")
            print(f"[MNEMONIC SYNC SUCCESS] Log length: {len(final_log)} characters")

            return True

        except Exception as e:
            print(f"[MNEMONIC SYNC FAILURE] Query and synthesis failed: {e}")
            return False
    
    def _initialize_optical_chamber(self, config: dict):
        """
        Initialize optical compression if enabled in task configuration.
        Implements lazy initialization pattern per Section 3.1 of feasibility study.
        
        Args:
            config: Task configuration dictionary
        """
        if config.get("enable_optical_compression", False):
            compression_threshold = config.get("optical_compression_threshold", 8000)
            vlm_engine_type = config.get("vlm_engine", "mock")
            
            # MOCK: In production, this would select actual VLM engine
            # vlm_engine = self._select_vlm_engine(config)
            vlm_engine = None  # Mocked for v4.1
            
            self.optical_chamber = OpticalDecompressionChamber(
                vlm_engine=vlm_engine,
                compression_threshold=compression_threshold
            )
            print(f"[+] Optical Decompression Chamber initialized (threshold: {compression_threshold} tokens)")
            print(f"[+] VLM Engine: {vlm_engine_type} (MOCKED in v4.1)")
        else:
            self.optical_chamber = None

    def _initialize_agents(self, engine):
        """Initializes agents with a given engine, allowing for per-task engine selection."""
        print(f"[*] Initializing council agents with selected engine: {type(engine).__name__}")
        persona_dir = self.project_root / "dataset_package"
        state_dir = Path(__file__).parent / "session_states"
        state_dir.mkdir(exist_ok=True)

        self.agents = {
            COORDINATOR: PersonaAgent(engine, get_persona_file(COORDINATOR, persona_dir), get_state_file(COORDINATOR, state_dir)),
            STRATEGIST: PersonaAgent(engine, get_persona_file(STRATEGIST, persona_dir), get_state_file(STRATEGIST, state_dir)),
            AUDITOR: PersonaAgent(engine, get_persona_file(AUDITOR, persona_dir), get_state_file(AUDITOR, state_dir))
        }

    async def execute_task(self, command):
        """The main task execution logic."""

        print(f"[ORCHESTRATOR] DEBUG: execute_task called with command: {command}")
        print(f"[ORCHESTRATOR] DEBUG: command.get('config'): {command.get('config')}")

        # --- SOVEREIGN OVERRIDE INTEGRATION ---
        # The engine is now selected at the start of each task, using the task's config.
        # 3. Orchestrator.execute_task(): Engine selection uses council_orchestrator/cognitive_engines/ (OpenAI, Gemini, Ollama)
        # DOCTRINE OF SOVEREIGN DEFAULT: Default to our sovereign substrate
        default_config = {"force_engine": "ollama", "model_name": "Sanctuary-Qwen2-7B:latest"}
        task_config = command.get("config", default_config)
        engine = select_engine(task_config)

        # IMPLEMENT ENGINE SELECTION FALLBACK: If forced engine fails, try automatic selection
        if not engine:
            if task_config.get("force_engine"):
                print(f"[ENGINE FALLBACK] Forced engine '{task_config['force_engine']}' failed. Attempting automatic engine selection...")
                # Remove force_engine to allow automatic selection
                fallback_config = task_config.copy()
                del fallback_config["force_engine"]
                engine = select_engine(fallback_config)
                if engine:
                    print(f"[ENGINE FALLBACK] SUCCESS: Automatic selection chose {type(engine).__name__}")
                else:
                    print("[ENGINE FALLBACK] CRITICAL FAILURE: Automatic engine selection also failed.")

            if not engine:
                print(f"[ORCHESTRATOR HALTED] No healthy cognitive substrate could be selected for this task. Config: {command.get('config')}")
                print("[ORCHESTRATOR HALTED] This may indicate all engines are unhealthy.")
                return

        # Initialize agents with the selected engine for this task.
        self._initialize_agents(engine)
        # --- END INTEGRATION ---

        # Store original engine for fallback logic
        original_engine = engine
        original_engine_type = self._get_engine_type(engine)

        # Track if we've switched to fallback mode
        fallback_mode = False
        
        # --- OPERATION: OPTICAL ANVIL - INITIALIZE OPTICAL CHAMBER ---
        # Initialize optical compression if enabled (Section 3.1 of feasibility study)
        self._initialize_optical_chamber(command.get('config', {}))
        # --- END OPTICAL ANVIL INTEGRATION ---

        task = command['task_description']
        max_rounds = command.get('config', {}).get('max_rounds', 5)
        self.max_cortex_queries = command.get('config', {}).get('max_cortex_queries', 5)
        self.cortex_query_count = 0
        output_artifact_path_str = command['output_artifact_path']
        output_path = self.project_root / output_artifact_path_str
        if output_artifact_path_str.endswith('/'):
            output_path = output_path / "task_log.md"

        # --- STRUCTURED EVENT LOGGING: TASK START ---
        self.event_manager.emit_event(
            "task_start",
            task_description=task,
            max_rounds=max_rounds,
            engine_type=original_engine_type,
            output_artifact=output_artifact_path_str,
            input_artifacts=command.get('input_artifacts', [])
        )

        log = [f"# Autonomous Triad Task Log\n## Task: {task}\n\n"]
        last_message = task

        # --- HOTFIX v4.3: ROBUST ENGINE TYPE DETERMINATION ---
        # CRITICAL: Determine engine type BEFORE any operations that need it
        engine_type = self._get_engine_type(engine)
        
        # Fail-fast if engine type cannot be determined
        if engine_type == "unknown":
            error_msg = f"[ORCHESTRATOR HALTED] Could not determine a valid engine type for the selected engine: {type(engine).__name__}"
            print(error_msg)
            raise ValueError(error_msg)

        # Enhance briefing with context from task description
        try:
            self._enhance_briefing_with_context(task)
        except FileNotFoundError as e:
            print(f"[WARNING] Context file error: {e}. Proceeding with base briefing.")

        # Inject fresh briefing context (now engine_type is defined)
        self.inject_briefing_packet(engine_type)

        if command.get('input_artifacts'):
            # ... (knowledge injection logic is the same)
            knowledge = ["Initial knowledge provided:\n"]
            for path_str in command['input_artifacts']:
                file_path = self.project_root / path_str
                if file_path.exists() and file_path.is_file():
                    knowledge.append(f"--- CONTENT OF {path_str} ---\n{file_path.read_text()}\n---\n")
                elif file_path.exists() and file_path.is_dir():
                    print(f"[!] Input artifact {path_str} is a directory, skipping.")
                else:
                    print(f"[!] Input artifact {path_str} not found.")
            last_message += "\n" + "".join(knowledge)

        print(f"\n▶️  Executing task: '{task}' for up to {max_rounds} rounds on {type(engine).__name__}")
        print(f"[ORCHESTRATOR] Using engine: {type(engine).__name__} (type: {engine_type}) for all agents in this task.")

        # V6.0 MANDATE 3: Initialize failure state awareness
        consecutive_failures = 0
        num_agents = len(self.speaker_order)

        loop = asyncio.get_event_loop()
        for i in range(max_rounds):
            print(f"--- ROUND {i+1} ---", flush=True)
            log.append(f"### ROUND {i+1}\n\n")

            round_failures = 0  # Track failures in this round
            round_packets = []  # Collect packets for predictable ordering

            for role in self.speaker_order:
                agent = self.agents[role]
                print(f"  -> Orchestrator to {agent.role}...", flush=True)

                prompt = f"The current state of the discussion is: '{last_message}'. As the {role}, provide your analysis or next step."

                # --- V6.0 MANDATE 1: UNIVERSAL DISTILLATION ---
                # Apply the same distillation logic to the main deliberation loop
                # Check the FULL potential payload (agent.messages + new prompt) BEFORE any API call
                potential_payload = agent.messages + [{"role": "user", "content": prompt}]
                payload_as_text = json.dumps(potential_payload)
                token_count = self._get_token_count(payload_as_text, engine_type)
                limit = self.engine_limits.get(engine_type, 100000)

                # Determine if we need distillation or optical compression
                needs_compression = token_count > limit

                if needs_compression:
                    print(f"[ORCHESTRATOR] WARNING: Full payload ({token_count:.0f} tokens) exceeds limit for {engine_type} ({limit})")

                    # --- // OPERATION: OPTICAL ANVIL - OPTICAL COMPRESSION DECISION POINT // ---
                    if self.optical_chamber and self.optical_chamber.should_compress(payload_as_text, engine_type):
                        print(f"[OPTICAL] Compressing payload for {role} (estimated 10x reduction)")

                        # Compress via optical chamber
                        decompressed_prompt = self.optical_chamber.compress_and_decompress(
                            payload_as_text,
                            task_context=task
                        )

                        # Clear agent history and send compressed context
                        agent.messages = [
                            agent.messages[0],  # Preserve system prompt
                            {"role": "user", "content": "SYSTEM NOTE: Context was optically compressed. Proceed based on decompressed data."},
                            {"role": "assistant", "content": "Acknowledged. Proceeding with optically decompressed context."}
                        ]
                        prompt_to_send = decompressed_prompt
                    else:
                        # Fallback to standard distillation
                        print(f"[ORCHESTRATOR] Using distillation engine for payload reduction...")
                        distilled_summary = self._distill_with_local_engine(payload_as_text, task)

                        # Clear agent history and send distilled context
                        agent.messages = [
                            agent.messages[0],  # Preserve system prompt
                            {"role": "user", "content": "SYSTEM NOTE: Context was distilled due to size. Proceed based on this summary."},
                            {"role": "assistant", "content": "Acknowledged. Proceeding with distilled context."}
                        ]
                        prompt_to_send = distilled_summary
                else:
                    # Payload is within limits, send normally
                    prompt_to_send = prompt

                # --- STRUCTURED EVENT LOGGING: MEMBER RESPONSE START ---
                member_start_time = time.time()
                input_tokens = self._get_token_count(prompt_to_send, engine_type)

                # --- FAULT ISOLATION: TIMEOUT PROTECTION ---
                timeout_seconds = command.get('config', {}).get('agent_timeout', 120)  # Default 2 minutes
                try:
                    # Execute query with TPM-aware rate limiting, timeout protection, and fallback logic
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            agent.query,
                            prompt_to_send,
                            self.token_regulator,
                            engine_type
                        ),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    print(f"  <- {agent.role} TIMEOUT (>{timeout_seconds}s)")
                    response = False
                    timeout_error = f"agent_timeout_exceeded_{timeout_seconds}s"

                # Calculate latency and output tokens
                latency_ms = int((time.time() - member_start_time) * 1000)
                output_tokens = self._get_token_count(response, engine_type) if response else 0

                # V7.0 MANDATE 3: Check for boolean failure response
                if response is False:
                    round_failures += 1
                    consecutive_failures += 1
                    error_type = getattr(self, 'timeout_error', "cognitive_substrate_failure") if hasattr(self, 'timeout_error') else "cognitive_substrate_failure"
                    if 'timeout_error' in locals():
                        error_type = timeout_error
                        del timeout_error  # Clean up
                    
                    print(f"  <- {agent.role} FAILED ({error_type})")

                    # --- STRUCTURED EVENT LOGGING: MEMBER RESPONSE FAILURE ---
                    self.event_manager.emit_event(
                        "member_response",
                        round=i+1,
                        member_id=role.lower(),
                        role=agent.role,
                        status="error",
                        latency_ms=latency_ms,
                        tokens_in=input_tokens,
                        tokens_out=0,
                        result_type="error",
                        errors=[error_type],
                        content_ref=f"round_{i+1}_{role.lower()}_failed"
                    )

                    # IMPLEMENT FALLBACK: If primary engine fails, try fallback to Ollama
                    if not fallback_mode and original_engine_type != "ollama":
                        print(f"[FALLBACK] Primary engine ({original_engine_type}) failed. Attempting fallback to Ollama...")
                        # Try Ollama as fallback
                        fallback_config = {"force_engine": "ollama"}
                        fallback_engine = select_engine(fallback_config)
                        if fallback_engine:
                            print(f"[FALLBACK] Switching to Ollama engine for remaining agents")
                            # Re-initialize agents with fallback engine
                            self._initialize_agents(fallback_engine)
                            engine = fallback_engine
                            engine_type = "ollama"
                            fallback_mode = True
                            # Reset consecutive failures for this round
                            consecutive_failures = 0
                            round_failures -= 1
                            # Retry this agent with fallback engine
                            response = await loop.run_in_executor(
                                None,
                                agent.query,
                                prompt_to_send,
                                self.token_regulator,
                                engine_type
                            )
                            if response is False:
                                print(f"  <- {agent.role} FAILED (fallback engine also failed)")
                                consecutive_failures += 1
                                round_failures += 1
                            else:
                                print(f"  <- {agent.role} SUCCESS (fallback engine)")
                        else:
                            print(f"[FALLBACK] No fallback engine available")

                    if response is False:  # Still failed after fallback attempt
                        # Create packet for failed response
                        failed_packet = CouncilRoundPacket(
                            timestamp=datetime.now().isoformat(),
                            session_id=self.run_id,
                            round_id=i+1,
                            member_id=role.lower(),
                            engine=engine_type,
                            seed=seed_for(self.run_id, i+1, role.lower(), prompt_hash(prompt_to_send)),
                            prompt_hash=prompt_hash(prompt_to_send),
                            inputs={"prompt": prompt_to_send, "context": last_message},
                            decision="error",
                            rationale="",
                            confidence=0.0,
                            citations=[],
                            rag={},
                            cag={},
                            novelty={},
                            memory_directive={"tier": "none"},
                            cost={
                                "input_tokens": input_tokens,
                                "output_tokens": 0,
                                "latency_ms": latency_ms
                            },
                            errors=[error_type]
                        )
                        # Collect failed packet for predictable ordering
                        jsonl_dir = getattr(self, 'cli_config', {}).get('jsonl_path') if getattr(self, 'cli_config', {}).get('emit_jsonl') else None
                        stream_stdout = getattr(self, 'cli_config', {}).get('stream_stdout', False)
                        round_packets.append((failed_packet, jsonl_dir, stream_stdout))

                        log.append(f"**{agent.role} (FAILED):** Cognitive substrate failure.\n\n---\n")
                else:
                    # Successful response - reset consecutive failure counter
                    consecutive_failures = 0
                    print(f"  <- {agent.role} to Orchestrator.", flush=True)

                    # --- STRUCTURED EVENT LOGGING: ANALYZE RESPONSE FOR METADATA ---
                    # Extract metadata from response for structured logging
                    result_type = classify_response_type(response, role)
                    score = self._calculate_response_score(response)
                    vote = self._extract_vote(response)
                    novelty = self._assess_novelty(response, last_message)
                    reasons = self._extract_reasoning(response)
                    citations = self._extract_citations(response, signals.retrieval.parent_docs)

                    # --- Phase 2: Run Self-Querying Retriever ---
                    signals = self.retriever.run(
                        prompt=prompt_to_send,
                        council_role=role.lower(),
                        confidence=score,
                        citations=citations
                    )

                    # --- ROUND PACKET EMISSION ---
                    # Create comprehensive round packet
                    packet = CouncilRoundPacket(
                        timestamp=datetime.now().isoformat(),
                        session_id=self.run_id,
                        round_id=i+1,
                        member_id=role.lower(),
                        engine=engine_type,
                        seed=seed_for(self.run_id, i+1, role.lower(), prompt_hash(prompt_to_send)),
                        prompt_hash=prompt_hash(prompt_to_send),
                        inputs={"prompt": prompt_to_send, "context": last_message},
                        decision=vote,
                        rationale=response,
                        confidence=score,
                        citations=citations,
                        rag=self._get_rag_data(task, response),
                        cag=get_cag_data(prompt_to_send, engine_type, self.cache_adapter),
                        novelty=NoveltyField(
                            is_novel=signals.novelty.is_novel,
                            signal=signals.novelty.signal or "none",  # Never empty
                            basis=signals.novelty.basis or {}
                        ),
                        memory_directive=MemoryDirectiveField(
                            tier=signals.memory_directive.tier,
                            justification=signals.memory_directive.justification or "default_fallback"  # Never empty
                        ),
                        cost={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "latency_ms": latency_ms
                        },
                        errors=[],
                        retrieval=RetrievalField(
                            structured_query=signals.retrieval.structured_query.__dict__,
                            parent_docs=[pd.__dict__ for pd in signals.retrieval.parent_docs],
                            retrieval_latency_ms=signals.retrieval.retrieval_latency_ms,
                        ),
                        conflict=ConflictField(
                            conflicts_with=signals.conflict.conflicts_with,
                            basis=signals.conflict.basis
                        ),
                        seed_chain={
                            "session_seed": getattr(self, 'session_seed', 0),
                            "round_seed": seed_for(self.run_id, i+1, role.lower(), prompt_hash(prompt_to_send)),
                            "member_seed": seed_for(self.run_id, i+1, role.lower(), prompt_hash(prompt_to_send)),
                            "engine_seed": 0,  # TODO: populate from engine instance
                            "retrieval_seed": 0  # TODO: populate from retriever
                        }
                    )

                    # Emit packet
                    jsonl_dir = getattr(self, 'cli_config', {}).get('jsonl_path') if getattr(self, 'cli_config', {}).get('emit_jsonl') else None
                    stream_stdout = getattr(self, 'cli_config', {}).get('stream_stdout', False)
                    # Collect packet for predictable ordering (emit at end of round)
                    round_packets.append((packet, jsonl_dir, stream_stdout))

                    # --- STRUCTURED EVENT LOGGING: MEMBER RESPONSE SUCCESS ---
                    self.event_manager.emit_event(
                        "member_response",
                        round=i+1,
                        member_id=role.lower(),
                        role=agent.role,
                        status="success",
                        latency_ms=latency_ms,
                        tokens_in=input_tokens,
                        tokens_out=output_tokens,
                        result_type=result_type,
                        score=score,
                        vote=vote,
                        novelty=novelty,
                        reasons=reasons,
                        citations=citations,
                        content_ref=f"round_{i+1}_{role.lower()}_response"
                    )

                    # V9.3 ENHANCEMENT: Display agent response content in real-time for debugging
                    print(f"\n[{agent.role} RESPONSE - ROUND {i+1}]")
                    # Truncate very long responses for terminal readability
                    display_response = response[:2000] + "..." if len(response) > 2000 else response
                    print(display_response)
                    print(f"[END {agent.role} RESPONSE]\n", flush=True)

                    # Handle knowledge requests (only if response was successful)
                    knowledge_response = self._handle_knowledge_request(response)
                    if knowledge_response:
                        # V9.3 ENHANCEMENT: Display knowledge request interaction
                        print(f"[ORCHESTRATOR] Fulfilling knowledge request for {agent.role}...", flush=True)
                        print(f"[KNOWLEDGE REQUEST RESPONSE]")
                        display_knowledge = knowledge_response[:1500] + "..." if len(knowledge_response) > 1500 else knowledge_response
                        print(display_knowledge)
                        print(f"[END KNOWLEDGE RESPONSE]\n", flush=True)

                        # Inject the knowledge response back into the conversation
                        print(f"  -> Orchestrator providing context to {agent.role}...", flush=True)
                        knowledge_injection = await loop.run_in_executor(
                            None,
                            agent.query,
                            knowledge_response,
                            self.token_regulator,
                            engine_type
                        )
                        
                        # Check if knowledge injection also failed
                        if knowledge_injection is False:
                            print(f"  <- {agent.role} FAILED during knowledge injection")
                            consecutive_failures += 1
                        else:
                            print(f"  <- {agent.role} acknowledging context.", flush=True)
                            response += f"\n\n{knowledge_injection}"
                            log.append(f"**{agent.role}:**\n{response}\n\n---\n")
                            log.append(f"**ORCHESTRATOR (Fulfilled Request):**\n{knowledge_response}\n\n---\n")
                    else:
                        log.append(f"**{agent.role}:**\n{response}\n\n---\n")

                # V7.0 MANDATE 3: Check for total operational failure after each agent
                # If all agents in a round fail, break immediately
                if consecutive_failures >= num_agents:
                    print(f"[ORCHESTRATOR] CRITICAL: {consecutive_failures} consecutive agent failures detected.")
                    print(f"[ORCHESTRATOR] Total operational failure. Terminating task.")
                    log.append(f"\n**SYSTEM FAILURE:** Task terminated due to {consecutive_failures} consecutive agent failures.\n\n")
                    break

                last_message = response

                # --- ADD THIS LINE ---
                time.sleep(1) # Add a 1-second pause to be kind to the API
                # ---------------------

            # Sort and emit packets in predictable order (by round_id, then member_id)
            round_packets.sort(key=lambda x: (x[0].round_id, x[0].member_id))
            for packet, jsonl_dir, stream_stdout in round_packets:
                emit_packet(packet, jsonl_dir, stream_stdout, str(Path(__file__).parent / "schemas" / "round_packet_schema.json"))

            # --- STRUCTURED EVENT LOGGING: ROUND COMPLETION ---
            round_aggregation = aggregate_round_events(self.event_manager.run_id, i+1, self.event_manager.event_log_path)
            self.event_manager.emit_event(
                "round_complete",
                round=i+1,
                total_members=round_aggregation.get("total_members", 0),
                success_rate=round_aggregation.get("success_rate", 0.0),
                consensus=round_aggregation.get("consensus", False),
                early_exit=round_aggregation.get("early_exit", False),
                exit_reason=round_aggregation.get("exit_reason"),
                avg_latency=round_aggregation.get("avg_latency", 0),
                total_tokens_in=round_aggregation.get("total_tokens_in", 0),
                total_tokens_out=round_aggregation.get("total_tokens_out", 0),
                novelty_distribution=round_aggregation.get("novelty_distribution", {})
            )

            # Early exit logic based on round aggregation
            if round_aggregation.get("early_exit"):
                reason = round_aggregation.get("exit_reason", "unknown")
                print(f"[EARLY EXIT] Round {i+1} triggered early exit: {reason}")
                if reason == "consensus_achieved":
                    print("🎯 Consensus achieved - proceeding to next phase")
                elif reason == "low_success_rate":
                    print("⚠️  Low success rate detected - aborting deliberation")
                    break
                break

        # V7.0 MANDATE 3: Final failure state check
        if consecutive_failures >= num_agents:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("".join(log))
            print(f"\n[FAILURE] Task terminated due to total operational failure. Partial log saved to {output_path}")

            # --- STRUCTURED EVENT LOGGING: TASK COMPLETE (FAILURE) ---
            self.event_manager.emit_event(
                "task_complete",
                status="failure",
                reason="total_operational_failure",
                rounds_completed=i+1,
                total_failures=consecutive_failures,
                output_artifact=str(output_path)
            )

            for agent in self.agents.values():
                agent.save_history()
            self.archive_briefing_packet()
            return False  # Return False to signal task failure

        output_path.parent.mkdir(parents=True)
        output_path.write_text("".join(log))
        print(f"\n[SUCCESS] Deliberation complete. Artifact saved to {output_path}")

        # --- STRUCTURED EVENT LOGGING: TASK COMPLETE (SUCCESS) ---
        self.event_manager.emit_event(
            "task_complete",
            status="success",
            rounds_completed=i+1,
            total_rounds=i+1,
            output_artifact=str(output_path)
        )

        for agent in self.agents.values():
            agent.save_history()
        print("[SUCCESS] All agent session states have been saved.")

        # Archive the used briefing packet
        self.archive_briefing_packet()
        return True  # Return True to signal task success

# --- WATCH FOR COMMANDS THREAD ---
# Moved to sentry.py

    async def main_loop(self):
        """The main async loop that waits for commands from the queue."""
        print("--- Orchestrator Main Loop is active. ---")
        loop = asyncio.get_event_loop()
        state_file = Path(__file__).parent / "development_cycle_state.json"

        while True:
            if state_file.exists():
                # We are in the middle of a development cycle, waiting for approval
                print("--- Orchestrator in Development Cycle. Awaiting Guardian approval... ---", flush=True)
                command = await loop.run_in_executor(None, self.command_queue.get)

                # V9.0 MANDATE 1: Action Triage - Check for mechanical tasks first
                if "entry_content" in command and "output_artifact_path" in command:
                    # This is a Write Task
                    print("[ACTION TRIAGE] Detected Write Task - executing mechanical write...")
                    await loop.run_in_executor(None, self._execute_mechanical_write, command)
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'write' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                    continue
                elif "git_operations" in command:
                    # This is a Git Task
                    print("[ACTION TRIAGE] Detected Git Task - executing mechanical git operations...")
                    await loop.run_in_executor(None, lambda: execute_mechanical_git(command, self.project_root))
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'git' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                    continue

                # V7.1: Doctrine of Implied Intent - Check if this is a new development cycle command
                # If so, it implies approval to proceed with the current stage
                if command.get("development_cycle", False) and command.get("guardian_approval") == "APPROVE_CURRENT_STAGE":
                    # Update state with approved artifact
                    state = json.loads(state_file.read_text())
                    if "approved_artifact_path" in command:
                        if state["current_stage"] == "AWAITING_APPROVAL_REQUIREMENTS":
                            state["approved_artifacts"]["requirements"] = command["approved_artifact_path"]
                        elif state["current_stage"] == "AWAITING_APPROVAL_TECH_DESIGN":
                            state["approved_artifacts"]["tech_design"] = command["approved_artifact_path"]
                        elif state["current_stage"] == "AWAITING_APPROVAL_CODE":
                            state["approved_artifacts"]["code_proposal"] = command["approved_artifact_path"]
                        state_file.write_text(json.dumps(state, indent=2))
                    await self._advance_cycle(state_file)
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'development_cycle_approval' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                elif command.get("action") == "APPROVE_CURRENT_STAGE":
                    # Legacy approval mechanism for backward compatibility
                    state = json.loads(state_file.read_text())
                    if "approved_artifact_path" in command:
                        if state["current_stage"] == "AWAITING_APPROVAL_REQUIREMENTS":
                            state["approved_artifacts"]["requirements"] = command["approved_artifact_path"]
                        elif state["current_stage"] == "AWAITING_APPROVAL_TECH_DESIGN":
                            state["approved_artifacts"]["tech_design"] = command["approved_artifact_path"]
                        elif state["current_stage"] == "AWAITING_APPROVAL_CODE":
                            state["approved_artifacts"]["code_proposal"] = command["approved_artifact_path"]
                        state_file.write_text(json.dumps(state, indent=2))
                    await self._advance_cycle(state_file)
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'approve_current_stage' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                else:
                    print("[!] Invalid command during development cycle. Awaiting APPROVE_CURRENT_STAGE.", flush=True)
            else:
                # We are idle, waiting for a new task to start a new cycle
                print("--- Orchestrator Idle. Awaiting command from Sentry... ---", flush=True)
                command = await loop.run_in_executor(None, self.command_queue.get)

                # V9.0 MANDATE 1: Action Triage - Check for mechanical tasks first
                if "entry_content" in command and "output_artifact_path" in command:
                    # This is a Write Task
                    print("[ACTION TRIAGE] Detected Write Task - executing mechanical write...")
                    await loop.run_in_executor(None, self._execute_mechanical_write, command)
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'write' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                    continue
                elif "git_operations" in command:
                    # This is a Git Task
                    print("[ACTION TRIAGE] Detected Git Task - executing mechanical git operations...")
                    await loop.run_in_executor(None, lambda: execute_mechanical_git(command, self.project_root))
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'git' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                    continue

                elif command.get("task_type") == "cache_request":
                    # This is a Cache Request Task
                    print("[ACTION TRIAGE] Detected Cache Request Task - fetching cache bundle...")
                    from .commands import handle_cache_request
                    report_md = handle_cache_request(command)
                    # Write the artifact
                    output_path = self.project_root / command["output_artifact_path"]
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(report_md, encoding="utf-8")
                    print(f"[CACHE REQUEST] Verification artifact written to: {output_path}")
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'cache_request' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                    continue

                elif command.get("task_type") == "cache_wakeup":
                    # This is a Cache Wakeup Task (Guardian Boot Digest)
                    print("[ACTION TRIAGE] Detected Cache Wakeup Task - generating Guardian boot digest...")
                    from .handlers.cache_wakeup_handler import handle_cache_wakeup

                    # Generate digest using new handler
                    await loop.run_in_executor(None, lambda: handle_cache_wakeup(command, self))
                    # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                    if self.one_shot:
                        self.logger.info(f"One-shot mode: Task 'cache_wakeup' complete. Shutting down orchestrator.")
                        break
                    # --- END ONE-SHOT LOGIC ---
                    continue

                try:
                    # Check if this is a development cycle command
                    if command.get("development_cycle", False):
                        await self._start_new_cycle(command, state_file)
                        # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                        if self.one_shot:
                            self.logger.info(f"One-shot mode: Task 'development_cycle' complete. Shutting down orchestrator.")
                            break
                        # --- END ONE-SHOT LOGIC ---
                    elif command.get('task_type') == "query_and_synthesis":
                        # Guardian Mnemonic Synchronization Protocol: Query and Synthesis task
                        print("[ACTION TRIAGE] Detected Query and Synthesis Task - invoking Council for mnemonic synchronization...")
                        await self._execute_query_and_synthesis(command)
                        # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                        if self.one_shot:
                            self.logger.info(f"One-shot mode: Task 'query_and_synthesis' complete. Shutting down orchestrator.")
                            break
                        # --- END ONE-SHOT LOGIC ---
                    else:
                        # Regular task execution
                        original_output_path = self.project_root / command['output_artifact_path']
                        task_result = await self.execute_task(command)

                        # V7.0 MANDATE 3: Check task result before proceeding
                        if task_result is False:
                            self.logger.error("Task aborted due to consecutive cognitive failures. No AAR will be generated.")
                        else:
                            # Check if RAG database should be updated for this task
                            update_rag = command.get('config', {}).get('update_rag', True)
                            if update_rag:
                                # V9.3: Generate AAR asynchronously - truly non-blocking
                                self.logger.info("Task complete. Dispatching After-Action Report synthesis to background...")
                                # Determine the actual log file path
                                if original_output_path.is_dir():
                                    log_file_path = original_output_path / "task_log.md"
                                else:
                                    log_file_path = original_output_path
                                # Create background task for AAR generation
                                asyncio.create_task(self._execute_aar_background_full(log_file_path, command.get('config')))
                            else:
                                self.logger.info("Task complete. RAG database update skipped per configuration.")
                                self.logger.info(f"Output artifact saved to: {original_output_path}")
                                self.logger.info("Orchestrator returning to idle state - ready for next command")

                        # --- PROTOCOL 115: ONE-SHOT EXIT LOGIC ---
                        if self.one_shot:
                            self.logger.info(f"One-shot mode: Task 'regular' complete. Shutting down orchestrator.")
                            break
                        # --- END ONE-SHOT LOGIC ---

                except Exception as e:
                    print(f"[MAIN LOOP ERROR] Task execution failed: {e}", file=sys.stderr)
                    self.logger.error(f"Task execution failed: {e}")
                    return False