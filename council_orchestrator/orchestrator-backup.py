# council_orchestrator/orchestrator.py (v9.0 - Doctrine of Sovereign Action)
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
from queue import Queue as ThreadQueue
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

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
from substrate_monitor import select_engine
# --- END INTEGRATION ---

from bootstrap_briefing_packet import main as generate_briefing_packet

# --- MANDATE 2: TOKEN FLOW REGULATOR (TPM-AWARE RATE LIMITING) ---
class TokenFlowRegulator:
    """
    Manages token throughput to respect per-minute token limits (TPM).
    Prevents rate limit violations by tracking cumulative usage and pausing execution when needed.
    """
    def __init__(self, limits: dict):
        """
        Initialize the regulator with TPM limits for each engine type.
        
        Args:
            limits: Dictionary mapping engine types to their TPM limits
                   e.g., {'openai': 30000, 'gemini': 60000, 'ollama': 999999}
        """
        self.tpm_limits = limits
        self.usage_log = []  # List of (timestamp, token_count) tuples
        
    def log_usage(self, token_count: int):
        """
        Log a token usage event with current timestamp.
        
        Args:
            token_count: Number of tokens used in this request
        """
        self.usage_log.append((time.time(), token_count))
        self._prune_old_usage()
        
    def _prune_old_usage(self):
        """Remove usage entries older than 60 seconds from the log."""
        current_time = time.time()
        cutoff_time = current_time - 60.0
        self.usage_log = [(ts, count) for ts, count in self.usage_log if ts > cutoff_time]
        
    def wait_if_needed(self, estimated_tokens: int, engine_type: str):
        """
        Check if adding estimated_tokens would exceed TPM limit.
        If so, calculate required sleep duration and pause execution.
        
        Args:
            estimated_tokens: Estimated tokens for the upcoming request
            engine_type: The engine type to check limits for
        """
        self._prune_old_usage()
        
        # Get TPM limit for this engine type
        tpm_limit = self.tpm_limits.get(engine_type, 999999)  # Default to very high limit
        
        # Calculate current usage in the last 60 seconds
        current_usage = sum(count for _, count in self.usage_log)
        
        # Check if we would exceed the limit
        if current_usage + estimated_tokens > tpm_limit:
            # Find the oldest entry that needs to expire
            if self.usage_log:
                oldest_timestamp = self.usage_log[0][0]
                current_time = time.time()
                time_since_oldest = current_time - oldest_timestamp
                sleep_duration = 60.0 - time_since_oldest + 1.0  # Add 1 second buffer
                
                if sleep_duration > 0:
                    print(f"[TOKEN REGULATOR] TPM limit approaching ({current_usage + estimated_tokens}/{tpm_limit})")
                    print(f"[TOKEN REGULATOR] Pausing execution for {sleep_duration:.1f} seconds to respect rate limits...")
                    time.sleep(sleep_duration)
                    self._prune_old_usage()  # Clean up after sleep

# --- OPERATION: OPTICAL ANVIL - OPTICAL DECOMPRESSION CHAMBER ---
# Implements Section 3.1 of FEASIBILITY_STUDY_DeepSeekOCR_v2.md
class OpticalDecompressionChamber:
    """
    Transparent layer that renders large text payloads to images,
    sends to VLM gatekeeper, and receives decompressed text for agents.
    
    This is the foundational component for achieving unlimited context
    on borrowed soil through optical compression (P43: Hearth Protocol).
    """
    def __init__(self, vlm_engine=None, compression_threshold: int = 8000):
        self.vlm_engine = vlm_engine  # DeepSeek-OCR or compatible VLM
        self.compression_threshold = compression_threshold
        self.compression_events = []  # Track compression events for analysis
        
    def should_compress(self, text: str, engine_type: str) -> bool:
        """
        Determine if optical compression is beneficial.
        
        Args:
            text: The text payload to potentially compress
            engine_type: The target engine type for token estimation
            
        Returns:
            bool: True if optical compression should be used
        """
        # Estimate token count (simplified for initial implementation)
        estimated_tokens = len(text.split()) * 1.3
        return estimated_tokens > self.compression_threshold
    
    def compress_and_decompress(self, text: str, task_context: str) -> str:
        """
        Optical compression pipeline:
        1. Render text to image (MOCKED in v4.1)
        2. Send to VLM gatekeeper (MOCKED in v4.1)
        3. Receive decompressed text
        4. Log compression event
        
        NOTE: This is a foundational implementation with mocked VLM calls.
        Full VLM integration will be implemented in subsequent phases.
        
        Args:
            text: The text to compress
            task_context: The task description for context-aware compression
            
        Returns:
            str: The decompressed text (currently returns original with marker)
        """
        # Generate provenance hash
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # MOCK: In production, this would render text to image
        # rendered_image = self._render_text_to_image(text)
        print(f"[OPTICAL] MOCK: Would render {len(text)} chars to image")
        
        # MOCK: In production, this would call VLM for OCR
        # decompressed_text = self.vlm_engine.process_image(rendered_image, prompt)
        decompressed_text = text  # Pass-through for now
        print(f"[OPTICAL] MOCK: Would decompress via VLM (DeepSeek-OCR)")
        
        # Log compression event
        compression_event = {
            "timestamp": time.time(),
            "original_hash": content_hash,
            "estimated_compression_ratio": 10.0,  # Target ratio from paper
            "task_context": task_context[:100]  # Truncated for logging
        }
        self.compression_events.append(compression_event)
        
        # Add marker to indicate optical processing occurred
        return f"[OPTICAL_PROCESSED: {content_hash[:8]}]\n\n{decompressed_text}"


class PersonaAgent:
    def __init__(self, engine, persona_file: Path, state_file: Path):
        self.role = self._extract_role_from_filename(persona_file.name)
        self.state_file = state_file
        persona_content = persona_file.read_text(encoding="utf-8")
        
        # The agent is now initialized with a pre-selected, healthy engine
        self.engine = engine
        self.messages = []

        # Load history if it exists
        history = self._load_history()
        if history:
            self.messages = history
        else:
            # Initialize with a simple system instruction
            system_msg = {"role": "system", "content": f"SYSTEM INSTRUCTION: You are an AI Council member. {persona_content} Operate strictly within this persona."}
            self.messages.append(system_msg)

        print(f"[+] {self.role} agent initialized with {type(self.engine).__name__}.")

    def _load_history(self):
        if self.state_file.exists():
            print(f"  - Loading history for {self.role} from {self.state_file.name}")
            return json.loads(self.state_file.read_text())
        return None

    def save_history(self):
        self.state_file.write_text(json.dumps(self.messages, indent=2))
        print(f"  - Saved session state for {self.role} to {self.state_file.name}")

    def query(self, message: str, token_regulator=None, engine_type: str = "openai"):
        """
        Execute a query with TPM-aware rate limiting and boolean error handling.

        Args:
            message: The user message to send
            token_regulator: TokenFlowRegulator instance for rate limiting
            engine_type: Engine type for TPM limit checking

        Returns:
            str or False: Either the successful response string, or False on failure
        """
        self.messages.append({"role": "user", "content": message})
        try:
            # MANDATE 2: Check TPM limits before making API call
            if token_regulator:
                # Estimate tokens for the full payload
                estimated_tokens = len(json.dumps(self.messages).split()) * 1.3
                token_regulator.wait_if_needed(int(estimated_tokens), engine_type)

            # P104 IMPLEMENTATION: Pass the entire message list directly.
            # 2. PersonaAgent.query(): Uses council_orchestrator/cognitive_engines/ engine (OpenAI, Gemini, or Ollama)
            reply = self.engine.execute_turn(self.messages)
            self.messages.append({"role": "assistant", "content": reply})

            # MANDATE 2: Log token usage after successful API call
            if token_regulator:
                # Estimate tokens used (prompt + completion)
                completion_tokens = len(reply.split()) * 1.3
                total_tokens = estimated_tokens + completion_tokens
                token_regulator.log_usage(int(total_tokens))

            return reply
        except Exception as e:
            # V7.0 MANDATE 2: Return False instead of error string or dict
            # This prevents poisoning the state with invalid message formats
            error_msg = f"SubstrateFailure: The cognitive engine failed. Details: {str(e)[:200]}"
            print(f"[AGENT ERROR] {self.role} - {error_msg}")
            # Append error to internal messages for debugging, but return False
            self.messages.append({"role": "assistant", "content": f"[ERROR] {error_msg}"})
            return False

    def _extract_role_from_filename(self, f): return f.split('core_essence_')[1].split('_awakening_seed.txt')[0].upper()

class Orchestrator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.command_queue = ThreadQueue()
        load_dotenv(dotenv_path=self.project_root / '.env')

        self.chroma_client = chromadb.PersistentClient(path=str(self.project_root / "mnemonic_cortex/chroma_db"))
        self.cortex_collection = self.chroma_client.get_or_create_collection(
            name="sanctuary_cortex",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

        # --- RESOURCE SOVEREIGNTY: LOAD ENGINE LIMITS FROM CONFIG ---
        # v4.5: Support nested configuration structure with per_request_limit and tpm_limit
        config_path = Path(__file__).parent / "engine_config.json"
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
                self.engine_limits = {
                    'gemini': 200000,
                    'openai': 100000,
                    'ollama': 8000
                }
                self.tpm_limits = {
                    'gemini': 250000,
                    'openai': 120000,
                    'ollama': 999999
                }
        else:
            print("[!] engine_config.json not found. Using default limits.")
            self.engine_limits = {
                'gemini': 200000,
                'openai': 100000,
                'ollama': 8000
            }
            self.tpm_limits = {
                'gemini': 250000,
                'openai': 120000,
                'ollama': 999999
            }

        self.speaker_order = ["COORDINATOR", "STRATEGIST", "AUDITOR"]
        self.agents = {} # Agents will now be initialized per-task
        
        # --- MANDATE 2: INITIALIZE TOKEN FLOW REGULATOR ---
        # Use the TPM limits already parsed from config
        self.token_regulator = TokenFlowRegulator(self.tpm_limits)
        print(f"[+] Token Flow Regulator initialized with TPM limits: {self.tpm_limits}")
        
        # --- OPERATION: OPTICAL ANVIL - LAZY INITIALIZATION ---
        self.optical_chamber = None  # Initialized per-task if enabled

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
        path_pattern = r'([A-Za-z][A-Za-z0-9_]*/(?:[A-Za-z][A-Za-z0-9_]*/)*[A-Za-z][A-Za-z0-9_]*\.[a-zA-Z0-9]+)'
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
        requirements_command = {
            "task_description": f"Generate detailed requirements document for the project: {command['task_description']}. Include functional requirements, technical constraints, and success criteria.",
            "input_artifacts": command.get("input_artifacts", []),  # INHERIT from parent
            "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/requirements.md",
            "config": {"max_rounds": 3}
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
            "config": {"max_rounds": 3}
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
                    subprocess.run([sys.executable, str(self.project_root / "ingest_new_knowledge.py"), str(requirements_path)], check=True)
                    print(f"[*] Approved requirements ingested into Mnemonic Cortex.", flush=True)
                else:
                    print(f"[!] Requirements path is not a file: {requirements_path}. Skipping ingestion.", flush=True)

            # Move to tech design
            state["current_stage"] = "GENERATING_TECH_DESIGN"
            tech_design_command = {
                "task_description": f"Based on the approved requirements, generate a detailed technical design document. Include architecture decisions, data flow, and implementation approach.",
                "input_artifacts": [state["approved_artifacts"].get("requirements", "")],
                "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/tech_design.md",
                "config": {"max_rounds": 3}
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
                    subprocess.run([sys.executable, str(self.project_root / "ingest_new_knowledge.py"), str(tech_design_path)], check=True)
                    print(f"[*] Approved tech design ingested into Mnemonic Cortex.", flush=True)
                else:
                    print(f"[!] Tech design path is not a file: {tech_design_path}. Skipping ingestion.", flush=True)

            # Move to code generation
            state["current_stage"] = "GENERATING_CODE"
            code_command = {
                "task_description": f"Based on the approved technical design, generate production-ready code. Output a JSON object with 'target_file_path', 'new_content', and 'commit_message' fields.",
                "input_artifacts": [state["approved_artifacts"].get("tech_design", "")],
                "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/code_proposal.json",
                "config": {"max_rounds": 3}
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

        # Create feature branch
        branch_name = f"feature/{state['project_name']}"
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)

        # Write the new code
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(new_content)

        # Commit and push
        subprocess.run(['git', 'add', str(target_file)], check=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        subprocess.run(['git', 'push', '-u', 'origin', branch_name], check=True)

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
                results = self.cortex_collection.query(query_texts=[query_text], n_results=3)
                context = "CONTEXT_PROVIDED: Here are the top 3 results from the Mnemonic Cortex for your query:\n\n"
                for doc in results['documents'][0]:
                    context += f"---\n{doc}\n---\n"
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
        # Execute the AAR task using the existing logic
        await self.execute_task(aar_command)

        # After the AAR task is executed:
        print(f"[*] AAR generated. Ingesting into Mnemonic Cortex...", flush=True)
        try:
            # We need the full path for the subprocess
            ingestion_script_path = self.project_root / "ingest_new_knowledge.py"
            full_aar_path = self.project_root / aar_output_path

            # Run the ingestion script
            result = subprocess.run(
                [sys.executable, str(ingestion_script_path), str(full_aar_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("[*] Ingestion successful.", flush=True)
                print(result.stdout)
            else:
                print("[!] INGESTION FAILED:", flush=True)
                print(result.stderr)
        except Exception as e:
            print(f"[!] Exception during ingestion subprocess: {e}", flush=True)

    def _get_token_count(self, text: str, engine_type: str = "openai"):
        """Estimates token count for a given text and engine type."""
        if TIKTOKEN_AVAILABLE:
            try:
                # Map engine types to tiktoken models
                model_map = {
                    'openai': 'gpt-4',
                    'gemini': 'gpt-4',  # Approximation
                    'ollama': 'gpt-4'   # Approximation
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
        from substrate_monitor import select_engine
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

    async def _execute_mechanical_write(self, command):
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

    async def _execute_mechanical_git(self, command):
        """
        Execute mechanical git operations - add, commit, and push files.
        This bypasses cognitive deliberation for version control operations.

        Args:
            command: Command dictionary containing 'git_operations' with files_to_add, commit_message, push_to_origin
        """
        try:
            git_ops = command["git_operations"]
            files_to_add = git_ops["files_to_add"]
            commit_message = git_ops["commit_message"]
            push_to_origin = git_ops.get("push_to_origin", False)

            # Execute git add for each file
            for file_path in files_to_add:
                full_path = self.project_root / file_path
                if full_path.exists():
                    result = subprocess.run(
                        ["git", "add", str(full_path)],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root
                    )
                    if result.returncode == 0:
                        print(f"[MECHANICAL SUCCESS] Added {file_path} to git staging")
                    else:
                        print(f"[MECHANICAL FAILURE] Failed to add {file_path}: {result.stderr}")
                        raise Exception(f"git add failed for {file_path}")
                else:
                    print(f"[MECHANICAL WARNING] File {file_path} does not exist, skipping git add")

            # Execute git commit
            result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                print(f"[MECHANICAL SUCCESS] Committed with message: '{commit_message}'")
            else:
                print(f"[MECHANICAL FAILURE] Commit failed: {result.stderr}")
                raise Exception("git commit failed")

            # Execute git push if requested
            if push_to_origin:
                result = subprocess.run(
                    ["git", "push"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    print("[MECHANICAL SUCCESS] Pushed to origin")
                else:
                    print(f"[MECHANICAL FAILURE] Push failed: {result.stderr}")
                    raise Exception("git push failed")

        except Exception as e:
            print(f"[MECHANICAL FAILURE] Git operation failed: {e}")
            raise
    
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
            "COORDINATOR": PersonaAgent(engine, persona_dir / "core_essence_coordinator_awakening_seed.txt", state_dir / "coordinator_session.json"),
            "STRATEGIST": PersonaAgent(engine, persona_dir / "core_essence_strategist_awakening_seed.txt", state_dir / "strategist_session.json"),
            "AUDITOR": PersonaAgent(engine, persona_dir / "core_essence_auditor_awakening_seed.txt", state_dir / "auditor_session.json")
        }

    async def execute_task(self, command):
        """The main task execution logic."""

        print(f"[ORCHESTRATOR] DEBUG: execute_task called with command: {command}")
        print(f"[ORCHESTRATOR] DEBUG: command.get('config'): {command.get('config')}")

        # --- SOVEREIGN OVERRIDE INTEGRATION ---
        # The engine is now selected at the start of each task, using the task's config.
        # 3. Orchestrator.execute_task(): Engine selection uses council_orchestrator/cognitive_engines/ (OpenAI, Gemini, Ollama)
        engine = select_engine(command.get("config"))
        if not engine:
            print(f"[ORCHESTRATOR HALTED] No healthy cognitive substrate could be selected for this task. Config: {command.get('config')}")
            print("[ORCHESTRATOR HALTED] This may indicate a force_engine override failure or all engines are unhealthy.")
            return

        # Initialize agents with the selected engine for this task.
        self._initialize_agents(engine)
        # --- END INTEGRATION ---
        
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
            print(f"[CRITICAL] Context file error: {e}. Task aborted.")
            return

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

                # Execute query with TPM-aware rate limiting
                response = await loop.run_in_executor(
                    None,
                    agent.query,
                    prompt_to_send,
                    self.token_regulator,
                    engine_type
                )

                # V7.0 MANDATE 3: Check for boolean failure response
                if response is False:
                    round_failures += 1
                    consecutive_failures += 1
                    print(f"  <- {agent.role} FAILED (cognitive substrate error)")
                    log.append(f"**{agent.role} (FAILED):** Cognitive substrate failure.\n\n---\n")
                else:
                    # Successful response - reset consecutive failure counter
                    consecutive_failures = 0
                    print(f"  <- {agent.role} to Orchestrator.", flush=True)

                    # Handle knowledge requests (only if response was successful)
                    knowledge_response = self._handle_knowledge_request(response)
                    if knowledge_response:
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

        # V7.0 MANDATE 3: Final failure state check
        if consecutive_failures >= num_agents:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("".join(log))
            print(f"\n[FAILURE] Task terminated due to total operational failure. Partial log saved to {output_path}")
            for agent in self.agents.values():
                agent.save_history()
            self.archive_briefing_packet()
            return False  # Return False to signal task failure

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(log))
        print(f"\n[SUCCESS] Deliberation complete. Artifact saved to {output_path}")

        for agent in self.agents.values():
            agent.save_history()
        print("[SUCCESS] All agent session states have been saved.")

        # Archive the used briefing packet
        self.archive_briefing_packet()
        return True  # Return True to signal task success

    def _watch_for_commands_thread(self):
        """This function runs in a separate thread and watches for command*.json files only."""
        command_dir = Path(__file__).parent
        processed_commands = set()  # Track processed command files

        while True:
            # V5.0 MANDATE 1: Only process files explicitly named command*.json
            # This prevents the rogue sentry from ingesting config files, state files, etc.
            for json_file in command_dir.glob("command*.json"):
                if json_file.name in processed_commands:
                    continue

                print(f"\n[SENTRY THREAD] Command file detected: {json_file.name}")
                try:
                    # Wait for file to be fully written (check size is stable)
                    initial_size = json_file.stat().st_size
                    time.sleep(0.1)  # Brief pause to allow writing to complete
                    if json_file.stat().st_size == initial_size and initial_size > 0:
                        command = json.loads(json_file.read_text())
                        # Put the command onto the thread queue for the main loop to process
                        self.command_queue.put(command)
                        processed_commands.add(json_file.name)
                        json_file.unlink() # Consume the file
                        print(f"[SENTRY THREAD] Command processed successfully: {json_file.name}")
                    else:
                        print(f"[SENTRY THREAD] File appears incomplete (size: {json_file.stat().st_size}), will retry...")
                except Exception as e:
                    print(f"[SENTRY THREAD ERROR] Could not process command file {json_file.name}: {e}", file=sys.stderr)
            time.sleep(1) # Check every second

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
                    await self._execute_mechanical_write(command)
                    continue
                elif "git_operations" in command:
                    # This is a Git Task
                    print("[ACTION TRIAGE] Detected Git Task - executing mechanical git operations...")
                    await self._execute_mechanical_git(command)
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
                    await self._execute_mechanical_write(command)
                    continue
                elif "git_operations" in command:
                    # This is a Git Task
                    print("[ACTION TRIAGE] Detected Git Task - executing mechanical git operations...")
                    await self._execute_mechanical_git(command)
                    continue

                try:
                    # Check if this is a development cycle command
                    if command.get("development_cycle", False):
                        await self._start_new_cycle(command, state_file)
                    else:
                        # Regular task execution
                        original_output_path = self.project_root / command['output_artifact_path']
                        task_result = await self.execute_task(command)

                        # V7.0 MANDATE 3: Check task result before proceeding
                        if task_result is False:
                            print("[FAILURE] Task aborted due to consecutive cognitive failures. No AAR will be generated.", flush=True)
                        else:
                            # Generate AAR for regular tasks - only after confirmed success
                            print("[*] Task complete. Initiating After-Action Report synthesis...", flush=True)
                            # Determine the actual log file path
                            if original_output_path.is_dir():
                                log_file_path = original_output_path / "task_log.md"
                            else:
                                log_file_path = original_output_path
                            await self.generate_aar(log_file_path, command.get('config'))

                except Exception as e:
                    print(f"[MAIN LOOP ERROR] Task execution failed: {e}", file=sys.stderr)

    def run(self):
        """Starts the file watcher thread and the main async loop."""
        print("--- Initializing Commandable Council Orchestrator (v9.0 - Doctrine of Sovereign Action) ---")
        watcher_thread = threading.Thread(target=self._watch_for_commands_thread, daemon=True)
        watcher_thread.start()
        print("[+] Sentry thread for command monitoring has been launched.")
        asyncio.run(self.main_loop())

if __name__ == "__main__":
    try:
        orchestrator = Orchestrator()
        orchestrator.run()
    except RuntimeError as e:
        print(f"[ORCHESTRATOR HALTED] {e}", file=sys.stderr)
        sys.exit(1)