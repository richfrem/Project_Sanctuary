# council_orchestrator/orchestrator.py (v3.6 - Distillation Engine Correctly Integrated)
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
from substrate_monitor import select_engine
# --- END INTEGRATION ---

from bootstrap_briefing_packet import main as generate_briefing_packet

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

    def query(self, message: str):
        self.messages.append({"role": "user", "content": message})
        try:
            # UPDATED POLYMORPHIC CALL - USE NEW UNIFIED INTERFACE
            # All engines now expect the full messages list
            reply = self.engine.execute_turn(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            error_msg = f"[AGENT ERROR] The engine '{type(self.engine).__name__}' failed during query: {e}"
            print(error_msg)
            self.messages.append({"role": "assistant", "content": error_msg})
            return error_msg

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
        config_path = Path(__file__).parent / "engine_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.engine_limits = config.get('engine_limits', {
                    'gemini': 100000,   # Conservative limit for Gemini
                    'openai': 100000,   # Conservative limit for OpenAI
                    'ollama': 8000      # Local model limit
                })
                print(f"[+] Engine limits loaded from config: {self.engine_limits}")
            except Exception as e:
                print(f"[!] Error loading engine config: {e}. Using defaults.")
                self.engine_limits = {
                    'gemini': 100000,   # Conservative limit for Gemini
                    'openai': 100000,   # Conservative limit for OpenAI
                    'ollama': 8000      # Local model limit
                }
        else:
            print("[!] engine_config.json not found. Using default limits.")
            self.engine_limits = {
                'gemini': 100000,   # Conservative limit for Gemini
                'openai': 100000,   # Conservative limit for OpenAI
                'ollama': 8000      # Local model limit
            }

        self.speaker_order = ["COORDINATOR", "STRATEGIST", "AUDITOR"]
        self.agents = {} # Agents will now be initialized per-task

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

    def inject_briefing_packet(self):
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
                    agent.query(system_msg)
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
        """Starts a new development cycle."""
        # Create initial state
        state = {
            "current_stage": "GENERATING_REQUIREMENTS",
            "project_name": command.get("project_name", "unnamed_project"),
            "original_command": command,
            "approved_artifacts": {},
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        state_file.write_text(json.dumps(state, indent=2))

        # Issue internal command to generate requirements
        requirements_command = {
            "task_description": f"Generate detailed requirements document for the project: {command['task_description']}. Include functional requirements, technical constraints, and success criteria.",
            "output_artifact_path": f"WORK_IN_PROGRESS/development_cycles/{state['project_name']}/requirements.md",
            "config": {"max_rounds": 3}
        }

        print(f"[*] Starting new development cycle for '{state['project_name']}'. Generating requirements...", flush=True)
        await self.execute_task(requirements_command)

        # Update state to awaiting approval
        state["current_stage"] = "AWAITING_APPROVAL_REQUIREMENTS"
        state_file.write_text(json.dumps(state, indent=2))
        print(f"[*] Requirements generated. Awaiting Guardian approval.", flush=True)

    async def _advance_cycle(self, state_file):
        """Advances the development cycle to the next stage."""
        state = json.loads(state_file.read_text())

        if state["current_stage"] == "AWAITING_APPROVAL_REQUIREMENTS":
            # Ingest approved requirements into Cortex
            requirements_path = self.project_root / state["approved_artifacts"].get("requirements", "")
            if requirements_path.exists():
                subprocess.run([sys.executable, str(self.project_root / "ingest_new_knowledge.py"), str(requirements_path)], check=True)
                print(f"[*] Approved requirements ingested into Mnemonic Cortex.", flush=True)

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
                subprocess.run([sys.executable, str(self.project_root / "ingest_new_knowledge.py"), str(tech_design_path)], check=True)
                print(f"[*] Approved tech design ingested into Mnemonic Cortex.", flush=True)

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
        limit = self.engine_limits.get(engine_type, 100000)  # Default conservative limit

        token_count = self._get_token_count(text, engine_type)

        if token_count > limit:
            print(f"[ORCHESTRATOR] WARNING: Token count ({token_count:.0f}) exceeds limit for {engine_type} ({limit}).")
            return self._distill_with_local_engine(text, task_description)
        else:
            return text

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
        engine = select_engine(command.get("config"))
        if not engine:
            print(f"[ORCHESTRATOR HALTED] No healthy cognitive substrate could be selected for this task. Config: {command.get('config')}")
            print("[ORCHESTRATOR HALTED] This may indicate a force_engine override failure or all engines are unhealthy.")
            return

        # Initialize agents with the selected engine for this task.
        self._initialize_agents(engine)
        # --- END INTEGRATION ---

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

        # Enhance briefing with context from task description
        try:
            self._enhance_briefing_with_context(task)
        except FileNotFoundError as e:
            print(f"[CRITICAL] Context file error: {e}. Task aborted.")
            return

        # Inject fresh briefing context
        self.inject_briefing_packet()

        if command.get('input_artifacts'):
            # ... (knowledge injection logic is the same)
            knowledge = ["Initial knowledge provided:\n"]
            for path_str in command['input_artifacts']:
                file_path = self.project_root / path_str
                if file_path.exists(): knowledge.append(f"--- CONTENT OF {path_str} ---\n{file_path.read_text()}\n---\n")
            last_message += "\n" + "".join(knowledge)

        # --- RESOURCE SOVEREIGNTY: DISTILL INPUT IF NECESSARY ---
        # Determine engine type for token checking
        engine_type = "ollama"  # Default assumption
        if hasattr(engine, '__class__'):
            engine_name = type(engine).__name__.lower()
            if "openai" in engine_name:
                engine_type = "openai"
            elif "gemini" in engine_name:
                engine_type = "gemini"
            elif "ollama" in engine_name:
                engine_type = "ollama"

        print(f"\n▶️  Executing task: '{task}' for up to {max_rounds} rounds on {type(engine).__name__}")
        print(f"[ORCHESTRATOR] Using engine: {type(engine).__name__} for all agents in this task.")

        loop = asyncio.get_event_loop()
        for i in range(max_rounds):
            print(f"--- ROUND {i+1} ---", flush=True)
            log.append(f"### ROUND {i+1}\n\n")
            for role in self.speaker_order:
                agent = self.agents[role]
                print(f"  -> Orchestrator to {agent.role}...", flush=True)

                # The raw prompt is constructed here, including growing history
                prompt = f"The current state of the discussion is: '{last_message}'. As the {role}, provide your analysis or next step."

                #
                # --- // BEGIN CRITICAL CORRECTION v3.6 // ---
                #
                # The distillation check MUST happen here, on the final, complete payload for the turn.
                prepared_prompt = self._prepare_input_for_engine(prompt, engine_type, task)
                #
                # --- // END CRITICAL CORRECTION v3.6 // ---
                #

                # Run the synchronous API call with the PREPARED and potentially distilled prompt
                response = await loop.run_in_executor(None, agent.query, prepared_prompt)
                print(f"  <- {agent.role} to Orchestrator.", flush=True)

                # Handle knowledge requests
                knowledge_response = self._handle_knowledge_request(response)
                if knowledge_response:
                    # Inject the knowledge response back into the conversation
                    print(f"  -> Orchestrator providing context to {agent.role}...", flush=True)
                    knowledge_injection = await loop.run_in_executor(None, agent.query, knowledge_response)
                    print(f"  <- {agent.role} acknowledging context.", flush=True)
                    response += f"\n\n{knowledge_injection}"
                    log.append(f"**{agent.role}:**\n{response}\n\n---\n")
                    log.append(f"**ORCHESTRATOR (Fulfilled Request):**\n{knowledge_response}\n\n---\n")
                else:
                    log.append(f"**{agent.role}:**\n{response}\n\n---\n")

                last_message = response

                # --- ADD THIS LINE ---
                time.sleep(1) # Add a 1-second pause to be kind to the API
                # ---------------------

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(log))
        print(f"\n[SUCCESS] Deliberation complete. Artifact saved to {output_path}")

        for agent in self.agents.values():
            agent.save_history()
        print("[SUCCESS] All agent session states have been saved.")

        # Archive the used briefing packet
        self.archive_briefing_packet()

    def _watch_for_commands_thread(self):
        """This function runs in a separate thread and watches for command.json files."""
        command_dir = Path(__file__).parent
        processed_commands = set()  # Track processed command files

        while True:
            # Look for any .json files in the orchestrator directory
            for json_file in command_dir.glob("*.json"):
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
                if command.get("action") == "APPROVE_CURRENT_STAGE":
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
                else:
                    print("[!] Invalid command during development cycle. Awaiting APPROVE_CURRENT_STAGE.", flush=True)
            else:
                # We are idle, waiting for a new task to start a new cycle
                print("--- Orchestrator Idle. Awaiting command from Sentry... ---", flush=True)
                command = await loop.run_in_executor(None, self.command_queue.get)
                try:
                    # Check if this is a development cycle command
                    if command.get("development_cycle", False):
                        await self._start_new_cycle(command, state_file)
                    else:
                        # Regular task execution
                        original_output_path = self.project_root / command['output_artifact_path']
                        await self.execute_task(command)

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
        print("--- Initializing Commandable Council Orchestrator (v3.7 - Unified Polymorphic Interface) ---")
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