# council_orchestrator/orchestrator.py (v2.1 - Briefing Integration Hardened)
import os
import sys
import time
import json
import re
import hashlib
import asyncio
import threading
import shutil
from queue import Queue as ThreadQueue
from pathlib import Path
from google import genai
from dotenv import load_dotenv

# --- Import briefing packet generator ---
from bootstrap_briefing_packet import main as generate_briefing_packet

# --- (PersonaAgent class remains the same) ---
class PersonaAgent:
    def __init__(self, client, persona_file: Path, state_file: Path):
        self.role = self._extract_role_from_filename(persona_file.name)
        self.state_file = state_file
        persona_content = persona_file.read_text(encoding="utf-8")

        self.client = client
        self.chat = client.chats.create(model="gemini-2.5-flash")
        self.messages = []

        # Load history if exists
        history = self._load_history()
        if history:
            self.messages = history
            # Replay history by sending messages
            for msg in history:
                if msg['role'] == 'user':
                    self.chat.send_message(msg['content'])
        else:
            # Initialize with system instruction
            system_msg = f"SYSTEM INSTRUCTION: You are an AI Council member. {persona_content} Operate strictly within this persona. Keep responses concise. If you need a file, request it with [ORCHESTRATOR_REQUEST: READ_FILE(path/to/your/file.md)]."
            self.chat.send_message(system_msg)
            self.messages.append({"role": "system", "content": system_msg})

        print(f"[+] {self.role} agent initialized.")

    def _load_history(self):
        if self.state_file.exists():
            print(f"  - Loading history for {self.role} from {self.state_file.name}")
            return json.loads(self.state_file.read_text())
        return None

    def save_history(self):
        # Save the messages
        self.state_file.write_text(json.dumps(self.messages, indent=2))
        print(f"  - Saved session state for {self.role} to {self.state_file.name}")

    def query(self, message: str):
        self.messages.append({"role": "user", "content": message})
        response = self.chat.send_message(message)
        reply = response.text.strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _extract_role_from_filename(self, f): return f.split('core_essence_')[1].split('_awakening_seed.txt')[0].upper()

class Orchestrator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.command_queue = ThreadQueue()
        self._configure_api()
        client = genai.Client(api_key=self.api_key)

        persona_dir = self.project_root / "dataset_package"
        state_dir = Path(__file__).parent / "session_states"
        state_dir.mkdir(exist_ok=True)

        self.agents = { "COORDINATOR": PersonaAgent(client, persona_dir / "core_essence_coordinator_awakening_seed.txt", state_dir / "coordinator_session.json"), "STRATEGIST": PersonaAgent(client, persona_dir / "core_essence_strategist_awakening_seed.txt", state_dir / "strategist_session.json"), "AUDITOR": PersonaAgent(client, persona_dir / "core_essence_auditor_awakening_seed.txt", state_dir / "auditor_session.json")}
        self.speaker_order = ["COORDINATOR", "STRATEGIST", "AUDITOR"]

    def _configure_api(self):
        load_dotenv(dotenv_path=self.project_root / '.env')
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key: raise ValueError("GEMINI_API_KEY not found.")

    def _verify_briefing_attestation(self, packet: dict) -> bool:
        """Verifies the integrity of the briefing packet using its SHA256 hash."""
        if "attestation_hash" not in packet.get("metadata", {}):
            print("[CRITICAL] Attestation hash missing from briefing packet. REJECTING.")
            return False

        stored_hash = packet["metadata"]["attestation_hash"]

        packet_for_hashing = packet.copy()
        del packet_for_hashing["metadata"]

        canonical_string = json.dumps(packet_for_hashing, sort_keys=True, separators=(',', ':'))
        calculated_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()

        return stored_hash == calculated_hash

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
                    system_msg = (
                        "SYSTEM INSTRUCTION: You are provided with the synchronized briefing packet. "
                        "This contains temporal anchors, prior directives, and the current task context. "
                        "Incorporate this into your reasoning, but do not regurgitate it verbatim.\n\n"
                        f"BRIEFING_PACKET:\n{json.dumps(packet, indent=2)}"
                    )
                    agent.query(system_msg)
                print(f"[+] Briefing packet injected into {len(self.agents)} agents.")
            except Exception as e:
                print(f"[!] Error injecting briefing packet: {e}")
        else:
            print("[!] briefing_packet.json not found — continuing without synchronized packet.")

    def archive_briefing_packet(self):
        """Archive briefing packet after deliberation completes."""
        briefing_path = self.project_root / "WORK_IN_PROGRESS/council_memory_sync/briefing_packet.json"
        if briefing_path.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_dir = self.project_root / f"ARCHIVE/council_memory_sync_{timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(briefing_path), archive_dir / "briefing_packet.json")
            print(f"[+] Briefing packet archived to {archive_dir}")

    async def execute_task(self, command):
        task = command['task_description']
        max_rounds = command.get('config', {}).get('max_rounds', 3)
        output_path = self.project_root / command['output_artifact_path']
        log = [f"# Autonomous Triad Task Log\n## Task: {task}\n\n"]
        last_message = task

        # Inject fresh briefing context
        self.inject_briefing_packet()

        if command.get('input_artifacts'):
            # ... (knowledge injection logic is the same)
            knowledge = ["Initial knowledge provided:\n"]
            for path_str in command['input_artifacts']:
                file_path = self.project_root / path_str
                if file_path.exists(): knowledge.append(f"--- CONTENT OF {path_str} ---\n{file_path.read_text()}\n---\n")
            last_message += "\n" + "".join(knowledge)

        print(f"\n▶️  Executing task: '{task}' for up to {max_rounds} rounds...")

        loop = asyncio.get_event_loop()
        for i in range(max_rounds):
            print(f"--- ROUND {i+1} ---", flush=True)
            log.append(f"### ROUND {i+1}\n\n")
            for role in self.speaker_order:
                agent = self.agents[role]
                print(f"  -> Orchestrator to {agent.role}...", flush=True)
                prompt = f"The current state of the discussion is: '{last_message}'. As the {role}, provide your analysis or next step."

                # Run the synchronous API call in a separate thread to avoid blocking the event loop
                response = await loop.run_in_executor(None, agent.query, prompt)
                print(f"  <- {agent.role} to Orchestrator.", flush=True)

                log.append(f"**{agent.role}:**\n{response}\n\n---\n")
                last_message = response # For simplicity, knowledge requests are not handled in this version yet.

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(log))
        print(f"\n[SUCCESS] Deliberation complete. Artifact saved to {output_path}")

        for agent in self.agents.values():
            agent.save_history()
        print("[SUCCESS] All agent session states have been saved.")

        # Archive the used briefing packet
        self.archive_briefing_packet()

    def _watch_for_commands_thread(self):
        """This function runs in a separate thread and watches for command.json."""
        command_file = Path(__file__).parent / "command.json"

        while True:
            if command_file.exists():
                print(f"\n[SENTRY THREAD] Command file detected!")
                try:
                    command = json.loads(command_file.read_text())
                    # Put the command onto the thread queue for the main loop to process
                    self.command_queue.put(command)
                    command_file.unlink() # Consume the file
                except Exception as e:
                    print(f"[SENTRY THREAD ERROR] Could not process command file: {e}", file=sys.stderr)
            time.sleep(1) # Check every second

    async def main_loop(self):
        """The main async loop that waits for commands from the queue."""
        print("--- Orchestrator Main Loop is active. ---")
        loop = asyncio.get_event_loop()
        while True:
            print("--- Orchestrator Idle. Awaiting command from Sentry... ---")
            command = await loop.run_in_executor(None, self.command_queue.get)
            try:
                await self.execute_task(command)
            except Exception as e:
                print(f"[MAIN LOOP ERROR] Task execution failed: {e}", file=sys.stderr)

    def run(self):
        """Starts the file watcher thread and the main async loop."""
        print("--- Initializing Commandable Council Orchestrator (v2.1) ---")

        # Start the file watcher in a separate, non-blocking thread
        watcher_thread = threading.Thread(target=self._watch_for_commands_thread, daemon=True)
        watcher_thread.start()
        print("[+] Sentry thread for command monitoring has been launched.")

        # Run the main async loop
        asyncio.run(self.main_loop())

if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()