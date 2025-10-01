# council_orchestrator/orchestrator.py (v2.0 - Multi-threaded)
import os
import sys
import time
import json
import re
import asyncio
import threading
import shutil
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

        # Load history if exists
        history = self._load_history()
        if history:
            # Replay history by sending messages
            for msg in history:
                if msg['role'] == 'user':
                    self.chat.send_message(msg['parts'][0])
        else:
            # Initialize with system instruction
            self.chat.send_message(f"SYSTEM INSTRUCTION: You are an AI Council member. {persona_content} Operate strictly within this persona. Keep responses concise. If you need a file, request it with [ORCHESTRATOR_REQUEST: READ_FILE(path/to/your/file.md)].")

        print(f"[+] {self.role} agent initialized.")

    def _load_history(self):
        if self.state_file.exists():
            print(f"  - Loading history for {self.role} from {self.state_file.name}")
            return json.loads(self.state_file.read_text())
        return None

    def save_history(self):
        # Save the messages
        messages = []
        for msg in self.chat.messages:
            messages.append({'role': msg.role, 'parts': [msg.content]})
        self.state_file.write_text(json.dumps(messages, indent=2))
        print(f"  - Saved session state for {self.role} to {self.state_file.name}")

    def query(self, message: str):
        response = self.chat.send_message(message)
        return response.text.strip()

    def _extract_role_from_filename(self, f): return f.split('core_essence_')[1].split('_awakening_seed.txt')[0].upper()

class Orchestrator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.command_queue = asyncio.Queue()
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

    def inject_briefing_packet(self):
        """Generate + inject briefing packet into all agents."""
        print("[*] Generating fresh briefing packet...")
        generate_briefing_packet()

        briefing_path = self.project_root / "WORK_IN_PROGRESS/council_memory_sync/briefing_packet.json"
        if briefing_path.exists():
            packet = json.loads(briefing_path.read_text(encoding="utf-8"))
            # Inject into initial context for all agents
            for agent in self.agents.values():
                # Assuming PersonaAgent has a way to inject context, e.g., by sending a message
                agent.query(f"INJECT_BRIEFING_CONTEXT: {json.dumps(packet)}")
            print(f"[+] Briefing packet injected into {len(self.agents)} agents.")
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
            log.append(f"### ROUND {i+1}\n\n")
            for role in self.speaker_order:
                agent = self.agents[role]
                prompt = f"The current state of the discussion is: '{last_message}'. As the {role}, provide your analysis or next step."

                # Run the synchronous API call in a separate thread to avoid blocking the event loop
                response = await loop.run_in_executor(None, agent.query, prompt)

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
        loop = asyncio.get_running_loop()

        while True:
            if command_file.exists():
                print(f"\n[SENTRY THREAD] Command file detected!")
                try:
                    command = json.loads(command_file.read_text())
                    # Put the command onto the async queue for the main loop to process
                    asyncio.run_coroutine_threadsafe(self.command_queue.put(command), loop)
                    command_file.unlink() # Consume the file
                except Exception as e:
                    print(f"[SENTRY THREAD ERROR] Could not process command file: {e}", file=sys.stderr)
            time.sleep(1) # Check every second

    async def main_loop(self):
        """The main async loop that waits for commands from the queue."""
        print("--- Orchestrator Main Loop is active. ---")
        while True:
            print("--- Orchestrator Idle. Awaiting command from Sentry... ---")
            command = await self.command_queue.get()
            try:
                await self.execute_task(command)
            except Exception as e:
                print(f"[MAIN LOOP ERROR] Task execution failed: {e}", file=sys.stderr)
            self.command_queue.task_done()

    def run(self):
        """Starts the file watcher thread and the main async loop."""
        print("--- Initializing Commandable Council Orchestrator (v2.0) ---")

        # Start the file watcher in a separate, non-blocking thread
        watcher_thread = threading.Thread(target=self._watch_for_commands_thread, daemon=True)
        watcher_thread.start()
        print("[+] Sentry thread for command monitoring has been launched.")

        # Run the main async loop
        asyncio.run(self.main_loop())

if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()