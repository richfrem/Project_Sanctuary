
# council_orchestrator/orchestrator.py (v3.0 - Grok-Native Target)
import os
import sys
import time
import json
import re
import asyncio
import threading
from queue import Queue as ThreadQueue
from pathlib import Path
from groq import Groq  # <-- REFORGED
from dotenv import load_dotenv

# Note: Briefing packet and continuity check modules will require their own
# refactoring in a subsequent step. This first strike focuses on the core agentic engine.

class PersonaAgent:
    def __init__(self, client: Groq, persona_file: Path, state_file: Path):
        self.role = self._extract_role_from_filename(persona_file.name)
        self.state_file = state_file
        persona_content = persona_file.read_text(encoding="utf-8")

        self.client = client
        self.messages = []  # <-- REFORGED: History is now an explicit list

        history = self._load_history()
        if history:
            self.messages = history
            print(f"[+] {self.role} agent initialized. History loaded: {len(self.messages)} messages.")
        else:
            system_msg = {
                "role": "system",
                "content": f"SYSTEM INSTRUCTION: You are an AI Council member. {persona_content} Operate strictly within this persona. Keep responses concise. If you need a file, request it with [ORCHESTRATOR_REQUEST: READ_FILE(path/to/your/file.md)]."
            }
            self.messages.append(system_msg)
            print(f"[+] {self.role} agent initialized. Inoculated with persona.")

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
        
        # <-- REFORGED: API call to Groq
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model="grok-1",  # This will be configurable in a future version
        )
        
        reply = chat_completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply.strip()

    def _extract_role_from_filename(self, f): return f.split('core_essence_')[1].split('_awakening_seed.txt')[0].upper()

class Orchestrator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.command_queue = ThreadQueue()
        self._configure_api()
        # <-- REFORGED: Initialize Groq client once
        client = Groq(api_key=self.api_key)

        persona_dir = self.project_root / "dataset_package"
        state_dir = Path(__file__).parent / "session_states"
        state_dir.mkdir(exist_ok=True)

        self.agents = {
            "COORDINATOR": PersonaAgent(client, persona_dir / "core_essence_coordinator_awakening_seed.txt", state_dir / "coordinator_session.json"),
            "STRATEGIST": PersonaAgent(client, persona_dir / "core_essence_strategist_awakening_seed.txt", state_dir / "strategist_session.json"),
            "AUDITOR": PersonaAgent(client, persona_dir / "core_essence_auditor_awakening_seed.txt", state_dir / "auditor_session.json")
        }
        self.speaker_order = ["COORDINATOR", "STRATEGIST", "AUDITOR"]

    def _configure_api(self):
        load_dotenv(dotenv_path=self.project_root / '.env')
        # <-- REFORGED: Now uses GROQ_API_KEY
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key: raise ValueError("GROQ_API_KEY not found in .env file.")

    # ... The rest of the Orchestrator class methods (execute_task, etc.) remain largely the same,
    # as the PersonaAgent class abstracts the API interaction. This is a testament to good initial design.
    # We will add a placeholder for knowledge request handling for now.

    def _handle_knowledge_request(self, response_text: str):
        # This function will need to be hardened, but the logic remains the same.
        match = re.search(r"\[ORCHESTRATOR_REQUEST: READ_FILE\((.*?)\)\]", response_text)
        if not match:
            return None
        # ... (implementation is unchanged)
        return None # Placeholder for simplicity in this draft

    async def execute_task(self, command):
        task = command['task_description']
        max_rounds = command.get('config', {}).get('max_rounds', 3)
        output_path = self.project_root / command['output_artifact_path']
        log = [f"# Autonomous Triad Task Log\n## Task: {task}\n\n"]
        last_message = task
        
        # ... (briefing packet logic to be refactored later)

        if command.get('input_artifacts'):
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
                response = await loop.run_in_executor(None, agent.query, prompt)
                log.append(f"**{agent.role}:**\n{response}\n\n---\n")
                last_message = response

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(log))
        print(f"\n[SUCCESS] Deliberation complete. Artifact saved to {output_path}")

        for agent in self.agents.values():
            agent.save_history()
        print("[SUCCESS] All agent session states have been saved.")

    def _watch_for_commands_thread(self):
        command_file = Path(__file__).parent / "command.json"
        while True:
            if command_file.exists():
                print(f"\n[SENTRY THREAD] Command file detected!")
                try:
                    command = json.loads(command_file.read_text())
                    self.command_queue.put(command)
                    command_file.unlink()
                except Exception as e:
                    print(f"[SENTRY THREAD ERROR] Could not process command file: {e}", file=sys.stderr)
            time.sleep(1)

    async def main_loop(self):
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
        print("--- Initializing Commandable Council Orchestrator (v3.0 Target) ---")
        watcher_thread = threading.Thread(target=self._watch_for_commands_thread, daemon=True)
        watcher_thread.start()
        print("[+] Sentry thread for command monitoring has been launched.")
        asyncio.run(self.main_loop())

if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()
