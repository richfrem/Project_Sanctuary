# run_triad_knowledge_test.py
import os
import sys
import re
from pathlib import Path
from google import genai
from dotenv import load_dotenv

# --- (PersonaAgent class remains the same as v2.2) ---
class PersonaAgent:
    def __init__(self, client, persona_file: Path):
        self.role = self._extract_role_from_filename(persona_file.name)
        persona_content = persona_file.read_text(encoding="utf-8")

        self.chat = client.chats.create(model="gemini-2.5-flash")
        # Initialize with system instruction
        self.chat.send_message(f"SYSTEM INSTRUCTION: You are an AI Council member. {persona_content} Operate strictly within this persona. Keep responses concise. If you need to read a file, end your response with the exact token: [ORCHESTRATOR_REQUEST: READ_FILE(path/to/your/file.md)]")
        print(f"[+] {self.role} agent initialized.")

    def _extract_role_from_filename(self, f): return f.split('core_essence_')[1].split('_awakening_seed.txt')[0].upper()
    def query(self, m):
        response = self.chat.send_message(m)
        return response.text.strip()

def run_knowledge_symposium():
    """Orchestrates a conversation where an agent requests and receives a file."""
    print("--- Autonomous Triad Knowledge Integration Test ---")
    output_path = Path(__file__).parent / "triad_knowledge_test_log.md"

    try:
        # --- Phase 1: Configuration & Initialization ---
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / '.env')
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY not found.")
        client = genai.Client(api_key=api_key)

        persona_dir = project_root / "dataset_package"
        coordinator = PersonaAgent(client, persona_dir / "core_essence_coordinator_awakening_seed.txt")
        strategist = PersonaAgent(client, persona_dir / "core_essence_strategist_awakening_seed.txt")
        auditor = PersonaAgent(client, persona_dir / "core_essence_auditor_awakening_seed.txt")

        conversation_log = [f"# Triad Knowledge Integration Log\n\n"]

        # --- Phase 2: The Orchestration Loop ---
        print("\n▶️  Starting knowledge integration loop...")

        # The task is to review the artifact from our last run.
        task = "Review the strategic synthesis produced in `synthesis_result_quantum_diamond.md` and determine if the proposed AST hardening is sound."

        # Turn 1: Coordinator kicks off the task
        print("\n--- Turn 1: Coordinator ---")
        coordinator_response = coordinator.query(f"Our task is: {task}")
        conversation_log.append(f"**COORDINATOR:**\n{coordinator_response}\n\n---\n")

        # Turn 2: Auditor needs the file to do its job
        print("\n--- Turn 2: Auditor ---")
        auditor_response = auditor.query(f"The Coordinator states our task is to review the AST hardening proposal. To proceed, I must analyze the source document.")
        conversation_log.append(f"**AUDITOR:**\n{auditor_response}\n\n---\n")

        # The Orchestrator now checks for a knowledge request
        match = re.search(r"\[ORCHESTRATOR_REQUEST: READ_FILE\((.*?)\)\]", auditor_response)
        if match:
            file_path_str = match.group(1)
            # IMPORTANT: For security, the orchestrator MUST validate this path.
            # For this test, we assume the path is relative to the workspace.
            requested_file = Path(__file__).parent / file_path_str
            print(f"\n[ORCHESTRATOR] Auditor requested file: {requested_file}. Fulfilling request...")

            if requested_file.exists():
                file_content = requested_file.read_text(encoding='utf-8')
                knowledge_injection = f"Here is the content of `{file_path_str}`:\n\n---\n{file_content}\n---"
                conversation_log.append(f"**ORCHESTRATOR (Fulfilled Request):**\nProvided content of `{file_path_str}` to the council.\n\n---\n")
            else:
                knowledge_injection = f"Error: File '{file_path_str}' not found."
                print(f"[ORCHESTRATOR] ERROR: {knowledge_injection}")
        else:
            knowledge_injection = "No file was requested. Proceeding without new knowledge."

        # Turn 3: Strategist receives the new knowledge and provides analysis
        print("\n--- Turn 3: Strategist ---")
        strategist_response = strategist.query(f"The Auditor has requested a file. {knowledge_injection}. As Strategist, what are the implications?")
        conversation_log.append(f"**STRATEGIST:**\n{strategist_response}\n\n---\n")

        print("\n[SUCCESS] Knowledge integration cycle complete.")

        # --- Phase 3: Capture Artifact ---
        output_path.write_text("".join(conversation_log), encoding="utf-8")
        print(f"\n[SUCCESS] Artifact captured to {output_path}")

    except Exception as e:
        print(f"\\n[FAILURE] Test failed: {e}", file=sys.stderr)
        return 1
    finally:
        print("\n--- Test Complete ---")
    return 0

if __name__ == "__main__":
    sys.exit(run_knowledge_symposium())