# run_autonomous_triad.py (v2.2 - Enhanced Logging)
import os
import sys
from pathlib import Path
from google import genai
from dotenv import load_dotenv

class PersonaAgent:
    """An agent representing one of the Triad personas, inoculated with a full Awakening Seed."""
    def __init__(self, client, persona_file: Path):
        self.role = self._extract_role_from_filename(persona_file.name)
        persona_content = persona_file.read_text(encoding="utf-8")

        self.chat = client.chats.create(model="gemini-2.5-flash")
        # Initialize with system instruction
        self.chat.send_message(f"SYSTEM INSTRUCTION: Your entire identity, purpose, and operational parameters are defined by the following text. You are a member of an AI council. You must operate strictly within this persona. Keep all responses concise (2-4 sentences). Do not break character. Here is your full Core Essence:\n\n--- START CORE ESSENCE ---\n{persona_content}\n--- END CORE ESSENCE ---")
        print(f"[+] {self.role} agent initialized from {persona_file.name}.")

    def _extract_role_from_filename(self, filename: str) -> str:
        # Extracts 'coordinator', 'strategist', etc. from the filename
        try:
            return filename.split('core_essence_')[1].split('_awakening_seed.txt')[0].upper()
        except IndexError:
            return "UNKNOWN"

    def query(self, message: str) -> str:
        """Sends a message to the agent and gets a response."""
        # This print statement is now more verbose for clarity
        print(f"\n  -> ORCHESTRATOR to {self.role}:")
        print(f"     '{message}'")

        response = self.chat.send_message(message)
        response_text = response.text.strip()

        # This print statement shows the response immediately
        print(f"\n  <- {self.role} to Orchestrator:")
        print(f"     '{response_text}'")

        return response_text

def run_bounded_symposium():
    """Orchestrates a structured, multi-round conversation with real-time logging."""
    print("--- Autonomous Triad Symposium (3-Round) Engaged ---")
    output_path = Path(__file__).parent / "triad_symposium_log.md"

    try:
        # --- Phase 1: Configuration & Agent Initialization ---
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / '.env')
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY not found.")
        client = genai.Client(api_key=api_key)

        persona_dir = project_root / "dataset_package"
        coordinator_seed = persona_dir / "core_essence_coordinator_awakening_seed.txt"
        strategist_seed = persona_dir / "core_essence_strategist_awakening_seed.txt"
        auditor_seed = persona_dir / "core_essence_auditor_awakening_seed.txt"

        if not all([coordinator_seed.exists(), strategist_seed.exists(), auditor_seed.exists()]):
            raise FileNotFoundError("One or more Awakening Seed files not found in dataset_package/.")

        coordinator = PersonaAgent(client, coordinator_seed)
        strategist = PersonaAgent(client, strategist_seed)
        auditor = PersonaAgent(client, auditor_seed)

        conversation_log = [f"# Autonomous Triad Symposium Log\n# Timestamp: {__import__('datetime').datetime.now().isoformat()}\n\n"]

        # --- Phase 2: The Orchestration Loop (3 Rounds) ---
        print("\n▶️  Starting bounded orchestration loop...")

        last_message = "The objective is to harden our defenses against 'Mnemonic Psychosis'. Propose a high-level approach."

        # ** ENHANCED LOGGING **
        print("\n" + "="*25 + " INITIAL TOPIC " + "="*25)
        print(last_message)
        print("="*67)

        speaker_order = [coordinator, strategist, auditor]

        for i in range(3):
            round_num = i + 1
            print(f"\n" + "#"*29 + f" ROUND {round_num} " + "#"*29)
            log_entry = f"## ROUND {round_num}\n\n"
            conversation_log.append(log_entry)

            for agent in speaker_order:
                prompt = f"The current topic of discussion is: '{last_message}'. As the {agent.role}, provide your analysis or next step."
                response = agent.query(prompt)

                log_entry = f"**{agent.role}:**\n{response}\n\n---\n"
                conversation_log.append(log_entry)

                last_message = response

        # --- Phase 3: Capture Artifact ---
        print(f"\n" + "="*25 + " FINALIZING LOG " + "="*25)
        print(f"Capturing full conversation to {output_path}...")
        output_path.write_text("".join(conversation_log), encoding="utf-8")
        print("[SUCCESS] Artifact captured.")

    except Exception as e:
        print(f"\\n[FAILURE] Symposium failed: {e}", file=sys.stderr)
        return 1
    finally:
        print("\n--- Symposium Complete ---")
    return 0

if __name__ == "__main__":
    sys.exit(run_bounded_symposium())