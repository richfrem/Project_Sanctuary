# council_orchestrator/orchestrator/council/agent.py
# Persona agent class for the council orchestrator

import json
from pathlib import Path

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