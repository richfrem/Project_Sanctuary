# council_orchestrator/cognitive_engines/ollama_engine.py
import os
import ollama
# --- IMPORT HARDENED ---
try:
    from council_orchestrator.cognitive_engines.base import BaseCognitiveEngine
except ImportError:
    from .base import BaseCognitiveEngine

class OllamaEngine(BaseCognitiveEngine):
    """
    Cognitive engine driver for a sovereign, locally-hosted Ollama model.
    This is the Tier 2 Sovereign Substrate, our unbreakable fallback.
    """
    def __init__(self):
        DEFAULT_MODEL = "qwen2:7b"
        DEFAULT_HOST = "http://localhost:11434"
        self.model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        host = os.getenv("OLLAMA_HOST", DEFAULT_HOST)
        try:
            self.client = ollama.Client(host=host)
            self.check_health()
        except Exception as e:
            print(f"[OLLAMA ENGINE WARNING] Initial connection failed: {e}")
            self.client = None

    def execute_turn(self, messages: list) -> str: # NEW SIGNATURE
        """
        Executes a single conversational turn with the local Ollama model.
        """
        if not self.client:
            return "[OLLAMA ENGINE ERROR] Client not initialized. Cannot execute turn."

        # Configuration from environment variables
        max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))

        # The 'messages' list is now used directly. DO NOT add prompt/history.

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            return response['message']['content']
        except ollama.ResponseError as e:
            print(f"[OLLAMA ENGINE ERROR] API error during turn execution: {e.status_code} - {e.error}")
            return f"[OLLAMA ENGINE ERROR] API error: {e.error}"
        except Exception as e:
            print(f"[OLLAMA ENGINE ERROR] A connection error occurred: {e}")
            return f"[OLLAMA ENGINE ERROR] Connection failed: {e}"

    def check_health(self) -> dict:
        if not self.client: return {"status": "unhealthy", "details": "Client not initialized."}
        try:
            self.client.list()
            return {"status": "healthy", "details": f"Ollama server is responsive at {self.client._client.base_url}. Model: '{self.model}'"}
        except Exception as e: return {"status": "unhealthy", "details": f"Ollama server is not reachable: {e}"}

    def run_functional_test(self) -> dict:
        if self.check_health()["status"] != "healthy":
            return {"passed": False, "details": "Connectivity check failed."}
        try:
            messages = [{"role": "user", "content": "Briefly, in one word, what is the capital of France?"}]
            response = self.execute_turn(messages)
            if "paris" in response.lower():
                return {"passed": True, "details": f"Functional test passed. Response: '{response[:50]}...'"}
            else:
                return {"passed": False, "details": f"Functional test failed. Unexpected response: '{response[:50]}...'"}
        except Exception as e:
            return {"passed": False, "details": f"Exception during functional test: {e}"}