# council_orchestrator/cognitive_engines/gemini_engine.py
import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
# --- IMPORT HARDENED ---
try:
    from council_orchestrator.cognitive_engines.base import BaseCognitiveEngine
except ImportError:
    from .base import BaseCognitiveEngine

class GeminiEngine(BaseCognitiveEngine):
    """
    Cognitive engine driver for the Google Gemini API.
    This is a Tier 1 Performance Substrate.
    """
    def __init__(self):
        DEFAULT_MODEL = "gemini-2.5-flash"
        self.model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.model = None
            return
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def execute_turn(self, messages: list) -> str:
        """
        Executes a single conversational turn with the Gemini model.
        Includes error handling for common API failures like quota and model not found.
        """
        if not self.model:
            return "[GEMINI ENGINE ERROR] Model not initialized due to missing API key."

        # Configuration from environment variables
        max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

        # Extract the current prompt and history from messages
        # messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        current_message = messages[-1]  # Last message is the current prompt
        history_messages = messages[:-1]  # All previous messages are history

        # The Gemini API uses a different history format, so we adapt.
        # This is a key function of the abstraction layer.
        chat = self.model.start_chat(history=[
            {'role': h['role'], 'parts': [h['content']]} for h in history_messages
        ])

        try:
            response = chat.send_message(current_message['content'], generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            ))
            return response.text
        except google_exceptions.ResourceExhausted as e:
            error_msg = f"[GEMINI ENGINE ERROR] Resource exhausted (quota limit). Details: {e}"
            print(error_msg)
            return error_msg
        except google_exceptions.NotFound as e:
            error_msg = f"[GEMINI ENGINE ERROR] Model not found. The specified model '{self.model_name}' may be incorrect or unavailable. Details: {e}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"[GEMINI ENGINE ERROR] An unexpected API error occurred: {e}"
            print(error_msg)
            return error_msg

    def check_health(self) -> dict:
        if not self.model: return {"status": "unhealthy", "details": "GEMINI_API_KEY not configured."}
        try:
            genai.list_models()
            return {"status": "healthy", "details": f"Gemini API is responsive. Model: '{self.model_name}'"}
        except Exception as e: return {"status": "unhealthy", "details": f"Gemini API is not reachable: {e}"}

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