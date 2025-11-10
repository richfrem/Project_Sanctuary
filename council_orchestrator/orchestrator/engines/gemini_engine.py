# council_orchestrator/cognitive_engines/gemini_engine.py
import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
# --- IMPORT HARDENED ---
try:
    from council_orchestrator.orchestrator.engines.base import BaseCognitiveEngine
except ImportError:
    from .base import BaseCognitiveEngine

class GeminiEngine(BaseCognitiveEngine):
    """
    Cognitive engine driver for the Google Gemini API.
    This is a Tier 1 Performance Substrate.
    Compatible with v9.0: Doctrine of Sovereign Action (orchestrator-level changes only).
    """
    def __init__(self, model_name: str = None):
        DEFAULT_MODEL = "gemini-2.5-flash"
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.model = None
            return
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def execute_turn(self, messages: list) -> str: # NEW SIGNATURE
        """
        Executes a single conversational turn with the Gemini model.
        Includes error handling for common API failures like quota and model not found.
        """
        if not self.model:
            return "[GEMINI ENGINE ERROR] Model not initialized due to missing API key."

        # Configuration from environment variables
        max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

        # V8.0: Doctrine of the Native Tongue - Perfect Gemini API translator
        # Process messages to create valid Gemini conversation structure
        processed_history = []
        system_prompt = None

        # First, extract the system prompt and any initial user/model history
        for msg in messages[:-1]:  # Process all but the last message
            role = msg['role']
            content = msg['content']
            if role == 'system':
                system_prompt = content
                continue  # Don't add system prompts to history directly

            # Translate roles for Gemini
            if role == 'assistant':
                gemini_role = 'model'
            else:  # 'user'
                gemini_role = 'user'

            # Ensure alternating roles (user, model, user, model...)
            if processed_history and processed_history[-1]['role'] == gemini_role:
                # If we have consecutive same roles, merge them
                processed_history[-1]['parts'][0] += f"\n\n--- (System Note: Merged Content) ---\n\n{content}"
            else:
                processed_history.append({'role': gemini_role, 'parts': [content]})

        # Start the chat with the processed history
        chat = self.model.start_chat(history=processed_history)

        # Prepare the final message to send
        last_message = messages[-1]
        final_content = last_message['content']

        # Prepend the system prompt to the final user message if it exists
        if system_prompt:
            final_content = f"SYSTEM PROMPT: {system_prompt}\n\n--- (User Request) ---\n\n{final_content}"

        try:
            # Send the final, consolidated message
            response = chat.send_message(final_content, generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            ))
            return response.text
        except google_exceptions.ResourceExhausted as e:
            # Gemini's ResourceExhausted can be quota (TPM/RPM) or other resource limits
            error_details = str(e).lower()
            is_quota_limit = "quota" in error_details or "rate" in error_details
            
            if is_quota_limit:
                error_msg = f"[GEMINI ENGINE ERROR] Rate limit/quota exhausted (likely TPM or RPM). Details: {e}"
                print(error_msg)
                print(f"[GEMINI ENGINE NOTE] Quota limit hit despite orchestrator pacing. This may indicate concurrent usage or config mismatch.")
                print(f"[GEMINI ENGINE RECOMMENDATION] Check TPM limits in engine_config.json match your Gemini tier.")
            else:
                error_msg = f"[GEMINI ENGINE ERROR] Resource exhausted. Details: {e}"
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