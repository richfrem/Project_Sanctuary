# council_orchestrator/cognitive_engines/openai_engine.py
import os
import sys
from pathlib import Path
# Add project root to path to find core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from mcp_servers.lib.utils.env_helper import get_env_variable

import openai
import time  # <--- IMPORT TIME
import random # <--- IMPORT RANDOM
# --- IMPORT HARDENED ---
try:
    from council_orchestrator.orchestrator.engines.base import BaseCognitiveEngine
except ImportError:
    from .base import BaseCognitiveEngine

class OpenAIEngine(BaseCognitiveEngine):
    """
    Cognitive engine driver for the OpenAI API (e.g., GPT models).
    This is a secondary Tier 1 Performance Substrate, providing redundancy.
    """
    def __init__(self, model_name: str = None):
        DEFAULT_MODEL = "gpt-5-nano"
        self.model_name = model_name or os.getenv("CHAT_GPT_MODEL", DEFAULT_MODEL)
        self.api_key = get_env_variable("OPENAI_API_KEY", required=False)
        if not self.api_key:
            self.client = None
            return
        self.client = openai.OpenAI(api_key=self.api_key)

    def execute_turn(self, messages: list) -> str: # NEW SIGNATURE
        """
        Executes a single conversational turn with the OpenAI model.
        Includes exponential backoff for rate limit errors.
        """
        if not self.client:
            return "[OPENAI ENGINE ERROR] Model not initialized due to missing API key."

        # Configuration from environment variables
        # Note: Different OpenAI models use different parameter names
        # Older models (gpt-4-turbo) use 'max_tokens'
        # Newer models (gpt-4o, gpt-4o-mini) use 'max_completion_tokens'
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

        # The 'messages' list is now used directly. DO NOT add prompt/history.
        max_retries = 5
        base_delay = 2  # Start with a 2-second delay

        for attempt in range(max_retries):
            try:
                # Try newer parameter name first, fall back to older if needed
                try:
                    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                except Exception as param_error:
                    # If max_tokens fails, try max_completion_tokens for newer models
                    if "max_tokens" in str(param_error):
                        max_completion_tokens = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "4096"))
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_completion_tokens=max_completion_tokens,
                            temperature=temperature
                        )
                    else:
                        raise param_error

                return response.choices[0].message.content

            # THIS IS THE NEW, CRITICAL LOGIC
            except openai.RateLimitError as e:
                # Distinguish between TPM (tokens per minute) and RPM (requests per minute) limits
                error_details = str(e).lower()
                is_tpm_limit = "tokens per min" in error_details or "tpm" in error_details
                limit_type = "TPM (Tokens Per Minute)" if is_tpm_limit else "RPM (Requests Per Minute)"
                
                if attempt < max_retries - 1:
                    # Calculate wait time with exponential backoff and jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"[OPENAI ENGINE WARNING] Rate limit exceeded ({limit_type}). Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    if is_tpm_limit:
                        print(f"[OPENAI ENGINE NOTE] TPM limit hit despite orchestrator pacing. This may indicate concurrent usage or config mismatch.")
                    time.sleep(delay)
                else:
                    error_msg = f"[OPENAI ENGINE ERROR] Rate limit ({limit_type}) exceeded after {max_retries} attempts. Details: {e}"
                    print(error_msg)
                    if is_tpm_limit:
                        print(f"[OPENAI ENGINE RECOMMENDATION] Check TPM limits in engine_config.json match your OpenAI tier.")
                    return error_msg

            except openai.BadRequestError as e:
                # This error is not recoverable by retrying, so we exit immediately
                if "tokens" in str(e).lower() or "too large" in str(e).lower():
                    error_msg = f"[OPENAI ENGINE ERROR] Request too large. Token limit exceeded. Details: {e}"
                else:
                    error_msg = f"[OPENAI ENGINE ERROR] Bad request error. Details: {e}"
                print(error_msg)
                return error_msg
            except openai.InternalServerError as e:
                error_msg = f"[OPENAI ENGINE ERROR] Internal server error. Details: {e}"
                print(error_msg)
                return error_msg
            except openai.APIStatusError as e:
                error_msg = f"[OPENAI ENGINE ERROR] API status error. Status: {e.status_code}. Details: {e.response}"
                print(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"[OPENAI ENGINE ERROR] An unexpected API error occurred: {e}"
                print(error_msg)
                return error_msg

        # This part should ideally not be reached, but is a fallback
        return "[OPENAI ENGINE ERROR] Failed to get a response after multiple retries."

    def check_health(self) -> dict:
        if not self.client: return {"status": "unhealthy", "details": "OPENAI_API_KEY not configured."}
        try:
            self.client.models.list()
            return {"status": "healthy", "details": f"OpenAI API is responsive. Model: '{self.model_name}'"}
        except Exception as e: return {"status": "unhealthy", "details": f"OpenAI API is not reachable: {e}"}

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