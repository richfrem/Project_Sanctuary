# council_orchestrator/config.py
# Configuration constants for the orchestrator
import os

# Load engine limits from environment variables with defaults
DEFAULT_ENGINE_LIMITS = {
    'gemini': int(os.getenv('GEMINI_PER_REQUEST_LIMIT', '200000')),
    'openai': int(os.getenv('OPENAI_PER_REQUEST_LIMIT', '100000')),
    'ollama': int(os.getenv('OLLAMA_PER_REQUEST_LIMIT', '8000'))
}

# Load TPM limits from environment variables with defaults
DEFAULT_TPM_LIMITS = {
    'gemini': int(os.getenv('GEMINI_TPM_LIMIT', '250000')),
    'openai': int(os.getenv('OPENAI_TPM_LIMIT', '120000')),
    'ollama': int(os.getenv('OLLAMA_TPM_LIMIT', '999999'))
}

# Council agent roles and speaking order
SPEAKER_ORDER = ["COORDINATOR", "STRATEGIST", "AUDITOR"]

# Agent role constants
COORDINATOR = "COORDINATOR"
STRATEGIST = "STRATEGIST"
AUDITOR = "AUDITOR"