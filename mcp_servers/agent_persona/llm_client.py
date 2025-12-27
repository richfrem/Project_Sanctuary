#============================================
# mcp_servers/agent_persona/llm_client.py
# Purpose: LLM Client Library for Agent Persona MCP.
#          Provides a standard interface for interacting with LLM providers (Ollama, OpenAI, Google).
# Role: Infrastructure Layer (Adapter)
# Used as: Helper module by agent.py
# LIST OF CLASSES/FUNCTIONS:
#   - LLMClient (Abstract)
#   - OllamaClient
#   - OpenAIClient
#   - get_llm_client_config
#   - get_llm_client
#============================================

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from mcp_servers.lib.env_helper import get_env_variable

logger = logging.getLogger(__name__)

# Protocol 116: Ollama Service Endpoint
# Default to localhost for host-based execution.
# For container network, set OLLAMA_HOST=http://ollama_model_mcp:11434 in docker-compose.
OLLAMA_ENDPOINT = "http://127.0.0.1:11434"

#============================================
# Class: LLMClient
# Purpose: Abstract base class for LLM clients.
#============================================
class LLMClient(ABC):
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    #============================================
    # Method: execute_turn
    # Purpose: Execute a single conversational turn.
    # Args:
    #   messages: List of message dicts (role, content)
    # Returns: The model's response content
    #============================================
    @abstractmethod
    def execute_turn(self, messages: List[Dict[str, str]]) -> str:
        pass
    
    #============================================
    # Method: check_health
    # Purpose: Check if the client is healthy and connected.
    # Returns: Dict with health status
    #============================================
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        pass

#============================================
# Class: OllamaClient
# Purpose: Client for Ollama models (including Sanctuary).
#============================================
class OllamaClient(LLMClient):
    """Client for Ollama models (including Sanctuary)"""
    
    def __init__(self, model_name: str = None, ollama_host: str = None):
        # Default to Sanctuary model if no model specified
        model_name = model_name or get_env_variable("OLLAMA_MODEL", required=False) or "Sanctuary-Qwen2-7B:latest"
        super().__init__(model_name)
        # Protocol 116: Use container network addressing by default for MCP infrastructure
        self.host = ollama_host or get_env_variable("OLLAMA_HOST", required=False) or OLLAMA_ENDPOINT
        
        # Warn if localhost is used (violates Protocol 116 for container networking)
        if "localhost" in self.host and get_env_variable("MCP_ENV", False) == "production":
            logger.warning(
                f"[OllamaClient] Using localhost address ({self.host}). "
                f"For MCP infrastructure, use container network: {OLLAMA_ENDPOINT}"
            )
        
        # Warn if localhost is used (violates Protocol 116 for MCP infrastructure)
        if "localhost" in self.host or "127.0.0.1" in self.host:
            logger.warning(
                f"[OllamaClient] Using localhost address ({self.host}). "
                f"For MCP infrastructure, use container network: {OLLAMA_ENDPOINT} "
                "(Protocol 116: Container Network Isolation)"
            )
        
        self.client = None
        self._initialize_client()

        
    def _initialize_client(self):
        try:
            import ollama
            self.client = ollama.Client(host=self.host)
        except ImportError:
            logger.error("Ollama python package not installed")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self.client = None
            
    def execute_turn(self, messages: List[Dict[str, str]]) -> str:
        if not self.client:
            return "Error: Ollama client not initialized"
            
        try:
            # Get config from env
            max_tokens = int(get_env_variable("OLLAMA_MAX_TOKENS", required=False) or "4096")
            temperature = float(get_env_variable("OLLAMA_TEMPERATURE", required=False) or "0.7")
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama execution failed: {e}")
            raise RuntimeError(f"Ollama execution failed: {e}")

    def check_health(self) -> Dict[str, Any]:
        if not self.client:
            return {"status": "error", "details": "Client not initialized"}
            
        try:
            self.client.list()
            return {"status": "healthy", "details": "Connected to Ollama"}
        except Exception as e:
            return {"status": "unhealthy", "details": str(e)}

#============================================
# Class: OpenAIClient
# Purpose: Client for OpenAI models.
#============================================
class OpenAIClient(LLMClient):
    """Client for OpenAI models"""
    
    def __init__(self, model_name: str = None):
        # Default to env var or hardcoded fallback
        default_model = get_env_variable("CHAT_GPT_QUOTE_AGENT_MODEL", required=False) or "gpt-4-turbo"
        super().__init__(model_name or default_model)
        self.api_key = get_env_variable("OPENAI_API_KEY", required=False)
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found")
            return
            
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("openai python package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def execute_turn(self, messages: List[Dict[str, str]]) -> str:
        if not self.client:
            return "Error: OpenAI client not initialized"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI execution failed: {e}")
            raise RuntimeError(f"OpenAI execution failed: {e}")

    def check_health(self) -> Dict[str, Any]:
        if not self.client:
            return {"status": "error", "details": "Client not initialized (check API key)"}
        return {"status": "healthy", "details": "Client initialized"}

#============================================
# Function: get_llm_client_config
# Purpose: Selects the correct LLM client configuration based on the model preference.
# Args:
#   model_preference: Preferred model type (ollama, gpt, gemini)
# Returns: Configuration dictionary
#============================================
def get_llm_client_config(model_preference: Optional[str] = None) -> Dict[str, Any]:
    preference = (model_preference or "gemini").lower()
    
    if preference == "ollama":
        # Configuration for the self-hosted model
        return {
            "model_type": "ollama",
            "base_url": OLLAMA_ENDPOINT,
            "api_key": None, # Local models often don't need a key
            "model_name": "Sanctuary-Qwen2-7B:latest" # Assuming this is the image name
        }
    
    elif preference == "gpt":
        # Configuration for OpenAI models
        return {
            "model_type": "openai",
            "api_key": get_env_variable("OPENAI_API_KEY", required=False),
            "model_name": "gpt-4o" # Example model
        }
        
    elif preference == "gemini":
        # Default configuration for Google models
        return {
            "model_type": "google",
            "api_key": get_env_variable("GEMINI_API_KEY", required=False),
            "model_name": "gemini-2.5-pro" # Example model
        }
        
    else:
        # Fallback to the primary default if preference is unrecognized
        return get_llm_client_config("gemini")

#============================================
# Function: get_llm_client
# Purpose: Factory function to get an LLM client.
# Args:
#   provider: LLM provider ("ollama", "openai", "gemini")
#   model_name: Specific model to use
#   ollama_host: Ollama host URL (for Protocol 116 compliance)
# Returns: Initialized LLMClient instance
#============================================
def get_llm_client(provider: str = "ollama", model_name: str = None, ollama_host: str = None) -> LLMClient:
    provider = provider.lower() if provider else "ollama"
    
    if provider == "ollama":
        return OllamaClient(model_name=model_name, ollama_host=ollama_host)
    elif provider == "openai":
        return OpenAIClient(model_name=model_name)
    else:
        # Default to Ollama if unknown
        logger.warning(f"Unknown provider '{provider}', defaulting to Ollama")
        return OllamaClient(model_name=model_name, ollama_host=ollama_host)
