
"""
LLM Client Library for Agent Persona MCP

Provides a standard interface for interacting with LLM providers (Ollama, OpenAI, etc.).
Replaces the legacy 'Substrate' and 'Cognitive Engine' terminology.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    @abstractmethod
    def execute_turn(self, messages: List[Dict[str, str]]) -> str:
        """
        Execute a single conversational turn.
        
        Args:
            messages: List of message dicts (role, content)
            
        Returns:
            The model's response content
        """
        pass
    
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """Check if the client is healthy and connected"""
        pass

class OllamaClient(LLMClient):
    """Client for Ollama models (including Sanctuary)"""
    
    def __init__(self, model_name: str = None):
        # Default to env var or hardcoded fallback
        default_model = os.getenv("OLLAMA_MODEL", "Sanctuary-Qwen2-7B:latest")
        super().__init__(model_name or default_model)
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
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
            max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
            temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
            
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

class OpenAIClient(LLMClient):
    """Client for OpenAI models"""
    
    def __init__(self, model_name: str = None):
        # Default to env var or hardcoded fallback
        default_model = os.getenv("CHAT_GPT_QUOTE_AGENT_MODEL", "gpt-4-turbo")
        super().__init__(model_name or default_model)
        self.api_key = os.getenv("OPENAI_API_KEY")
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

def get_llm_client(provider: str = "ollama", model_name: str = None) -> LLMClient:
    """Factory function to get an LLM client"""
    provider = provider.lower() if provider else "ollama"
    
    if provider == "ollama":
        return OllamaClient(model_name=model_name)
    elif provider == "openai":
        return OpenAIClient(model_name=model_name)
    else:
        # Default to Ollama if unknown
        logger.warning(f"Unknown provider '{provider}', defaulting to Ollama")
        return OllamaClient(model_name=model_name)
