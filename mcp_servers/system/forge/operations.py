"""
Forge MCP Operations
Domain: project_sanctuary.system.forge

Core operations for interacting with the fine-tuned Sanctuary model.
"""
import os
from typing import Optional, List, Dict, Any
from .models import ModelQueryResponse


class ForgeOperations:
    """Operations for Forge MCP server."""
    
    def __init__(self, project_root: str):
        """Initialize Forge operations.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.sanctuary_model = "hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M"
    
    def query_sanctuary_model(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> ModelQueryResponse:
        """Query the fine-tuned Sanctuary model via Ollama.
        
        Args:
            prompt: The user prompt/question
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt for context
            
        Returns:
            ModelQueryResponse with the model's answer
        """
        try:
            import ollama
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Query Ollama
            response = ollama.chat(
                model=self.sanctuary_model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            # Extract response
            answer = response['message']['content']
            
            # Get token counts if available
            prompt_tokens = response.get('prompt_eval_count')
            completion_tokens = response.get('eval_count')
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens and completion_tokens else None
            
            return ModelQueryResponse(
                model=self.sanctuary_model,
                response=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                temperature=temperature,
                status="success"
            )
            
        except ImportError:
            return ModelQueryResponse(
                model=self.sanctuary_model,
                response="",
                status="error",
                error="ollama package not installed. Install with: pip install ollama"
            )
        except Exception as e:
            return ModelQueryResponse(
                model=self.sanctuary_model,
                response="",
                status="error",
                error=f"Failed to query model: {str(e)}"
            )
    
    def check_model_availability(self) -> Dict[str, Any]:
        """Check if the Sanctuary model is available in Ollama.
        
        Returns:
            Dictionary with availability status
        """
        try:
            import ollama
            
            # List available models
            models_response = ollama.list()
            
            # Extract model names - handle different response formats
            if isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            else:
                models_list = models_response
            
            model_names = [m.get('name', m.get('model', str(m))) if isinstance(m, dict) else str(m) for m in models_list]
            
            # Check if our model is available
            is_available = any(self.sanctuary_model in name for name in model_names)
            
            return {
                "status": "success",
                "model": self.sanctuary_model,
                "available": is_available,
                "all_models": model_names
            }
            
        except ImportError:
            return {
                "status": "error",
                "error": "ollama package not installed"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
