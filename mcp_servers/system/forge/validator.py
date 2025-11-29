"""
Forge MCP Validator
Domain: project_sanctuary.system.forge

Validation logic for Forge MCP operations.
"""
from typing import Optional


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ForgeValidator:
    """Validator for Forge MCP operations."""
    
    def __init__(self, project_root: str):
        """Initialize validator.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
    
    def validate_query_sanctuary_model(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str]
    ) -> dict:
        """Validate query_sanctuary_model parameters.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            system_prompt: Optional system prompt
            
        Returns:
            Validated parameters dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt cannot be empty")
        
        if len(prompt) > 10000:
            raise ValidationError("Prompt too long (max 10000 characters)")
        
        # Validate temperature
        if not 0.0 <= temperature <= 2.0:
            raise ValidationError("Temperature must be between 0.0 and 2.0")
        
        # Validate max_tokens
        if not 1 <= max_tokens <= 8192:
            raise ValidationError("max_tokens must be between 1 and 8192")
        
        # Validate system_prompt if provided
        if system_prompt and len(system_prompt) > 5000:
            raise ValidationError("System prompt too long (max 5000 characters)")
        
        return {
            "prompt": prompt.strip(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt.strip() if system_prompt else None
        }
