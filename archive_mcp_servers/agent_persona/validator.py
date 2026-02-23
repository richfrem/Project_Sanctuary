#!/usr/bin/env python3
"""
Agent Persona Validator
=====================================

Purpose:
    Validation logic for Agent Persona Operations.
    Ensures safe role names, valid paths, and persona definitions.

Layer: Validation (Logic)

Key Classes:
    - PersonaValidator: Main safety logic
        - __init__(persona_dir)
        - validate_role_name(role)
        - validate_custom_persona_creation(role, definition)
        - validate_file_path(path)
"""
import re
from pathlib import Path
from typing import List, Optional
from .models import PersonaConstants

class PersonaValidator:
    """
    Class: PersonaValidator
    Purpose: Enforces Logic and Safety Rules for Personas.
    """

    def __init__(self, persona_dir: Path):
        self.persona_dir = persona_dir

    #============================================
    # Method: validate_role_name
    # Purpose: Ensure role name is safe and valid.
    # Args:
    #   role: Role name string
    # Returns: Normalized role name
    # Throws: ValueError if invalid
    #============================================
    def validate_role_name(self, role: str) -> str:
        """Validate and normalize role name."""
        if not role or not role.strip():
            raise ValueError("Role name cannot be empty")
        
        # Normalize: lower case, replace spaces/special chars with underscores
        normalized = role.lower().strip()
        normalized = re.sub(r'[^a-z0-9_]', '_', normalized)
        
        return normalized

    #============================================
    # Method: validate_custom_persona_creation
    # Purpose: Validates inputs for creating a custom persona.
    # Args:
    #   role: Normalized role name
    #   definition: Persona content
    # Throws: ValueError if invalid or exists
    #============================================
    def validate_custom_persona_creation(self, role: str, definition: str) -> None:
        """Validate custom persona creation params."""
        if not definition or len(definition.strip()) < 10:
            raise ValueError("Persona definition is too short")

        target_file = self.persona_dir / f"{role}.txt"
        if target_file.exists():
            raise ValueError(f"Persona '{role}' already exists")

    #============================================
    # Method: validate_file_path
    # Purpose: Ensure external file paths are valid/safe-ish.
    # Args:
    #   path: File path string
    # Returns: Path object
    # Throws: FileNotFoundError/ValueError
    #============================================
    def validate_file_path(self, path: str) -> Path:
        """Validate and return file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not p.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return p
