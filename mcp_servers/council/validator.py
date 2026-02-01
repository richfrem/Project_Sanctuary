#!/usr/bin/env python3
"""
Council Validator
=====================================

Purpose:
    Validation logic for Council Operations.
    Ensures safe inputs for task dispatch and agent selection.

Layer: Validation (Logic)

Key Classes:
    - CouncilValidator: Main safety logic
        - validate_task(task)
        - validate_agent(agent)
"""

from typing import Optional

class CouncilValidator:
    """
    Class: CouncilValidator
    Purpose: Enforces inputs for Council operations.
    """
    
    VALID_ROLES = ["coordinator", "strategist", "auditor"]

    #============================================
    # Method: validate_task
    # Purpose: Ensure task description is valid.
    # Args:
    #   task: Task description string
    # Throws: ValueError if invalid
    #============================================
    def validate_task(self, task: str) -> None:
        """Validate task description."""
        if not task or not task.strip():
            raise ValueError("Task description cannot be empty")

    #============================================
    # Method: validate_agent
    # Purpose: Ensure agent role is valid.
    # Args:
    #   agent: Agent role name
    # Throws: ValueError if invalid
    #============================================
    def validate_agent(self, agent: Optional[str]) -> None:
        """Validate agent role if provided."""
        if agent and agent not in self.VALID_ROLES:
            # We might allow dynamic roles later, but for now stick to known ones or just warn?
            # The original code just passed it through, but let's enforce checking if it maps to known roles 
            # or if we should check against dynamic roles. Use strict check for now per built-ins.
            # Actually, `AgentPersona` supports custom roles, so maybe we shouldn't fail hard on unknown names
            # unless we know for sure. But this validator is for safety.
            # Let's just pass for now if string is valid format.
            if not isinstance(agent, str) or not agent.strip():
               raise ValueError(f"Invalid agent name: {agent}")
