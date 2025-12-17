"""
Sanctuary Allowlist Plugin for IBM ContextForge / Sanctuary Gateway.

This plugin enforces Protocol 101 (Tool Allowlisting) by validating tool calls
against a defined policy.

Usage:
    Copy this file to: sanctuary-gateway/plugins/sanctuary_allowlist.py
"""
import logging
from typing import Dict, Any, Optional

# IBM ContextForge Plugin Interface (Simulated based on context)
class BasePlugin:
    async def initialize(self, config: Dict[str, Any]):
        pass
    
    async def validate_tool_call(self, tool_name: str, args: Dict[str, Any]) -> bool:
        return True

logger = logging.getLogger(__name__)

class SanctuaryAllowlistPlugin(BasePlugin):
    """Enforces Project Sanctuary tool allowlist policies."""
    
    def __init__(self):
        self.allowed_tools = set()
        self.policy_mode = "monitor" # monitor or enforce

    async def initialize(self, config: Dict[str, Any]):
        """Load allowlist configuration."""
        self.policy_mode = config.get("policy_mode", "monitor")
        self.allowed_tools = set(config.get("allowed_tools", []))
        logger.info(f"SanctuaryAllowlist initialized in {self.policy_mode} mode with {len(self.allowed_tools)} allowed tools.")

    async def validate_tool_call(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """
        Validate if a tool is allowed to be executed.
        Returns True if allowed, False otherwise.
        """
        if tool_name in self.allowed_tools:
            return True
        
        msg = f"Tool '{tool_name}' is not in the allowed list."
        
        if self.policy_mode == "enforce":
            logger.warning(f"BLOCKED: {msg}")
            return False # Block execution
        else:
            logger.info(f"AUDIT (ALLOWED): {msg}")
            return True # Allow but log
