#============================================
# mcp_servers/workflow/models.py
# Purpose: Data models for the Workflow MCP server.
# Role: Data Layer
#============================================
from pydantic import BaseModel, Field
from typing import Optional

class WorkflowReadRequest(BaseModel):
    filename: str = Field(..., description="The name of the workflow file to read (e.g., 'git_workflow.md')")
