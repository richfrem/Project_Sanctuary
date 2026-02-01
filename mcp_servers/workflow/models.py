#!/usr/bin/env python3
"""
Workflow Models
=====================================

Purpose:
    Data definitions for the Workflow MCP server.
    Defines Pydantic models for tool requests.

Layer: Data (DTOs)

Key Models:
    - WorkflowReadRequest: Request to read a workflow file
"""
from pydantic import BaseModel, Field
from typing import Optional

class WorkflowReadRequest(BaseModel):
    filename: str = Field(..., description="The name of the workflow file to read (e.g., 'git_workflow.md')")
