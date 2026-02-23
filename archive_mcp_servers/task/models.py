#!/usr/bin/env python3
"""
Task Models
=====================================

Purpose:
    Data definitions for the Task MCP server.
    Defines internal schemas, enums, and request models.

Layer: Data (DTOs)

Key Models:
    # Internal / Enums
    - taskstatus (Enum): backlog, todo, in-progress, complete, blocked
    - TaskPriority (Enum): Critical, High, Medium, Low
    - taskschema: Internal task representation
    - FileOperationResult: Result of file I/O operations

    # MCP Requests
    - TaskCreateRequest
    - TaskUpdateRequest
    - TaskUpdateStatusRequest
    - TaskGetRequest
    - TaskListRequest
    - tasksearchRequest
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class taskstatus(str, Enum):
    #----------------------------------------------------------------------
    # taskstatus
    # Purpose: Enum for task status states
    # Values: backlog, todo, in-progress, complete, blocked
    #----------------------------------------------------------------------
    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in-progress"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    #----------------------------------------------------------------------
    # TaskPriority
    # Purpose: Enum for task priority levels
    # Values: Critical, High, Medium, Low
    #----------------------------------------------------------------------
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class taskschema:
    number: int
    title: str
    status: taskstatus
    priority: TaskPriority
    lead: str
    dependencies: Optional[str] = None
    related_documents: Optional[str] = None
    objective: str = ""
    deliverables: List[str] = None
    acceptance_criteria: List[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.deliverables is None:
            self.deliverables = []
        if self.acceptance_criteria is None:
            self.acceptance_criteria = []


@dataclass
class FileOperationResult:
    file_path: str
    content: str
    operation: str  # "created", "updated", "moved"
    task_number: int
    status: str = "success"
    message: str = ""
    
    def to_dict(self):
        return {
            "file_path": self.file_path,
            "content": self.content,
            "operation": self.operation,
            "task_number": self.task_number,
            "status": self.status,
            "message": self.message
        }

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field

class TaskCreateRequest(BaseModel):
    title: str = Field(..., description="Task title")
    objective: str = Field(..., description="Core objective of the task")
    deliverables: List[str] = Field(..., description="List of concrete outputs")
    acceptance_criteria: List[str] = Field(..., description="List of completion conditions")
    priority: str = Field("Medium", description="Priority level (Critical, High, Medium, Low)")
    status: str = Field("backlog", description="Initial status (backlog, todo, in-progress, complete, blocked)")
    lead: str = Field("Unassigned", description="Assigned lead for the task")
    dependencies: Optional[str] = Field(None, description="Task dependencies (e.g., 'Requires #012')")
    related_documents: Optional[str] = Field(None, description="Related files or protocols")
    notes: Optional[str] = Field(None, description="Additional context or notes")
    task_number: Optional[int] = Field(None, description="Specific task number (auto-generated if omitted)")

class TaskUpdateRequest(BaseModel):
    task_number: int = Field(..., description="The task number to update")
    updates: Dict[str, Any] = Field(..., description="Dictionary of fields to update")

class TaskUpdateStatusRequest(BaseModel):
    task_number: int = Field(..., description="The task number to move")
    new_status: str = Field(..., description="New status (backlog, todo, in-progress, complete, blocked)")
    notes: Optional[str] = Field(None, description="Reason for status change")

class TaskGetRequest(BaseModel):
    task_number: int = Field(..., description="Task number to retrieve")

class TaskListRequest(BaseModel):
    status: Optional[str] = Field(None, description="Filter by status")
    priority: Optional[str] = Field(None, description="Filter by priority")

class tasksearchRequest(BaseModel):
    query: str = Field(..., description="Search term or regex pattern")
