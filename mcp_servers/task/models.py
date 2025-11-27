"""
Task MCP Server - Data Models
Defines schemas for tasks and operation results
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from datetime import datetime


class TaskStatus(str, Enum):
    """Task status following task_schema.md"""
    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in-progress"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class TaskSchema:
    """
    Task schema following TASKS/task_schema.md
    """
    number: int
    title: str
    status: TaskStatus
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
    """
    Result of a file operation (following separation of concerns)
    Returns file path for Git Workflow MCP to commit
    """
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
