#============================================
# mcp_servers/task/operations.py
# Purpose: Core business logic for Task MCP.
#          Handles task file CRUD operations (create, update, status change).
# Role: Business Logic Layer
# Used as: Helper module by server.py
# LIST OF CLASSES:
#   - TaskOperations
#============================================

"""
Task MCP Server - File Operations
Handles all task file operations (create, update, move, read, list, search)
Following separation of concerns: File operations only, no Git commits
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os

from mcp_servers.lib.logging_utils import setup_mcp_logging

logger = setup_mcp_logging(__name__)

from .models import TaskSchema, TaskStatus, TaskPriority, FileOperationResult
from .validator import TaskValidator


class TaskOperations:

    #----------------------------------------------------------------------
    # __init__
    # Purpose: Initialize TaskOperations with project root and validator.
    # Args:
    #   project_root: Root directory of the project
    #----------------------------------------------------------------------
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.tasks_dir = self.project_root / "TASKS"
        self.validator = TaskValidator(self.project_root)
        
        # Status to directory mapping
        self.status_dirs = {
            TaskStatus.BACKLOG: self.tasks_dir / "backlog",
            TaskStatus.TODO: self.tasks_dir / "todo",
            TaskStatus.IN_PROGRESS: self.tasks_dir / "in-progress",
            TaskStatus.COMPLETE: self.tasks_dir / "done",
            TaskStatus.BLOCKED: self.tasks_dir / "in-progress"  # Blocked tasks stay in in-progress
        }
    
    #----------------------------------------------------------------------
    # create_task
    # Purpose: Create a new task file
    # Args:
    #   title: Task title
    #   objective: What and why
    #   deliverables: List of concrete outputs
    #   acceptance_criteria: List of completion conditions
    #   priority: Task priority (default: MEDIUM)
    #   status: Initial status (default: BACKLOG)
    #   lead: Assigned person/agent (default: Unassigned)
    #   dependencies: Task dependencies (e.g., 'Requires #012')
    #   related_documents: Related files/protocols
    #   notes: Additional context
    #   task_number: Specific task number (auto-generated if None)
    # Returns: FileOperationResult with file path and content
    #----------------------------------------------------------------------
    def create_task(
        self,
        title: str,
        objective: str,
        deliverables: List[str],
        acceptance_criteria: List[str],
        priority: TaskPriority = TaskPriority.MEDIUM,
        status: TaskStatus = TaskStatus.BACKLOG,
        lead: str = "Unassigned",
        dependencies: Optional[str] = None,
        related_documents: Optional[str] = None,
        notes: Optional[str] = None,
        task_number: Optional[int] = None
    ) -> FileOperationResult:
        # Get next task number if not provided
        if task_number is None:
            task_number = self.validator.get_next_task_number()
        
        # Validate task number is unique
        is_valid, error_msg = self.validator.validate_task_number(task_number)
        if not is_valid:
            return FileOperationResult(
                file_path="",
                content="",
                operation="create",
                task_number=task_number,
                status="error",
                message=error_msg
            )
        
        # Create task schema
        task = TaskSchema(
            number=task_number,
            title=title,
            status=status,
            priority=priority,
            lead=lead,
            dependencies=dependencies or "None",
            related_documents=related_documents or "None",
            objective=objective,
            deliverables=deliverables,
            acceptance_criteria=acceptance_criteria,
            notes=notes
        )
        
        # Validate schema
        is_valid, errors = self.validator.validate_task_schema(task)
        if not is_valid:
            return FileOperationResult(
                file_path="",
                content="",
                operation="create",
                task_number=task_number,
                status="error",
                message=f"Schema validation failed: {', '.join(errors)}"
            )
        
        # Validate dependencies
        if dependencies and dependencies.lower() != "none":
            is_valid, error_msg = self.validator.validate_dependencies(dependencies)
            if not is_valid:
                return FileOperationResult(
                    file_path="",
                    content="",
                    operation="create",
                    task_number=task_number,
                    status="error",
                    message=error_msg
                )
        
        # Generate file content
        content = self._generate_task_markdown(task)
        
        # Determine file path
        target_dir = self.status_dirs[status]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{task_number:03d}_{self._title_to_filename(title)}.md"
        file_path = target_dir / filename
        
        # Write file
        file_path.write_text(content, encoding='utf-8')
        
        return FileOperationResult(
            file_path=str(file_path.relative_to(self.project_root)),
            content=content,
            operation="created",
            task_number=task_number,
            status="success",
            message=f"Task #{task_number:03d} created successfully"
        )
    
    #----------------------------------------------------------------------
    # update_task
    # Purpose: Update an existing task
    # Args:
    #   task_number: Task number to update
    #   updates: Dictionary of fields to update
    # Returns: FileOperationResult with updated file path and content
    #----------------------------------------------------------------------
    def update_task(
        self,
        task_number: int,
        updates: Dict[str, Any]
    ) -> FileOperationResult:
        # Find existing task
        exists, current_dir = self.validator.task_exists(task_number)
        if not exists:
            return FileOperationResult(
                file_path="",
                content="",
                operation="update",
                task_number=task_number,
                status="error",
                message=f"Task #{task_number:03d} not found"
            )
        
        # Read current task
        current_path = self._find_task_file(task_number, Path(current_dir))
        if not current_path:
            return FileOperationResult(
                file_path="",
                content="",
                operation="update",
                task_number=task_number,
                status="error",
                message=f"Task file not found for #{task_number:03d}"
            )
        
        current_content = current_path.read_text(encoding='utf-8')
        
        # Parse current task
        task = self._parse_task_markdown(current_content, task_number)
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        # Validate updated task
        is_valid, errors = self.validator.validate_task_schema(task)
        if not is_valid:
            return FileOperationResult(
                file_path="",
                content="",
                operation="update",
                task_number=task_number,
                status="error",
                message=f"Validation failed: {', '.join(errors)}"
            )
        
        # Generate updated content
        new_content = self._generate_task_markdown(task)
        
        # Write updated file
        current_path.write_text(new_content, encoding='utf-8')
        
        return FileOperationResult(
            file_path=str(current_path.relative_to(self.project_root)),
            content=new_content,
            operation="updated",
            task_number=task_number,
            status="success",
            message=f"Task #{task_number:03d} updated successfully"
        )
    
    #----------------------------------------------------------------------
    # update_task_status
    # Purpose: Update task status (moves file between directories)
    # Args:
    #   task_number: Task number
    #   new_status: New status
    #   notes: Optional notes about status change
    # Returns: FileOperationResult with new file path
    #----------------------------------------------------------------------
    def update_task_status(
        self,
        task_number: int,
        new_status: TaskStatus,
        notes: Optional[str] = None
    ) -> FileOperationResult:
        # Find current task
        exists, current_dir = self.validator.task_exists(task_number)
        if not exists:
            return FileOperationResult(
                file_path="",
                content="",
                operation="move",
                task_number=task_number,
                status="error",
                message=f"Task #{task_number:03d} not found"
            )
        
        current_path = self._find_task_file(task_number, Path(current_dir))
        if not current_path:
            return FileOperationResult(
                file_path="",
                content="",
                operation="move",
                task_number=task_number,
                status="error",
                message=f"Task file not found"
            )
        
        # Read and parse task
        content = current_path.read_text(encoding='utf-8')
        task = self._parse_task_markdown(content, task_number)
        
        # Update status
        old_status = task.status
        task.status = new_status
        
        # Add notes if provided
        if notes:
            if task.notes:
                task.notes += f"\n\n**Status Change ({datetime.now().strftime('%Y-%m-%d')}):** {old_status.value} → {new_status.value}\n{notes}"
            else:
                task.notes = f"**Status Change ({datetime.now().strftime('%Y-%m-%d')}):** {old_status.value} → {new_status.value}\n{notes}"
        
        # Generate updated content
        new_content = self._generate_task_markdown(task)
        
        # Determine new directory
        new_dir = self.status_dirs[new_status]
        new_dir.mkdir(parents=True, exist_ok=True)
        
        new_path = new_dir / current_path.name
        
        # Move file
        current_path.rename(new_path)
        
        # Write updated content
        new_path.write_text(new_content, encoding='utf-8')
        
        return FileOperationResult(
            file_path=str(new_path.relative_to(self.project_root)),
            content=new_content,
            operation="moved",
            task_number=task_number,
            status="success",
            message=f"Task #{task_number:03d} moved to {new_status.value}"
        )
    
    #----------------------------------------------------------------------
    # get_task
    # Purpose: Get task by number
    # Args:
    #   task_number: Task number to retrieve
    # Returns: Dictionary of task details or None
    #----------------------------------------------------------------------
    def get_task(self, task_number: int) -> Optional[Dict]:
        exists, task_dir = self.validator.task_exists(task_number)
        if not exists:
            return None
        
        task_path = self._find_task_file(task_number, Path(task_dir))
        if not task_path:
            return None
        
        content = task_path.read_text(encoding='utf-8')
        task = self._parse_task_markdown(content, task_number)
        
        return {
            "number": task.number,
            "title": task.title,
            "status": task.status.value,
            "priority": task.priority.value,
            "lead": task.lead,
            "file_path": str(task_path.relative_to(self.project_root)),
            "content": content
        }
    
    #----------------------------------------------------------------------
    # list_tasks
    # Purpose: List tasks with optional filters
    # Args:
    #   status: Filter by status
    #   priority: Filter by priority
    # Returns: List of task dictionaries
    #----------------------------------------------------------------------
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None
    ) -> List[Dict]:
        tasks = []
        
        # Determine which directories to search
        if status:
            dirs_to_search = [self.status_dirs[status]]
        else:
            dirs_to_search = list(self.status_dirs.values())
        
        # Search directories
        for task_dir in dirs_to_search:
            if not task_dir.exists():
                continue
            
            for file_path in task_dir.glob("*.md"):
                # Extract task number from filename
                match = re.match(r"^(\d{3})_", file_path.name)
                if not match:
                    continue
                
                task_num = int(match.group(1))
                content = file_path.read_text(encoding='utf-8')
                task = self._parse_task_markdown(content, task_num)
                
                # Apply priority filter
                if priority and task.priority != priority:
                    continue
                
                tasks.append({
                    "number": task.number,
                    "title": task.title,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "lead": task.lead,
                    "file_path": str(file_path.relative_to(self.project_root))
                })
        
        # Sort by task number
        tasks.sort(key=lambda x: x["number"])
        return tasks
    
    #----------------------------------------------------------------------
    # search_tasks
    # Purpose: Search tasks by content
    # Args:
    #   query: Search string
    # Returns: List of matching tasks with snippets
    #----------------------------------------------------------------------
    def search_tasks(self, query: str) -> List[Dict]:
        results = []
        query_lower = query.lower()
        
        for task_dir in self.status_dirs.values():
            if not task_dir.exists():
                continue
            
            for file_path in task_dir.glob("*.md"):
                content = file_path.read_text(encoding='utf-8')
                
                if query_lower in content.lower():
                    match = re.match(r"^(\d{3})_", file_path.name)
                    if match:
                        task_num = int(match.group(1))
                        task = self._parse_task_markdown(content, task_num)
                        
                        results.append({
                            "number": task.number,
                            "title": task.title,
                            "status": task.status.value,
                            "priority": task.priority.value,
                            "file_path": str(file_path.relative_to(self.project_root)),
                            "matches": self._find_matches(content, query)
                        })
        
        return results
    
    # Helper methods
    
    #----------------------------------------------------------------------
    # _generate_task_markdown
    # Purpose: Generate markdown content from task schema
    # Args:
    #   task: The TaskSchema object
    # Returns: Formatted markdown string
    #----------------------------------------------------------------------
    def _generate_task_markdown(self, task: TaskSchema) -> str:
        # Handle both enum and string values for status/priority
        status_value = task.status.value if isinstance(task.status, TaskStatus) else task.status
        priority_value = task.priority.value if isinstance(task.priority, TaskPriority) else task.priority
        
        lines = [
            f"# TASK: {task.title}",
            "",
            f"**Status:** {status_value}",
            f"**Priority:** {priority_value}",
            f"**Lead:** {task.lead}",
            f"**Dependencies:** {task.dependencies}",
            f"**Related Documents:** {task.related_documents}",
            "",
            "---",
            "",
            "## 1. Objective",
            "",
            task.objective,
            "",
            "## 2. Deliverables",
            ""
        ]
        
        for i, deliverable in enumerate(task.deliverables, 1):
            lines.append(f"{i}. {deliverable}")
        
        lines.extend([
            "",
            "## 3. Acceptance Criteria",
            ""
        ])
        
        for criterion in task.acceptance_criteria:
            lines.append(f"- {criterion}")
        
        if task.notes:
            lines.extend([
                "",
                "## Notes",
                "",
                task.notes
            ])
        
        lines.append("")  # Final newline
        return "\n".join(lines)
    
    #----------------------------------------------------------------------
    # _parse_task_markdown
    # Purpose: Parse markdown content into task schema
    # Args:
    #   content: Markdown content string
    #   task_number: Task number
    # Returns: TaskSchema object
    #----------------------------------------------------------------------
    def _parse_task_markdown(self, content: str, task_number: int) -> TaskSchema:
        lines = content.split("\n")
        
        # Extract metadata
        title = ""
        status = TaskStatus.BACKLOG
        priority = TaskPriority.MEDIUM
        lead = "Unassigned"
        dependencies = "None"
        related_documents = "None"
        objective = ""
        deliverables = []
        acceptance_criteria = []
        notes = ""
        
        current_section = None
        
        for line in lines:
            # Title
            if line.startswith("# TASK:"):
                title = line.replace("# TASK:", "").strip()
            
            # Metadata
            elif line.startswith("**Status:**"):
                status_str = line.split("**Status:**")[1].strip()
                # Case-insensitive lookup
                try:
                    status = TaskStatus(status_str.lower())
                except ValueError:
                    # Try to match by value (case-insensitive)
                    for s in TaskStatus:
                        if s.value.lower() == status_str.lower():
                            status = s
                            break
            elif line.startswith("**Priority:**"):
                priority_str = line.split("**Priority:**")[1].strip()
                # Case-insensitive lookup
                try:
                    priority = TaskPriority(priority_str)
                except ValueError:
                    # Try to match by value (case-insensitive)
                    for p in TaskPriority:
                        if p.value.lower() == priority_str.lower():
                            priority = p
                            break
            elif line.startswith("**Lead:**"):
                lead = line.split("**Lead:**")[1].strip()
            elif line.startswith("**Dependencies:**"):
                dependencies = line.split("**Dependencies:**")[1].strip()
            elif line.startswith("**Related Documents:**"):
                related_documents = line.split("**Related Documents:**")[1].strip()
            
            # Sections
            elif line.startswith("## 1. Objective"):
                current_section = "objective"
            elif line.startswith("## 2. Deliverables"):
                current_section = "deliverables"
            elif line.startswith("## 3. Acceptance Criteria"):
                current_section = "acceptance"
            elif line.startswith("## Notes"):
                current_section = "notes"
            elif line.startswith("##"):
                current_section = None
            
            # Content
            elif current_section == "objective" and line.strip() and not line.startswith("---"):
                objective += line + "\n"
            elif current_section == "deliverables" and line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                deliverables.append(line.split(".", 1)[1].strip())
            elif current_section == "acceptance" and line.strip().startswith("-"):
                acceptance_criteria.append(line.strip()[2:])
            elif current_section == "notes" and line.strip():
                notes += line + "\n"
        
        return TaskSchema(
            number=task_number,
            title=title,
            status=status,
            priority=priority,
            lead=lead,
            dependencies=dependencies,
            related_documents=related_documents,
            objective=objective.strip(),
            deliverables=deliverables,
            acceptance_criteria=acceptance_criteria,
            notes=notes.strip() if notes else None
        )
    
    #----------------------------------------------------------------------
    # _title_to_filename
    # Purpose: Convert title to filename-safe format
    # Args:
    #   title: Task title
    # Returns: Safe filename string
    #----------------------------------------------------------------------
    def _title_to_filename(self, title: str) -> str:
        # Lowercase
        filename = title.lower()
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        # Remove special characters
        filename = re.sub(r'[^a-z0-9_]', '', filename)
        # Limit length
        return filename[:50]
    
    #----------------------------------------------------------------------
    # _find_task_file
    # Purpose: Find task file by number in directory
    # Args:
    #   task_number: Task number
    #   directory: Directory to search
    # Returns: Path to file or None
    #----------------------------------------------------------------------
    def _find_task_file(self, task_number: int, directory: Path) -> Optional[Path]:
        pattern = f"{task_number:03d}_*.md"
        files = list(directory.glob(pattern))
        return files[0] if files else None
    
    #----------------------------------------------------------------------
    # _find_matches
    # Purpose: Find matching lines in content
    # Args:
    #   content: Text content to search
    #   query: Search string
    # Returns: List of matching lines (max 3)
    #----------------------------------------------------------------------
    def _find_matches(self, content: str, query: str) -> List[str]:
        matches = []
        query_lower = query.lower()
        
        for line in content.split("\n"):
            if query_lower in line.lower():
                matches.append(line.strip())
                if len(matches) >= 3:  # Limit to 3 matches
                    break
        
        return matches
