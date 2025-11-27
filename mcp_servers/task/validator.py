"""
Task MCP Server - Schema Validator
Validates tasks against TASKS/task_schema.md
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from .models import TaskSchema, TaskStatus, TaskPriority


class TaskValidator:
    """Validates task files against canonical schema"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tasks_dir = project_root / "TASKS"
        
    def validate_task_number(self, number: int) -> Tuple[bool, str]:
        """
        Validate task number is unique across all task directories
        Returns: (is_valid, error_message)
        """
        task_dirs = [
            self.tasks_dir / "backlog",
            self.tasks_dir / "todo", 
            self.tasks_dir / "in-progress",
            self.tasks_dir / "done"
        ]
        
        task_pattern = re.compile(rf"^{number:03d}_.*\.md$")
        
        for task_dir in task_dirs:
            if not task_dir.exists():
                continue
                
            for file in task_dir.iterdir():
                if task_pattern.match(file.name):
                    return False, f"Task #{number:03d} already exists in {task_dir.name}/"
        
        return True, ""
    
    def validate_task_schema(self, task: TaskSchema) -> Tuple[bool, List[str]]:
        """
        Validate task follows required schema
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        if not task.title:
            errors.append("Title is required")
        
        if not task.objective:
            errors.append("Objective section is required")
            
        if not task.deliverables or len(task.deliverables) == 0:
            errors.append("At least one deliverable is required")
            
        if not task.acceptance_criteria or len(task.acceptance_criteria) == 0:
            errors.append("At least one acceptance criterion is required")
        
        # Validate status
        if task.status not in TaskStatus:
            errors.append(f"Invalid status: {task.status}")
        
        # Validate priority
        if task.priority not in TaskPriority:
            errors.append(f"Invalid priority: {task.priority}")
        
        # Validate task number format
        if task.number < 1 or task.number > 999:
            errors.append("Task number must be between 1 and 999")
        
        return len(errors) == 0, errors
    
    def validate_dependencies(self, dependencies_str: str) -> Tuple[bool, str]:
        """
        Validate task dependencies format and check for circular dependencies
        Returns: (is_valid, error_message)
        """
        if not dependencies_str or dependencies_str.lower() == "none":
            return True, ""
        
        # Extract task numbers from dependencies string
        task_refs = re.findall(r'#(\d+)', dependencies_str)
        
        if not task_refs:
            return True, ""  # No task references found, that's okay
        
        # Check if referenced tasks exist
        for ref in task_refs:
            task_num = int(ref)
            exists, _ = self.task_exists(task_num)
            if not exists:
                return False, f"Referenced task #{task_num:03d} does not exist"
        
        return True, ""
    
    def task_exists(self, number: int) -> Tuple[bool, str]:
        """
        Check if a task exists in any directory
        Returns: (exists, directory_path)
        """
        task_dirs = [
            self.tasks_dir / "backlog",
            self.tasks_dir / "todo",
            self.tasks_dir / "in-progress", 
            self.tasks_dir / "done"
        ]
        
        task_pattern = re.compile(rf"^{number:03d}_.*\.md$")
        
        for task_dir in task_dirs:
            if not task_dir.exists():
                continue
                
            for file in task_dir.iterdir():
                if task_pattern.match(file.name):
                    return True, str(task_dir)
        
        return False, ""
    
    def validate_file_path(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate file path is within TASKS directory
        Returns: (is_valid, error_message)
        """
        try:
            file_path.resolve().relative_to(self.tasks_dir.resolve())
            return True, ""
        except ValueError:
            return False, f"File path must be within TASKS directory: {file_path}"
