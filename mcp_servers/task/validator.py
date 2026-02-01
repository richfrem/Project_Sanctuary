#!/usr/bin/env python3
"""
Task Validator
=====================================

Purpose:
    Validation logic for Task MCP.
    Validates task schema, unique IDs, dependency chains, and file paths.

Layer: Validation (Logic)

Key Classes:
    - TaskValidator: Main validation logic
        - __init__(project_root)
        - validate_task_number(number)
        - get_next_task_number()
        - validate_task_schema(task)
        - validate_dependencies(dependencies_str)
        - task_exists(number)
        - validate_file_path(file_path)
"""

"""
Task MCP Server - Schema Validator
Validates tasks against tasks/task_schema.md
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from .models import taskschema, taskstatus, TaskPriority


class TaskValidator:
    """Validates task files against canonical schema"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tasks_dir = project_root / "tasks"
        
    #----------------------------------------------------------------------
    # validate_task_number
    # Purpose: Validate task number is unique across all task directories
    # Args:
    #   number: The task number to check
    # Returns: (is_valid, error_message)
    #----------------------------------------------------------------------
    def validate_task_number(self, number: int) -> Tuple[bool, str]:
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
    
    #----------------------------------------------------------------------
    # get_next_task_number
    # Purpose: Get the next sequential task number by scanning all task directories.
    # Returns: The next available task integer
    #----------------------------------------------------------------------
    def get_next_task_number(self) -> int:
        existing_numbers = []
        
        # Scan all status directories
        task_dirs = [
            self.tasks_dir / "backlog",
            self.tasks_dir / "todo",
            self.tasks_dir / "in-progress",
            self.tasks_dir / "done"
        ]
        
        for task_dir in task_dirs:
            if not task_dir.exists():
                continue
            
            for file in task_dir.iterdir():
                if file.suffix == '.md':
                    # Extract task number from filename (format: 001_task_name.md)
                    match = re.match(r'^(\d{3})_', file.name)
                    if match:
                        existing_numbers.append(int(match.group(1)))
        
        # Return next number
        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1
    
    #----------------------------------------------------------------------
    # validate_task_schema
    # Purpose: Validate task follows required schema
    # Args:
    #   task: The taskschema object to validate
    # Returns: (is_valid, list_of_errors)
    #----------------------------------------------------------------------
    def validate_task_schema(self, task: taskschema) -> Tuple[bool, List[str]]:
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
        if task.status not in taskstatus:
            errors.append(f"Invalid status: {task.status}")
        
        # Validate priority
        if task.priority not in TaskPriority:
            errors.append(f"Invalid priority: {task.priority}")
        
        # Validate task number format
        if task.number < 1 or task.number > 999:
            errors.append("Task number must be between 1 and 999")
        
        return len(errors) == 0, errors
    
    #----------------------------------------------------------------------
    # validate_dependencies
    # Purpose: Validate task dependencies format and check for circular dependencies
    # Args:
    #   dependencies_str: The dependencies string to validate
    # Returns: (is_valid, error_message)
    #----------------------------------------------------------------------
    def validate_dependencies(self, dependencies_str: str) -> Tuple[bool, str]:
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
    
    #----------------------------------------------------------------------
    # task_exists
    # Purpose: Check if a task exists in any directory
    # Args:
    #   number: The task number to look for
    # Returns: (exists, directory_path)
    #----------------------------------------------------------------------
    def task_exists(self, number: int) -> Tuple[bool, str]:
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
    
    #----------------------------------------------------------------------
    # validate_file_path
    # Purpose: Validate file path is within tasks directory
    # Args:
    #   file_path: The path to validate
    # Returns: (is_valid, error_message)
    #----------------------------------------------------------------------
    def validate_file_path(self, file_path: Path) -> Tuple[bool, str]:
        try:
            file_path.resolve().relative_to(self.tasks_dir.resolve())
            return True, ""
        except ValueError:
            return False, f"File path must be within tasks directory: {file_path}"
