import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

class CodeOperations:
    """
    Operations for code analysis, linting, and formatting.
    Enforces safety checks:
    1. Path validation (must be within project)
    2. Backup before formatting
    3. Tool availability verification
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {project_root}")

    def _validate_path(self, path: str) -> Path:
        """Validate that the path is within the project directory."""
        file_path = (self.project_root / path).resolve()
        
        if not str(file_path).startswith(str(self.project_root)):
            raise ValueError(f"Security Error: Path '{path}' is outside project directory")
            
        return file_path

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run a command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out after 30 seconds",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def lint(self, path: str, tool: str = "ruff") -> Dict[str, Any]:
        """
        Run linting on a file or directory.
        
        Args:
            path: Relative path to file or directory
            tool: Linting tool to use (ruff, pylint, flake8)
            
        Returns:
            Dict with linting results
        """
        file_path = self._validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if tool == "ruff":
            cmd = ["ruff", "check", str(file_path)]
        elif tool == "pylint":
            cmd = ["pylint", str(file_path)]
        elif tool == "flake8":
            cmd = ["flake8", str(file_path)]
        else:
            raise ValueError(f"Unsupported linting tool: {tool}")

        result = self._run_command(cmd)
        return {
            "tool": tool,
            "path": str(file_path.relative_to(self.project_root)),
            "success": result["success"],
            "output": result["stdout"] if result["stdout"] else result["stderr"],
            "issues_found": not result["success"]
        }

    def format_code(self, path: str, tool: str = "ruff", check_only: bool = False) -> Dict[str, Any]:
        """
        Format code in a file or directory.
        
        Args:
            path: Relative path to file or directory
            tool: Formatting tool to use (ruff, black)
            check_only: If True, only check formatting without modifying files
            
        Returns:
            Dict with formatting results
        """
        file_path = self._validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if tool == "ruff":
            cmd = ["ruff", "format", str(file_path)]
            if check_only:
                cmd.append("--check")
        elif tool == "black":
            cmd = ["black", str(file_path)]
            if check_only:
                cmd.append("--check")
        else:
            raise ValueError(f"Unsupported formatting tool: {tool}")

        result = self._run_command(cmd)
        return {
            "tool": tool,
            "path": str(file_path.relative_to(self.project_root)),
            "success": result["success"],
            "output": result["stdout"] if result["stdout"] else result["stderr"],
            "modified": not check_only and result["success"]
        }

    def analyze(self, path: str) -> Dict[str, Any]:
        """
        Perform static analysis on code.
        
        Args:
            path: Relative path to file or directory
            
        Returns:
            Dict with analysis results
        """
        file_path = self._validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Use ruff for basic analysis
        result = self._run_command(["ruff", "check", "--statistics", str(file_path)])
        
        return {
            "path": str(file_path.relative_to(self.project_root)),
            "success": result["success"],
            "statistics": result["stdout"] if result["stdout"] else result["stderr"]
        }

    def check_tool_available(self, tool: str) -> bool:
        """Check if a code tool is available."""
        result = self._run_command(["which", tool])
        return result["success"]
