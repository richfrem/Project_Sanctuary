#============================================
# mcp_servers/code/operations.py
# Purpose: Core business logic for Code Operations.
#          Handles file I/O, code analysis, and tool execution.
# Role: Business Logic Layer
# Used as: Helper module by server.py
# LIST OF CLASSES/FUNCTIONS:
#   - CodeOperations
#     - __init__
#     - lint
#     - format_code
#     - analyze
#     - check_tool_available
#     - find_file
#     - list_files
#     - search_content
#     - read_file
#     - write_file
#     - get_file_info
#     - _run_command
#     - delete_file
#============================================

import subprocess
import shutil
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from .validator import CodeValidator
from .models import FileInfo, OperationResult

class CodeOperations:
    """
    Class: CodeOperations
    Purpose: Operations for code analysis, linting, and formatting.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {project_root}")
        
        # Initialize Validator
        self.validator = CodeValidator(self.project_root)

    #============================================
    # Method: _run_command
    # Purpose: Execute shell command safely.
    # Args:
    #   cmd: List of command parts
    #   cwd: Working directory (optional)
    # Returns: Dict with execution result
    #============================================
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

    #============================================
    # Method: lint
    # Purpose: Run linting tool on target.
    # Args:
    #   path: Relative file/dir path
    #   tool: Tool name (ruff, pylint, flake8)
    # Returns: Dict with lint results
    #============================================
    def lint(self, path: str, tool: str = "ruff") -> Dict[str, Any]:
        """Run linting on a file or directory."""
        file_path = self.validator.validate_path(path)
        
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

    #============================================
    # Method: format_code
    # Purpose: Format code using specified tool.
    # Args:
    #   path: Relative file/dir path
    #   tool: Tool name (ruff, black)
    #   check_only: Dry run flag
    # Returns: Dict with format results
    #============================================
    def format_code(self, path: str, tool: str = "ruff", check_only: bool = False) -> Dict[str, Any]:
        """Format code in a file or directory."""
        file_path = self.validator.validate_path(path)
        
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

    #============================================
    # Method: analyze
    # Purpose: Perform static analysis.
    # Args:
    #   path: Relative file/dir path
    # Returns: Dict with analysis stats
    #============================================
    def analyze(self, path: str) -> Dict[str, Any]:
        """Perform static analysis on code."""
        file_path = self.validator.validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Use ruff for basic analysis
        result = self._run_command(["ruff", "check", "--statistics", str(file_path)])
        
        return {
            "path": str(file_path.relative_to(self.project_root)),
            "success": result["success"],
            "statistics": result["stdout"] if result["stdout"] else result["stderr"]
        }

    #============================================
    # Method: check_tool_available
    # Purpose: Verify tool presence in environment.
    # Args:
    #   tool: Tool binary name
    # Returns: Boolean presence
    #============================================
    def check_tool_available(self, tool: str) -> bool:
        """Check if a code tool is available."""
        result = self._run_command(["which", tool])
        return result["success"]

    #============================================
    # Method: find_file
    # Purpose: Search for files by name pattern.
    # Args:
    #   name_pattern: Glob pattern
    #   directory: Search root (relative)
    # Returns: List of relative paths
    #============================================
    def find_file(self, name_pattern: str, directory: str = ".") -> List[str]:
        """Find files by name or glob pattern."""
        search_dir = self.validator.validate_path(directory)
        
        if not search_dir.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        matches = []
        for file_path in search_dir.rglob(name_pattern):
            if file_path.is_file():
                matches.append(str(file_path.relative_to(self.project_root)))
        
        return sorted(matches)

    #============================================
    # Method: list_files
    # Purpose: List directory contents.
    # Args:
    #   directory: Target directory
    #   pattern: Filter pattern
    #   recursive: Deep search flag
    # Returns: List of file metadata dicts
    #============================================
    def list_files(self, directory: str = ".", pattern: str = "*", recursive: bool = True, max_files: int = 5000) -> List[Dict[str, Any]]:
        """List files in a directory with optional pattern.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            recursive: Whether to search subdirectories (iterative tree walk, not recursive calls)
            max_files: Maximum number of files to return (default 5000, prevents unbounded scans)
        """
        search_dir = self.validator.validate_path(directory)
        
        if not search_dir.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        files = []
        glob_method = search_dir.rglob if recursive else search_dir.glob
        
        # Pre-compute project root for faster relative path calculations
        project_root = self.project_root
        
        for file_path in glob_method(pattern):
            # Skip symlinks first (fastest check)
            if file_path.is_symlink():
                continue
            
            # Then check if it's a file (requires stat call)
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    files.append({
                        "path": str(file_path.relative_to(project_root)),
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
                except (OSError, ValueError):
                    # Skip files we can't stat or get relative path for
                    continue
                
                # Safety limit to prevent unbounded scans
                if len(files) >= max_files:
                    break
        
        # Return unsorted - filesystem order is fine and much faster
        return files

    #============================================
    # Method: search_content
    # Purpose: Grep-like content search.
    # Args:
    #   query: Regex/Text query
    #   file_pattern: File filter
    #   case_sensitive: Case flag
    # Returns: List of match dicts
    #============================================
    def search_content(self, query: str, file_pattern: str = "*.py", case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Search for text/patterns in code files."""
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)
        
        for file_info in self.list_files(".", file_pattern, recursive=True):
            file_path = self.project_root / file_info["path"]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern.search(line):
                            matches.append({
                                "file": file_info["path"],
                                "line": line_num,
                                "content": line.rstrip()
                            })
            except (UnicodeDecodeError, PermissionError):
                continue
                
        return matches

    #============================================
    # Method: read_file
    # Purpose: Read file content safely.
    # Args:
    #   path: Relative file path
    #   max_size_mb: Size limit constraint
    # Returns: File content string
    #============================================
    def read_file(self, path: str, max_size_mb: int = 10) -> str:
        """Read file contents."""
        file_path = self.validator.validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
            
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"File too large ({size_mb:.1f}MB > {max_size_mb}MB): {path}")
            
        return file_path.read_text(encoding='utf-8')

    #============================================
    # Method: write_file
    # Purpose: Write file handling backups and safety.
    # Args:
    #   path: Relative path
    #   content: New content
    #   backup: Backup flag
    #   create_dirs: Mkdir flag
    # Returns: Dict with write stats
    #============================================
    def write_file(self, path: str, content: str, backup: bool = True, create_dirs: bool = True) -> Dict[str, Any]:
        """Write/update file with safety checks and backup."""
        file_path = self.validator.validate_path(path)
        
        # POKA-YOKE: Check safety enforced by validator
        self.validator.validate_safe_write(file_path, content)
        
        file_existed = file_path.exists()
        
        if create_dirs and not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        backup_path = None
        if backup and file_existed:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f"{file_path.suffix}.{timestamp}.bak")
            shutil.copy2(file_path, backup_path)
        
        file_path.write_text(content, encoding='utf-8')
        
        return {
            "path": str(file_path.relative_to(self.project_root)),
            "size": len(content),
            "backup": str(backup_path.relative_to(self.project_root)) if backup_path else None,
            "created": not file_existed
        }

    #============================================
    # Method: get_file_info
    # Purpose: Retrieve file metadata.
    # Args:
    #   path: Relative path
    # Returns: Dict with stats
    #============================================
    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file metadata."""
        file_path = self.validator.validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
            
        stat = file_path.stat()
        
        # Detect language from extension
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
        except (UnicodeDecodeError, PermissionError):
            line_count = None
        
        return {
            "path": str(file_path.relative_to(self.project_root)),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "lines": line_count,
            "language": ext_to_lang.get(file_path.suffix, "Unknown")
        }

    #============================================
    # Method: delete_file
    # Purpose: Delete a file with safety checks.
    # Args:
    #   path: Relative file path
    #   force: Skip confirmation for protected patterns
    # Returns: Dict with deletion status
    #============================================
    def delete_file(self, path: str, force: bool = False) -> Dict[str, Any]:
        """Delete a file with safety checks.
        
        Safety constraints:
        - Path must be within project root
        - File must exist and be a file (not directory)
        - Protected patterns (*.py test files, config) require force=True
        """
        file_path = self.validator.validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        if not file_path.is_file():
            raise ValueError(f"Path is not a file (use rmdir for directories): {path}")
        
        # Protected patterns - require force=True
        protected_patterns = [
            "*.pyc", "__pycache__/*",  # Never delete compiled files this way
            ".git/*", ".env",          # Never delete git or env
            "server.py", "operations.py", "models.py",  # Never delete core modules
        ]
        
        rel_path = str(file_path.relative_to(self.project_root))
        
        for pattern in protected_patterns:
            if file_path.match(pattern) and not force:
                raise ValueError(
                    f"Protected file pattern '{pattern}' matched. Use force=True to override."
                )
        
        # Perform the deletion
        file_size = file_path.stat().st_size
        file_path.unlink()
        
        return {
            "path": rel_path,
            "deleted": True,
            "size": file_size,
            "message": f"Deleted: {rel_path}"
        }

