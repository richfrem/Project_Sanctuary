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

    def find_file(self, name_pattern: str, directory: str = ".") -> List[str]:
        """
        Find files by name or glob pattern.
        
        Args:
            name_pattern: File name or glob pattern (e.g., "server.py", "*.py")
            directory: Directory to search in (relative to project root)
            
        Returns:
            List of matching file paths (relative to project root)
        """
        search_dir = self._validate_path(directory)
        
        if not search_dir.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        matches = []
        for file_path in search_dir.rglob(name_pattern):
            if file_path.is_file():
                matches.append(str(file_path.relative_to(self.project_root)))
        
        return sorted(matches)

    def list_files(self, directory: str = ".", pattern: str = "*", recursive: bool = True) -> List[Dict[str, Any]]:
        """
        List files in a directory with optional pattern.
        
        Args:
            directory: Directory to list (relative to project root)
            pattern: Glob pattern for filtering (default: "*")
            recursive: If True, search recursively
            
        Returns:
            List of dicts with file info (path, size, modified)
        """
        search_dir = self._validate_path(directory)
        
        if not search_dir.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        files = []
        glob_method = search_dir.rglob if recursive else search_dir.glob
        
        for file_path in glob_method(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "path": str(file_path.relative_to(self.project_root)),
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        
        return sorted(files, key=lambda x: x["path"])

    def search_content(self, query: str, file_pattern: str = "*.py", case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for text/patterns in code files.
        
        Args:
            query: Text or pattern to search for
            file_pattern: File pattern to search in (default: "*.py")
            case_sensitive: If True, perform case-sensitive search
            
        Returns:
            List of matches with file path, line number, and context
        """
        import re
        
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
                # Skip binary files or files we can't read
                continue
                
        return matches

    def read_file(self, path: str, max_size_mb: int = 10) -> str:
        """
        Read file contents.
        
        Args:
            path: Relative path to file
            max_size_mb: Maximum file size in MB (default: 10)
            
        Returns:
            File contents as string
        """
        file_path = self._validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
            
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"File too large ({size_mb:.1f}MB > {max_size_mb}MB): {path}")
            
        return file_path.read_text(encoding='utf-8')

    def write_file(self, path: str, content: str, backup: bool = True, create_dirs: bool = True) -> Dict[str, Any]:
        """
        Write/update file with automatic backup.
        
        Args:
            path: Relative path to file
            content: Content to write
            backup: If True, create backup before overwriting
            create_dirs: If True, create parent directories if needed
            
        Returns:
            Dict with operation results
        """
        import shutil
        import time
        
        file_path = self._validate_path(path)
        
        # Check if file exists before writing
        file_existed = file_path.exists()
        
        # Create parent directories if needed
        if create_dirs and not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists
        backup_path = None
        if backup and file_existed:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f"{file_path.suffix}.{timestamp}.bak")
            shutil.copy2(file_path, backup_path)
        
        # Write file
        file_path.write_text(content, encoding='utf-8')
        
        return {
            "path": str(file_path.relative_to(self.project_root)),
            "size": len(content),
            "backup": str(backup_path.relative_to(self.project_root)) if backup_path else None,
            "created": not file_existed
        }

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """
        Get file metadata.
        
        Args:
            path: Relative path to file
            
        Returns:
            Dict with file metadata
        """
        file_path = self._validate_path(path)
        
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
        
        # Count lines
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
