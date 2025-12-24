#============================================
# mcp_servers/code/models.py
# Purpose: Data definition layer for Code Server.
# Role: Data Layer
# Used as: Type definitions for operations and validator.
# LIST OF CLASSES/CONSTANTS:
#   - CodeTool (Enum)
#   - FileInfo (DataClass)
#   - OperationResult (DataClass)
#   - HIGH_RISK_FILES (Constant)
#============================================

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Dict, List

# Poka-Yoke: High-Risk File List (Protocol 122)
# These files require content loss prevention checks before writing
HIGH_RISK_FILES = [".gitignore", ".env", ".env.local", "Dockerfile", "package.json"]

#============================================
# Class: CodeTool
# Purpose: Enum for supported code analysis tools.
#============================================
class CodeTool(Enum):
    RUFF = "ruff"
    BLACK = "black"
    PYLINT = "pylint"
    FLAKE8 = "flake8"

#============================================
# Class: FileInfo
# Purpose: DataClass for file metadata.
#============================================
@dataclass
class FileInfo:
    path: str
    size: int
    modified: float
    lines: Optional[int]
    language: str

#============================================
# Class: OperationResult
# Purpose: DataClass for standardized operation results.
#============================================
@dataclass
class OperationResult:
    success: bool
    path: str
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "success": self.success,
            "path": self.path
        }
        if self.output:
            result["output"] = self.output
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result.update(self.metadata)
        return result

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field

class CodeAnalysisRequest(BaseModel):
    path: str = Field(..., description="File or directory path to analyze")

class CodeLintRequest(BaseModel):
    path: str = Field(..., description="File or directory path to lint")
    tool: str = Field("ruff", description="Linting tool name (ruff, pylint, flake8)")

class CodeFormatRequest(BaseModel):
    path: str = Field(..., description="File or directory path to format")
    tool: str = Field("ruff", description="Formatting tool name (ruff, black)")
    check_only: bool = Field(False, description="If True, only checks formatting without applying changes")

class CodeFindFileRequest(BaseModel):
    name_pattern: str = Field(..., description="Name or glob pattern to search for")
    path: str = Field(".", description="Base search directory")

class CodeListFilesRequest(BaseModel):
    path: str = Field(".", description="Directory path to list")
    pattern: str = Field("*", description="Glob pattern filter")
    recursive: bool = Field(True, description="Whether to search subdirectories recursively")

class CodeSearchContentRequest(BaseModel):
    query: str = Field(..., description="Text or regex pattern to search for")
    file_pattern: str = Field("*.py", description="File pattern to include in search")
    case_sensitive: bool = Field(False, description="Whether the search should be case-sensitive")

class CodeReadRequest(BaseModel):
    path: str = Field(..., description="Path to the file to read")
    max_size_mb: int = Field(10, description="Maximum file size in MB to prevent memory issues")

class CodeWriteRequest(BaseModel):
    path: str = Field(..., description="Path to the file to write/update")
    content: str = Field(..., description="Content to write to the file")
    backup: bool = Field(True, description="Whether to create a backup if the file already exists")
    create_dirs: bool = Field(True, description="Whether to create parent directories if they don't exist")

class CodeGetInfoRequest(BaseModel):
    path: str = Field(..., description="Path to the file or directory to get metadata for")
