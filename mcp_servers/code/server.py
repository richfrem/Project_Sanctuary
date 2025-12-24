#============================================
# mcp_servers/code/server.py
# Purpose: MCP Server for FileSystem and Code Operations.
#          Provides tools for listing, reading, writing, and analyzing code.
# Role: Interface Layer
# Used as: Main service entry point for the mcp_servers.code module.
#============================================

import os
import sys
import json
import logging
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.code.operations import CodeOperations
from .models import (
    CodeAnalysisRequest,
    CodeLintRequest,
    CodeFormatRequest,
    CodeFindFileRequest,
    CodeListFilesRequest,
    CodeSearchContentRequest,
    CodeReadRequest,
    CodeWriteRequest,
    CodeGetInfoRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("sanctuary_filesystem")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.filesystem.code",
    instructions="""
    Use this server for all file system and code quality operations.
    - Read, write, and list files in the workspace.
    - Run linting (ruff) and formatting (ruff/black) on source code.
    - Search for content across files and perform static analysis.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
ops = CodeOperations(PROJECT_ROOT)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def code_lint(request: CodeLintRequest) -> str:
    """Run linting on a file or directory."""
    try:
        result = ops.lint(request.path, request.tool)
        output = [f"Linting {result['path']} with {result['tool']}:", ""]
        if result['issues_found']:
            output.append("❌ Issues found:")
            output.append(result['output'])
        else:
            output.append("✅ No issues found")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_lint: {e}")
        raise ToolError(f"Linting failed: {str(e)}")

@mcp.tool()
async def code_format(request: CodeFormatRequest) -> str:
    """Format code in a file or directory."""
    try:
        result = ops.format_code(request.path, request.tool, request.check_only)
        output = [f"Formatting {result['path']} with {result['tool']}:", ""]
        if request.check_only:
            if result['success']:
                output.append("✅ Code is properly formatted")
            else:
                output.append("❌ Code needs formatting")
                output.append(result['output'])
        else:
            if result['modified']:
                output.append("✅ Code formatted successfully")
            else:
                output.append("❌ Formatting failed")
                output.append(result['output'])
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_format: {e}")
        raise ToolError(f"Formatting failed: {str(e)}")

@mcp.tool()
async def code_analyze(request: CodeAnalysisRequest) -> str:
    """Perform static analysis on code."""
    try:
        result = ops.analyze(request.path)
        output = [f"Analyzing {result['path']}:", "", result['statistics']]
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_analyze: {e}")
        raise ToolError(f"Analysis failed: {str(e)}")

@mcp.tool()
async def code_check_tools() -> str:
    """Check which code quality tools are available."""
    try:
        tools = ["ruff", "black", "pylint", "flake8", "mypy"]
        available = []
        unavailable = []
        for tool in tools:
            if ops.check_tool_available(tool):
                available.append(f"✅ {tool}")
            else:
                unavailable.append(f"❌ {tool}")
        output = ["Available code tools:", ""]
        output.extend(available)
        if unavailable:
            output.append("")
            output.append("Unavailable:")
            output.extend(unavailable)
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_check_tools: {e}")
        raise ToolError(f"Tool check failed: {str(e)}")

@mcp.tool()
async def code_find_file(request: CodeFindFileRequest) -> str:
    """Find files by name or glob pattern."""
    try:
        matches = ops.find_file(request.name_pattern, request.path)
        if not matches:
            return f"No files found matching '{request.name_pattern}' in '{request.path}'"
        output = [f"Found {len(matches)} file(s) matching '{request.name_pattern}':", ""]
        for match in matches:
            output.append(f"  {match}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_find_file: {e}")
        raise ToolError(f"Find file failed: {str(e)}")

@mcp.tool()
async def code_list_files(request: CodeListFilesRequest) -> str:
    """List files in a directory with pattern support."""
    try:
        files = ops.list_files(request.path, request.pattern, request.recursive)
        if not files:
            return f"No files found in '{request.path}' matching '{request.pattern}'"
        output = [f"Found {len(files)} file(s) in '{request.path}':", ""]
        for file_info in files:
            size_kb = file_info['size'] / 1024
            output.append(f"  {file_info['path']} ({size_kb:.1f} KB)")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_list_files: {e}")
        raise ToolError(f"List files failed: {str(e)}")

@mcp.tool()
async def code_search_content(request: CodeSearchContentRequest) -> str:
    """Search for text/patterns in code files."""
    try:
        matches = ops.search_content(request.query, request.file_pattern, request.case_sensitive)
        if not matches:
            return f"No matches found for '{request.query}' in files matching '{request.file_pattern}'"
        output = [f"Found {len(matches)} match(es) for '{request.query}':", ""]
        current_file = None
        for match in matches[:50]:
            if match['file'] != current_file:
                current_file = match['file']
                output.append(f"\n{current_file}:")
            output.append(f"  Line {match['line']}: {match['content']}")
        if len(matches) > 50:
            output.append(f"\n... and {len(matches) - 50} more matches")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_search_content: {e}")
        raise ToolError(f"Search content failed: {str(e)}")

@mcp.tool()
async def code_read(request: CodeReadRequest) -> str:
    """Read file contents."""
    try:
        content = ops.read_file(request.path, request.max_size_mb)
        # We don't wrap in decorators anymore, just return string
        return content
    except Exception as e:
        logger.error(f"Error in code_read: {e}")
        raise ToolError(f"Read failed: {str(e)}")

@mcp.tool()
async def code_write(request: CodeWriteRequest) -> str:
    """Write/update file with automatic backup."""
    try:
        result = ops.write_file(request.path, request.content, request.backup, request.create_dirs)
        output = [f"{'Created' if result['created'] else 'Updated'} file: {result['path']}"]
        output.append(f"Size: {result['size']} bytes")
        if result['backup']:
            output.append(f"Backup: {result['backup']}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_write: {e}")
        raise ToolError(f"Write failed: {str(e)}")

@mcp.tool()
async def code_get_info(request: CodeGetInfoRequest) -> str:
    """Get file metadata."""
    try:
        import time
        info = ops.get_file_info(request.path)
        output = [f"File info for {info['path']}:", ""]
        output.append(f"  Language: {info['language']}")
        output.append(f"  Size: {info['size']} bytes ({info['size']/1024:.1f} KB)")
        output.append(f"  Lines: {info['lines'] if info['lines'] else 'N/A'}")
        output.append(f"  Modified: {time.ctime(info['modified'])}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error in code_get_info: {e}")
        raise ToolError(f"Get info failed: {str(e)}")

#============================================
# Main Execution Entry Point
#============================================

if __name__ == "__main__":
    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Local/CLI Mode)
    port_env = get_env_variable("PORT", required=False)
    transport = "sse" if port_env else "stdio"
    
    if transport == "sse":
        port = int(port_env) if port_env else 8005
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
