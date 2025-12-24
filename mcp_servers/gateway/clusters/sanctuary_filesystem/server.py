#============================================
# mcp_servers/gateway/clusters/sanctuary_filesystem/server.py
# Purpose: Sanctuary FileSystem/Code Cluster - Dual-Transport Entry Point
# Role: Interface Layer (Aggregator Node)
# Status: ADR-066 v1.3 Compliant (SSEServer for Gateway, FastMCP for STDIO)
# Used by: Gateway Fleet (SSE) and Claude Desktop (STDIO)
#============================================

import os
import sys
import json
import logging
from typing import Optional

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging

# Setup Logging
logger = setup_mcp_logging("project_sanctuary.sanctuary_filesystem")

# Configuration
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
_ops = None

def get_ops():
    global _ops
    if _ops is None:
        from mcp_servers.code.operations import CodeOperations
        _ops = CodeOperations(str(PROJECT_ROOT))
    return _ops

#============================================
# Tool Schema Definitions (for SSEServer registration)
#============================================
LINT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File or directory to lint"},
        "tool": {"type": "string", "description": "Lint tool (ruff, pylint, flake8)"}
    },
    "required": ["path"]
}

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File or directory to format"},
        "tool": {"type": "string", "description": "Format tool (black, ruff)"},
        "check_only": {"type": "boolean", "description": "Only check, don't modify"}
    },
    "required": ["path"]
}

ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File or directory to analyze"}
    },
    "required": ["path"]
}

FIND_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "name_pattern": {"type": "string", "description": "Glob pattern for filename"},
        "path": {"type": "string", "description": "Directory to search"}
    },
    "required": ["name_pattern"]
}

LIST_FILES_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list"},
        "pattern": {"type": "string", "description": "Optional glob pattern"},
        "recursive": {"type": "boolean", "description": "Search recursively"}
    },
    "required": ["path"]
}

SEARCH_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Text/pattern to search"},
        "file_pattern": {"type": "string", "description": "Optional file pattern"},
        "case_sensitive": {"type": "boolean", "description": "Case-sensitive search"}
    },
    "required": ["query"]
}

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to read"},
        "max_size_mb": {"type": "number", "description": "Max file size in MB"}
    },
    "required": ["path"]
}

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write"},
        "content": {"type": "string", "description": "Content to write"},
        "backup": {"type": "boolean", "description": "Create backup first"},
        "create_dirs": {"type": "boolean", "description": "Create parent dirs"}
    },
    "required": ["path", "content"]
}

GET_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path"}
    },
    "required": ["path"]
}

EMPTY_SCHEMA = {"type": "object", "properties": {}}


#============================================
# SSE Transport Implementation (Gateway Mode)
#============================================
def run_sse_server(port: int):
    """Run using SSEServer for Gateway compatibility (ADR-066 v1.3)."""
    from mcp_servers.lib.sse_adaptor import SSEServer
    
    server = SSEServer("sanctuary_filesystem", version="1.0.0")
    ops = get_ops()
    
    # Wrapper functions for SSE (sync wrappers around the operations)
    def code_lint(path: str, tool: str = "ruff"):
        result = ops.lint(path, tool)
        output = [f"Linting {result['path']} with {result['tool']}:", ""]
        if result['issues_found']:
            output.append("❌ Issues found:")
            output.append(result['output'])
        else:
            output.append("✅ No issues found")
        return "\n".join(output)
    
    def code_format(path: str, tool: str = "black", check_only: bool = False):
        result = ops.format_code(path, tool, check_only)
        output = [f"Formatting {result['path']} with {result['tool']}:", ""]
        if check_only:
            if result['success']:
                output.append("✅ Code is properly formatted")
            else:
                output.append("❌ Code needs formatting")
                output.append(result['output'] or "")
        else:
            if result['modified']:
                output.append("✅ Code formatted successfully")
            else:
                output.append("❌ Formatting failed or no changes needed")
        return "\n".join(output)
    
    def code_analyze(path: str):
        result = ops.analyze(path)
        return f"Analyzing {result['path']}:\n\n{result['statistics']}"
    
    def code_check_tools():
        tools = ["ruff", "black", "pylint", "flake8", "mypy"]
        available = [f"✅ {t}" for t in tools if ops.check_tool_available(t)]
        unavailable = [f"❌ {t}" for t in tools if not ops.check_tool_available(t)]
        return "Available code tools:\n\n" + "\n".join(available + unavailable)
    
    def code_find_file(name_pattern: str, path: str = "."):
        matches = ops.find_file(name_pattern, path)
        if not matches:
            return f"No files found matching '{name_pattern}'"
        return f"Found {len(matches)} file(s):\n" + "\n".join(f"  {m}" for m in matches)
    
    def code_list_files(path: str, pattern: str = "*", recursive: bool = False):
        files = ops.list_files(path, pattern, recursive)
        if not files:
            return f"No files found in '{path}'"
        output = [f"Found {len(files)} file(s) in '{path}':"]
        for f in files:
            output.append(f"  {f['path']} ({f['size']/1024:.1f} KB)")
        return "\n".join(output)
    
    def code_search_content(query: str, file_pattern: str = "*.py", case_sensitive: bool = False):
        matches = ops.search_content(query, file_pattern, case_sensitive)
        if not matches:
            return f"No matches found for '{query}'"
        output = [f"Found {len(matches)} match(es):"]
        current_file = None
        for m in matches[:50]:
            if m['file'] != current_file:
                current_file = m['file']
                output.append(f"\n{current_file}:")
            output.append(f"  Line {m['line']}: {m['content']}")
        return "\n".join(output)
    
    def code_read(path: str, max_size_mb: float = 5.0):
        content = ops.read_file(path, max_size_mb)
        return f"Contents of {path}:\n{'='*60}\n{content}\n{'='*60}"
    
    def code_write(path: str, content: str, backup: bool = True, create_dirs: bool = True):
        result = ops.write_file(path, content, backup, create_dirs)
        output = [f"{'Created' if result['created'] else 'Updated'} file: {result['path']}"]
        output.append(f"Size: {result['size']} bytes")
        if result['backup']:
            output.append(f"Backup: {result['backup']}")
        return "\n".join(output)
    
    def code_get_info(path: str):
        import time
        info = ops.get_file_info(path)
        return f"""File info for {info['path']}:

  Language: {info['language']}
  Size: {info['size']} bytes ({info['size']/1024:.1f} KB)
  Lines: {info['lines'] if info['lines'] else 'N/A'}
  Modified: {time.ctime(info['modified'])}"""
    
    # Register tools
    server.register_tool("code_lint", code_lint, LINT_SCHEMA)
    server.register_tool("code_format", code_format, FORMAT_SCHEMA)
    server.register_tool("code_analyze", code_analyze, ANALYZE_SCHEMA)
    server.register_tool("code_check_tools", code_check_tools, EMPTY_SCHEMA)
    server.register_tool("code_find_file", code_find_file, FIND_FILE_SCHEMA)
    server.register_tool("code_list_files", code_list_files, LIST_FILES_SCHEMA)
    server.register_tool("code_search_content", code_search_content, SEARCH_CONTENT_SCHEMA)
    server.register_tool("code_read", code_read, READ_SCHEMA)
    server.register_tool("code_write", code_write, WRITE_SCHEMA)
    server.register_tool("code_get_info", code_get_info, GET_INFO_SCHEMA)
    
    logger.info(f"Starting SSEServer on port {port} (Gateway Mode)")
    server.run(port=port, transport="sse")


#============================================
# STDIO Transport Implementation (Local Mode)
#============================================
def run_stdio_server():
    """Run using FastMCP for local development (Claude Desktop)."""
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from mcp_servers.code.models import (
        CodeAnalysisRequest, CodeLintRequest, CodeFormatRequest,
        CodeFindFileRequest, CodeListFilesRequest, CodeSearchContentRequest,
        CodeReadRequest, CodeWriteRequest, CodeGetInfoRequest
    )
    
    mcp = FastMCP(
        "sanctuary_filesystem",
        instructions="""
        Sanctuary FileSystem / Code Cluster.
        - specialized in code analysis, linting, and formatting.
        - provides secure file operations and content search.
        """
    )
    
    @mcp.tool()
    async def code_lint(request: CodeLintRequest) -> str:
        """Run linting on a file or directory."""
        try:
            result = get_ops().lint(request.path, request.tool)
            output = [f"Linting {result['path']} with {result['tool']}:", ""]
            if result['issues_found']:
                output.append("❌ Issues found:")
                output.append(result['output'])
            else:
                output.append("✅ No issues found")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Linting failed: {str(e)}")
    
    @mcp.tool()
    async def code_format(request: CodeFormatRequest) -> str:
        """Format code in a file or directory."""
        try:
            result = get_ops().format_code(request.path, request.tool, request.check_only)
            output = [f"Formatting {result['path']} with {result['tool']}:", ""]
            if request.check_only:
                output.append("✅ Code is properly formatted" if result['success'] else "❌ Code needs formatting")
            else:
                output.append("✅ Code formatted successfully" if result['modified'] else "❌ No changes needed")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Formatting failed: {str(e)}")
    
    @mcp.tool()
    async def code_analyze(request: CodeAnalysisRequest) -> str:
        """Perform static analysis on code."""
        try:
            result = get_ops().analyze(request.path)
            return f"Analyzing {result['path']}:\n\n{result['statistics']}"
        except Exception as e:
            raise ToolError(f"Analysis failed: {str(e)}")
    
    @mcp.tool()
    async def code_check_tools() -> str:
        """Check which code quality tools are available."""
        tools = ["ruff", "black", "pylint", "flake8", "mypy"]
        ops = get_ops()
        available = [f"✅ {t}" for t in tools if ops.check_tool_available(t)]
        unavailable = [f"❌ {t}" for t in tools if not ops.check_tool_available(t)]
        return "Available code tools:\n\n" + "\n".join(available + unavailable)
    
    @mcp.tool()
    async def code_find_file(request: CodeFindFileRequest) -> str:
        """Find files by name or glob pattern."""
        try:
            matches = get_ops().find_file(request.name_pattern, request.path)
            if not matches:
                return f"No files found matching '{request.name_pattern}'"
            return f"Found {len(matches)} file(s):\n" + "\n".join(f"  {m}" for m in matches)
        except Exception as e:
            raise ToolError(f"Find failed: {str(e)}")
    
    @mcp.tool()
    async def code_list_files(request: CodeListFilesRequest) -> str:
        """List files in a directory with optional pattern."""
        try:
            files = get_ops().list_files(request.path, request.pattern, request.recursive)
            if not files:
                return f"No files found in '{request.path}'"
            output = [f"Found {len(files)} file(s):"]
            for f in files:
                output.append(f"  {f['path']} ({f['size']/1024:.1f} KB)")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"List failed: {str(e)}")
    
    @mcp.tool()
    async def code_search_content(request: CodeSearchContentRequest) -> str:
        """Search for text/patterns in code files."""
        try:
            matches = get_ops().search_content(request.query, request.file_pattern, request.case_sensitive)
            if not matches:
                return f"No matches found for '{request.query}'"
            output = [f"Found {len(matches)} match(es):"]
            for m in matches[:50]:
                output.append(f"  {m['file']}:{m['line']}: {m['content']}")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Search failed: {str(e)}")
    
    @mcp.tool()
    async def code_read(request: CodeReadRequest) -> str:
        """Read file contents."""
        try:
            content = get_ops().read_file(request.path, request.max_size_mb)
            return f"Contents of {request.path}:\n{'='*60}\n{content}\n{'='*60}"
        except Exception as e:
            raise ToolError(f"Read failed: {str(e)}")
    
    @mcp.tool()
    async def code_write(request: CodeWriteRequest) -> str:
        """Write/update file with automatic backup."""
        try:
            result = get_ops().write_file(request.path, request.content, request.backup, request.create_dirs)
            output = [f"{'Created' if result['created'] else 'Updated'} file: {result['path']}"]
            output.append(f"Size: {result['size']} bytes")
            if result['backup']:
                output.append(f"Backup: {result['backup']}")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Write failed: {str(e)}")
    
    @mcp.tool()
    async def code_get_info(request: CodeGetInfoRequest) -> str:
        """Get file metadata."""
        try:
            import time
            info = get_ops().get_file_info(request.path)
            return f"""File info for {info['path']}:

  Language: {info['language']}
  Size: {info['size']} bytes ({info['size']/1024:.1f} KB)
  Lines: {info['lines'] if info['lines'] else 'N/A'}
  Modified: {time.ctime(info['modified'])}"""
        except Exception as e:
            raise ToolError(f"Info retrieval failed: {str(e)}")
    
    logger.info("Starting FastMCP server (STDIO Mode)")
    mcp.run(transport="stdio")


#============================================
# Main Execution Entry Point (ADR-066 v1.3 Canonical Selector)
#============================================
def run_server():
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    if MCP_TRANSPORT not in {"stdio", "sse"}:
        logger.error(f"Invalid MCP_TRANSPORT: {MCP_TRANSPORT}. Must be 'stdio' or 'sse'.")
        sys.exit(1)
    
    if MCP_TRANSPORT == "sse":
        port = int(os.getenv("PORT", 8000))
        run_sse_server(port)
    else:
        run_stdio_server()


if __name__ == "__main__":
    run_server()
