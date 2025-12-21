
"""
Sanctuary FileSystem / Code Server
Domain: project_sanctuary.code / sanctuary_filesystem

Refactored to use SSEServer for Gateway integration (202 Accepted + Async SSE).
Also maps to "sanctuary_filesystem" in the topology.
"""
import os
import sys
import logging

# Import SSEServer
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from mcp_servers.lib.sse_adaptor import SSEServer
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib.sse_adaptor import SSEServer

from mcp_servers.code.code_ops import CodeOperations

# Initialize
server = SSEServer("sanctuary_filesystem") # Maps to code folder
app = server.app

# Operations
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
ops = CodeOperations(PROJECT_ROOT)

# Tool Wrappers (Async wrappers for synchronous ops)
async def code_lint(path: str, tool: str = "ruff") -> str:
    """Run linting on a file or directory."""
    try:
        result = ops.lint(path, tool)
        output = [f"Linting {result['path']} with {result['tool']}:", ""]
        if result['issues_found']:
            output.append("❌ Issues found:")
            output.append(result['output'])
        else:
            output.append("✅ No issues found")
        return "\n".join(output)
    except Exception as e:
        return f"Error linting '{path}': {str(e)}"

async def code_format(path: str, tool: str = "ruff", check_only: bool = False) -> str:
    """Format code in a file or directory."""
    try:
        result = ops.format_code(path, tool, check_only)
        output = [f"Formatting {result['path']} with {result['tool']}:", ""]
        if check_only:
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
        return f"Error formatting '{path}': {str(e)}"

async def code_analyze(path: str) -> str:
    """Perform static analysis on code."""
    try:
        result = ops.analyze(path)
        output = [f"Analyzing {result['path']}:", "", result['statistics']]
        return "\n".join(output)
    except Exception as e:
        return f"Error analyzing '{path}': {str(e)}"

async def code_check_tools() -> str:
    """Check which code quality tools are available."""
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

async def code_find_file(name_pattern: str, path: str = ".") -> str:
    """Find files by name or glob pattern."""
    try:
        matches = ops.find_file(name_pattern, path)
        if not matches:
            return f"No files found matching '{name_pattern}' in '{path}'"
        output = [f"Found {len(matches)} file(s) matching '{name_pattern}':", ""]
        for match in matches:
            output.append(f"  {match}")
        return "\n".join(output)
    except Exception as e:
        return f"Error finding files: {str(e)}"

async def code_list_files(path: str = ".", pattern: str = "*", recursive: bool = True) -> str:
    """List files in a directory with optional pattern."""
    try:
        files = ops.list_files(path, pattern, recursive)
        if not files:
            return f"No files found in '{path}' matching '{pattern}'"
        output = [f"Found {len(files)} file(s) in '{path}':", ""]
        for file_info in files:
            size_kb = file_info['size'] / 1024
            output.append(f"  {file_info['path']} ({size_kb:.1f} KB)")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing files: {str(e)}"

async def code_search_content(query: str, file_pattern: str = "*.py", case_sensitive: bool = False) -> str:
    """Search for text/patterns in code files."""
    try:
        matches = ops.search_content(query, file_pattern, case_sensitive)
        if not matches:
            return f"No matches found for '{query}' in files matching '{file_pattern}'"
        output = [f"Found {len(matches)} match(es) for '{query}':", ""]
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
        return f"Error searching content: {str(e)}"

async def code_read(path: str, max_size_mb: int = 10) -> str:
    """Read file contents."""
    try:
        content = ops.read_file(path, max_size_mb)
        output = [f"Contents of {path}:", "=" * 60, content, "=" * 60]
        return "\n".join(output)
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

async def code_write(path: str, content: str, backup: bool = True, create_dirs: bool = True) -> str:
    """Write/update file with automatic backup."""
    try:
        result = ops.write_file(path, content, backup, create_dirs)
        output = [f"{'Created' if result['created'] else 'Updated'} file: {result['path']}"]
        output.append(f"Size: {result['size']} bytes")
        if result['backup']:
            output.append(f"Backup: {result['backup']}")
        return "\n".join(output)
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"

async def code_get_info(path: str) -> str:
    """Get file metadata."""
    try:
        import time
        info = ops.get_file_info(path)
        output = [f"File info for {info['path']}:", ""]
        output.append(f"  Language: {info['language']}")
        output.append(f"  Size: {info['size']} bytes ({info['size']/1024:.1f} KB)")
        output.append(f"  Lines: {info['lines'] if info['lines'] else 'N/A'}")
        output.append(f"  Modified: {time.ctime(info['modified'])}")
        return "\n".join(output)
    except Exception as e:
        return f"Error getting file info for '{path}': {str(e)}"


# Register Tools
server.register_tool("code_lint", code_lint, {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "tool": {"type": "string", "default": "ruff"}
    },
    "required": ["path"]
})
server.register_tool("code_format", code_format, {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "tool": {"type": "string", "default": "ruff"},
        "check_only": {"type": "boolean", "default": False}
    },
    "required": ["path"]
})
server.register_tool("code_analyze", code_analyze, {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"]
})
server.register_tool("code_check_tools", code_check_tools, {"type": "object", "properties": {}})
server.register_tool("code_find_file", code_find_file, {
    "type": "object",
    "properties": {
        "name_pattern": {"type": "string"},
        "path": {"type": "string", "default": "."}
    },
    "required": ["name_pattern"]
})
server.register_tool("code_list_files", code_list_files, {
    "type": "object",
    "properties": {
        "path": {"type": "string", "default": "."},
        "pattern": {"type": "string", "default": "*"},
        "recursive": {"type": "boolean", "default": True}
    }
})
server.register_tool("code_search_content", code_search_content, {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "file_pattern": {"type": "string", "default": "*.py"},
        "case_sensitive": {"type": "boolean", "default": False}
    },
    "required": ["query"]
})
server.register_tool("code_read", code_read, {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "max_size_mb": {"type": "integer", "default": 10}
    },
    "required": ["path"]
})
server.register_tool("code_write", code_write, {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "backup": {"type": "boolean", "default": True},
        "create_dirs": {"type": "boolean", "default": True}
    },
    "required": ["path", "content"]
})
server.register_tool("code_get_info", code_get_info, {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"]
})

if __name__ == "__main__":
    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Legacy Mode)
    import os
    port_env = os.getenv("PORT")
    transport = "sse" if port_env else "stdio"
    port = int(port_env) if port_env else 8001
    
    server.run(port=port, transport=transport)
