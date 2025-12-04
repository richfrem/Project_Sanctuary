from fastmcp import FastMCP
import os
from typing import Optional
from mcp_servers.code.code_ops import CodeOperations

# Initialize FastMCP
mcp = FastMCP("project_sanctuary.code")

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

# Initialize operations
ops = CodeOperations(PROJECT_ROOT)

@mcp.tool()
def code_lint(path: str, tool: str = "ruff") -> str:
    """
    Run linting on a file or directory.
    
    Args:
        path: Relative path to file or directory
        tool: Linting tool to use (ruff, pylint, flake8)
        
    Returns:
        Linting results with any issues found.
    """
    try:
        result = ops.lint(path, tool)
        
        output = [f"Linting {result['path']} with {result['tool']}:"]
        output.append("")
        
        if result['issues_found']:
            output.append("❌ Issues found:")
            output.append(result['output'])
        else:
            output.append("✅ No issues found")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error linting '{path}': {str(e)}"

@mcp.tool()
def code_format(path: str, tool: str = "ruff", check_only: bool = False) -> str:
    """
    Format code in a file or directory.
    
    Args:
        path: Relative path to file or directory
        tool: Formatting tool to use (ruff, black)
        check_only: If True, only check formatting without modifying files
        
    Returns:
        Formatting results.
    """
    try:
        result = ops.format_code(path, tool, check_only)
        
        output = [f"Formatting {result['path']} with {result['tool']}:"]
        output.append("")
        
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

@mcp.tool()
def code_analyze(path: str) -> str:
    """
    Perform static analysis on code.
    
    Args:
        path: Relative path to file or directory
        
    Returns:
        Analysis results with statistics.
    """
    try:
        result = ops.analyze(path)
        
        output = [f"Analyzing {result['path']}:"]
        output.append("")
        output.append(result['statistics'])
        
        return "\n".join(output)
    except Exception as e:
        return f"Error analyzing '{path}': {str(e)}"

@mcp.tool()
def code_check_tools() -> str:
    """
    Check which code quality tools are available.
    
    Returns:
        List of available tools.
    """
    tools = ["ruff", "black", "pylint", "flake8", "mypy"]
    available = []
    unavailable = []
    
    for tool in tools:
        if ops.check_tool_available(tool):
            available.append(f"✅ {tool}")
        else:
            unavailable.append(f"❌ {tool}")
    
    output = ["Available code tools:"]
    output.append("")
    output.extend(available)
    if unavailable:
        output.append("")
        output.append("Unavailable:")
        output.extend(unavailable)
    
    return "\n".join(output)

@mcp.tool()
def code_find_file(name_pattern: str, directory: str = ".") -> str:
    """
    Find files by name or glob pattern.
    
    Args:
        name_pattern: File name or glob pattern (e.g., "server.py", "*.py")
        directory: Directory to search in (default: project root)
        
    Returns:
        List of matching file paths.
    """
    try:
        matches = ops.find_file(name_pattern, directory)
        
        if not matches:
            return f"No files found matching '{name_pattern}' in '{directory}'"
        
        output = [f"Found {len(matches)} file(s) matching '{name_pattern}':"]
        output.append("")
        for match in matches:
            output.append(f"  {match}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error finding files: {str(e)}"

@mcp.tool()
def code_list_files(directory: str = ".", pattern: str = "*", recursive: bool = True) -> str:
    """
    List files in a directory with optional pattern.
    
    Args:
        directory: Directory to list (default: project root)
        pattern: Glob pattern for filtering (default: "*")
        recursive: If True, search recursively (default: True)
        
    Returns:
        List of files with metadata.
    """
    try:
        files = ops.list_files(directory, pattern, recursive)
        
        if not files:
            return f"No files found in '{directory}' matching '{pattern}'"
        
        output = [f"Found {len(files)} file(s) in '{directory}':"]
        output.append("")
        for file_info in files:
            size_kb = file_info['size'] / 1024
            output.append(f"  {file_info['path']} ({size_kb:.1f} KB)")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error listing files: {str(e)}"

@mcp.tool()
def code_search_content(query: str, file_pattern: str = "*.py", case_sensitive: bool = False) -> str:
    """
    Search for text/patterns in code files.
    
    Args:
        query: Text or pattern to search for
        file_pattern: File pattern to search in (default: "*.py")
        case_sensitive: If True, perform case-sensitive search (default: False)
        
    Returns:
        Search results with file paths, line numbers, and context.
    """
    try:
        matches = ops.search_content(query, file_pattern, case_sensitive)
        
        if not matches:
            return f"No matches found for '{query}' in files matching '{file_pattern}'"
        
        output = [f"Found {len(matches)} match(es) for '{query}':"]
        output.append("")
        
        current_file = None
        for match in matches[:50]:  # Limit to first 50 matches
            if match['file'] != current_file:
                current_file = match['file']
                output.append(f"\n{current_file}:")
            output.append(f"  Line {match['line']}: {match['content']}")
        
        if len(matches) > 50:
            output.append(f"\n... and {len(matches) - 50} more matches")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error searching content: {str(e)}"

@mcp.tool()
def code_read(path: str, max_size_mb: int = 10) -> str:
    """
    Read file contents.
    
    Args:
        path: Relative path to file
        max_size_mb: Maximum file size in MB (default: 10)
        
    Returns:
        File contents.
    """
    try:
        content = ops.read_file(path, max_size_mb)
        
        output = [f"Contents of {path}:"]
        output.append("=" * 60)
        output.append(content)
        output.append("=" * 60)
        
        return "\n".join(output)
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

@mcp.tool()
def code_write(path: str, content: str, backup: bool = True, create_dirs: bool = True) -> str:
    """
    Write/update file with automatic backup.
    
    Args:
        path: Relative path to file
        content: Content to write
        backup: If True, create backup before overwriting (default: True)
        create_dirs: If True, create parent directories if needed (default: True)
        
    Returns:
        Operation results.
    """
    try:
        result = ops.write_file(path, content, backup, create_dirs)
        
        output = [f"{'Created' if result['created'] else 'Updated'} file: {result['path']}"]
        output.append(f"Size: {result['size']} bytes")
        if result['backup']:
            output.append(f"Backup: {result['backup']}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"

@mcp.tool()
def code_get_info(path: str) -> str:
    """
    Get file metadata.
    
    Args:
        path: Relative path to file
        
    Returns:
        File metadata (size, modified date, line count, language).
    """
    try:
        import time
        info = ops.get_file_info(path)
        
        output = [f"File info for {info['path']}:"]
        output.append("")
        output.append(f"  Language: {info['language']}")
        output.append(f"  Size: {info['size']} bytes ({info['size']/1024:.1f} KB)")
        output.append(f"  Lines: {info['lines'] if info['lines'] else 'N/A'}")
        output.append(f"  Modified: {time.ctime(info['modified'])}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error getting file info for '{path}': {str(e)}"

if __name__ == "__main__":
    mcp.run()
