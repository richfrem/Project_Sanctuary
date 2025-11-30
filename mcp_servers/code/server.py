from fastmcp import FastMCP
import os
from typing import Optional
from mcp_servers.lib.code.code_ops import CodeOperations

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

if __name__ == "__main__":
    mcp.run()
