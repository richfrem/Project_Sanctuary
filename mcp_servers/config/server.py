from fastmcp import FastMCP
import os
from typing import Optional, Dict, Any, Union
from mcp_servers.lib.config.config_ops import ConfigOperations

# Initialize FastMCP
mcp = FastMCP("project_sanctuary.config")

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
CONFIG_DIR = os.path.join(PROJECT_ROOT, ".agent/config")

# Initialize operations
ops = ConfigOperations(CONFIG_DIR)

@mcp.tool()
def config_list() -> str:
    """
    List all configuration files in the .agent/config directory.
    
    Returns:
        Formatted list of config files with sizes and modification times.
    """
    try:
        configs = ops.list_configs()
        if not configs:
            return "No configuration files found."
            
        output = [f"Found {len(configs)} configuration files:"]
        for c in configs:
            output.append(f"- {c['name']} ({c['size']} bytes, {c['modified']})")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing configs: {str(e)}"

@mcp.tool()
def config_read(filename: str) -> str:
    """
    Read a configuration file.
    
    Args:
        filename: Name of the config file (e.g., 'mcp_config.json')
        
    Returns:
        Content of the configuration file.
    """
    try:
        content = ops.read_config(filename)
        if isinstance(content, (dict, list)):
            import json
            return json.dumps(content, indent=2)
        return str(content)
    except Exception as e:
        return f"Error reading config '{filename}': {str(e)}"

@mcp.tool()
def config_write(filename: str, content: str) -> str:
    """
    Write a configuration file.
    
    Args:
        filename: Name of the config file
        content: Content to write (string or JSON string)
        
    Returns:
        Status message with path to written file.
    """
    try:
        # Try to parse content as JSON if file extension implies it
        import json
        if filename.endswith('.json'):
            try:
                data = json.loads(content)
                path = ops.write_config(filename, data)
            except json.JSONDecodeError:
                # Write as raw string if not valid JSON
                path = ops.write_config(filename, content)
        else:
            path = ops.write_config(filename, content)
            
        return f"Successfully wrote config to {path}"
    except Exception as e:
        return f"Error writing config '{filename}': {str(e)}"

@mcp.tool()
def config_delete(filename: str) -> str:
    """
    Delete a configuration file.
    
    Args:
        filename: Name of the config file to delete
        
    Returns:
        Status message.
    """
    try:
        ops.delete_config(filename)
        return f"Successfully deleted config '{filename}'"
    except Exception as e:
        return f"Error deleting config '{filename}': {str(e)}"

if __name__ == "__main__":
    mcp.run()
