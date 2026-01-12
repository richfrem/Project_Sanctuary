import os
import json
import sys
from pathlib import Path

def generate_config():
    project_root = Path(os.getcwd()).resolve()
    mcp_servers_dir = project_root / "mcp_servers"
    config_path = project_root / ".agent" / "mcp_config.json"
    
    if not mcp_servers_dir.exists():
        print(f"Error: {mcp_servers_dir} does not exist.")
        sys.exit(1)

    servers_config = {}

    # Walk through mcp_servers directory
    for root, dirs, files in os.walk(mcp_servers_dir):
        if "server.py" in files:
            server_path = Path(root)
            relative_path = server_path.relative_to(project_root)
            
            # Convert path to python module format
            # e.g. mcp_servers/system/git_workflow -> mcp_servers.system.git_workflow.server
            module_path = str(relative_path).replace(os.sep, ".") + ".server"
            
            # Determine server name (folder name)
            server_name = server_path.name
            
            # Determine display name (Title Case)
            display_name = server_name.replace("_", " ").title() + " MCP"
            
            print(f"Found server: {server_name} at {relative_path}")
            
            servers_config[server_name] = {
                "displayName": display_name,
                "command": "python",
                "args": ["-m", module_path],
                "env": {
                    "PROJECT_ROOT": str(project_root),
                    "PYTHONPATH": str(project_root)
                }
            }

    # Add specific configuration for 'git' server as requested
    servers_config["git"] = {
        "displayName": "Git MCP",
        "command": "python3",
        "args": ["-m", "mcp_servers.git.server"],
        "env": {
            "PROJECT_ROOT": str(project_root),
            "PYTHONPATH": str(project_root)
        }
    }

    full_config = {
        "mcpServers": servers_config
    }

    # Ensure .agent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)
        
    print(f"\nGenerated config at: {config_path}")
    print(json.dumps(full_config, indent=2))

if __name__ == "__main__":
    generate_config()
