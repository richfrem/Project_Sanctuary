#!/usr/bin/env python3
#============================================
# scripts/create_fastmcp_server.py
# Purpose: Scaffold a new MCP server following the 4-layer architecture (ADR 076) 
#          and the FastMCP standard (ADR 066).
# Role: Utility Script
#============================================

import os
import sys
import argparse
from pathlib import Path

# Template for server.py
SERVER_TEMPLATE = """#============================================
# mcp_servers/{server_name}/server.py
# Purpose: {server_title} MCP Server.
# Role: Interface Layer (ADR 076)
#============================================

from fastmcp import FastMCP
import os
import sys
from pathlib import Path

# Local Imports
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.{server_name}.operations import {class_name}Operations

# Setup Logging
logger = setup_mcp_logging("{server_name}")

# Initialize FastMCP
mcp = FastMCP("project_sanctuary.{server_name}")

# Configuration
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", default=None) or find_project_root()

# Initialize Operations
ops = {class_name}Operations(PROJECT_ROOT)

@mcp.tool()
async def {server_name}_example_tool(param: str) -> str:
    \"\"\"Example tool for {server_title}.\"\"\"
    try:
        # result = ops.do_something(param)
        return f"Hello {{param}} from {server_title}"
    except Exception as e:
        logger.error(f"Error in {server_name}_example_tool: {{e}}")
        return f"Error: {{str(e)}}"

if __name__ == "__main__":
    # Support dual-mode (SSE if PORT is set, else Stdio)
    port_env = get_env_variable("PORT", default=None)
    transport = "sse" if port_env else "stdio"
    port = int(port_env) if port_env else 8000
    
    mcp.run(port=port, transport=transport)
"""

# Template for operations.py
OPERATIONS_TEMPLATE = """#============================================
# mcp_servers/{server_name}/operations.py
# Role: Business Logic Layer (ADR 076)
#============================================

import logging
from pathlib import Path
from mcp_servers.{server_name}.validator import {class_name}Validator

logger = logging.getLogger("mcp.{{server_name}}")

class {class_name}Operations:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.validator = {class_name}Validator(self.project_root)

    def do_something(self, param: str):
        \"\"\"Core logic implementation.\"\"\"
        # self.validator.validate_param(param)
        return f"Processed {{param}}"
"""

# Template for validator.py
VALIDATOR_TEMPLATE = """#============================================
# mcp_servers/{server_name}/validator.py
# Role: Safety Layer (ADR 076)
#============================================

from pathlib import Path

class ValidationError(Exception):
    pass

class {class_name}Validator:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def validate_param(self, param: str):
        \"\"\"Apply Poka-Yoke safety checks (ADR 076).\"\"\"
        if not param:
            raise ValidationError("Parameter cannot be empty")
"""

# Template for models.py
MODELS_TEMPLATE = """#============================================
# mcp_servers/{server_name}/models.py
# Role: Data Layer (ADR 076)
#============================================

from pydantic import BaseModel
from typing import Optional, List

class {class_name}Data(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
"""

def main():
    parser = argparse.ArgumentParser(description="Scaffold a new FastMCP server.")
    parser.add_argument("name", help="Server name (e.g. 'my_server')")
    args = parser.parse_args()

    server_name = args.name.lower().replace("-", "_")
    class_name = "".join(x.capitalize() for x in server_name.split("_"))
    server_title = server_name.replace("_", " ").capitalize()

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    server_dir = project_root / "mcp_servers" / server_name

    if server_dir.exists():
        print(f"Error: Directory already exists: {server_dir}")
        sys.exit(1)

    print(f"Creating FastMCP server scaffold in {server_dir}...")
    server_dir.mkdir(parents=True, exist_ok=True)

    # Create files
    files = {
        "__init__.py": "",
        "server.py": SERVER_TEMPLATE.format(
            server_name=server_name, 
            server_title=server_title, 
            class_name=class_name
        ),
        "operations.py": OPERATIONS_TEMPLATE.format(
            server_name=server_name, 
            class_name=class_name
        ),
        "validator.py": VALIDATOR_TEMPLATE.format(
            class_name=class_name
        ),
        "models.py": MODELS_TEMPLATE.format(
            class_name=class_name
        )
    }

    for filename, content in files.items():
        file_path = server_dir / filename
        with open(file_path, "w") as f:
            f.write(content.strip() + "\n")
        print(f"  - Created {file_path.relative_to(project_root)}")

    print(f"\nSuccess! You can now start the server with:")
    print(f"  python3 -m mcp_servers.{server_name}.server")

if __name__ == "__main__":
    main()
