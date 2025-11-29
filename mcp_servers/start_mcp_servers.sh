#!/bin/bash

# Start MCP Servers for Project Sanctuary
# Usage: ./start_mcp_servers.sh

echo "Starting Project Sanctuary MCP Servers..."

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected. It is recommended to run this inside a venv."
fi

# Define server paths
CORTEX_SERVER="cognitive/cortex/server.py"
CHRONICLE_SERVER="chronicle/server.py"
PROTOCOL_SERVER="protocol/server.py"
ORCHESTRATOR_SERVER="orchestrator/server.py"

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Server file not found: $1"
        exit 1
    fi
}

check_file "$CORTEX_SERVER"
check_file "$CHRONICLE_SERVER"
check_file "$PROTOCOL_SERVER"
check_file "$ORCHESTRATOR_SERVER"

echo "All server files located."
echo ""
echo "To run a specific server, use:"
echo "  python $CORTEX_SERVER"
echo "  python $CHRONICLE_SERVER"
echo "  python $PROTOCOL_SERVER"
echo "  python $ORCHESTRATOR_SERVER"
echo ""
echo "Note: These servers are designed to be run by an MCP Client (like Claude Desktop)."
echo "Please configure your client to point to these scripts."
