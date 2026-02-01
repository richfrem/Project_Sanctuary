#!/bin/bash
# Shim for /workflow-bundle
# Aligned with ADR-036: Thick Python / Thin Shim architecture
# This script directs the user to the Context Bundler tool (bundle.py)

# Ensure checking for help flag to avoid raw python tracebacks if possible
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    python3 tools/retrieve/bundler/bundle.py --help
    exit 0
fi

# Execute the python tool directly
exec python3 tools/retrieve/bundler/bundle.py "$@"
