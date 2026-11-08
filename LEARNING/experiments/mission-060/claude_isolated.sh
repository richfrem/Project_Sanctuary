#!/bin/sh
# Native safe mode suppresses custom hooks/context without changing OAuth auth.
exec /Users/richardfremmerlid/.local/bin/claude --safe-mode --tools '' --strict-mcp-config "$@"
