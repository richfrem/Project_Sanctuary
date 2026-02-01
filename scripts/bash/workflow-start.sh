#!/bin/bash
# workflow-start.sh - Pre-Flight Check & Spec Initializer
# Enforces "One Spec = One Branch" via Py Orchestrator (ADR-0030)

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
CLI_PATH="$PROJECT_ROOT/tools/cli.py"

# Args
WORKFLOW_NAME="$1"
TARGET_ID="$2"
TARGET_TYPE="${3:-generic}"

if [ -z "$WORKFLOW_NAME" ]; then
    echo "Usage: ./workflow-start.sh [WorkflowName] [TargetID] [Type]"
    exit 1
fi

# Handover to Python (ADR-0030: The Orchestrator handles all logic)
exec python3 "$CLI_PATH" workflow start --name "$WORKFLOW_NAME" --target "$TARGET_ID" --type "$TARGET_TYPE"
