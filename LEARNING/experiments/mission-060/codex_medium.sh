#!/bin/sh
# Per-dispatch isolation only: disable hooks to preserve the empty neutral cwd.
# Native read-only sandbox and approval controls remain selected by cli-agent.
exec /opt/homebrew/bin/codex -c 'model_reasoning_effort="medium"' --disable hooks "$@"
