---
description: Protocol 128 Phase I - The Learning Scout (Debrief & Orientation)
---
# Workflow: Scout

1. **Wakeup & Debrief**:
   // turbo
   python3 scripts/cortex_cli.py debrief --hours 24

2. **Read Truth Anchor**:
   The output of the previous command provided a path to `learning_package_snapshot.md`.
   You MUST read this file now using `view_file`.

3. **Guardian Check**:
   // turbo
   python3 scripts/cortex_cli.py guardian --mode TELEMETRY
