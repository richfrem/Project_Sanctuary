---
description: Protocol 128 Phase I - The Learning Scout (Debrief & Orientation)
---
# Workflow: Scout

> **CLI Commands**:
> - Debrief: `python3 tools/cli.py debrief --hours 24`
> - Guardian: `python3 tools/cli.py guardian --mode TELEMETRY`
> **Key Output**: Path to `learning_package_snapshot.md`

## Steps

1. **Wakeup & Debrief**:
   // turbo
   python3 tools/cli.py debrief --hours 24

2. **Read Truth Anchor**:
   The output of the previous command provided a path to `learning_package_snapshot.md`.
   You MUST read this file now using `view_file`.

3. **Guardian Check**:
   // turbo
   python3 tools/cli.py guardian --mode TELEMETRY

