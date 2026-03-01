---
description: Protocol 128 Phase I - The Learning Scout (Debrief & Orientation)
---
# Workflow: Scout

> **CLI Commands**:
> - Debrief: `python3 plugins/sanctuary-guardian/scripts/learning_debrief.py --hours 24`
> - Guardian: `python3 plugins/sanctuary-guardian/scripts/guardian_wakeup.py --mode TELEMETRY`
> **Key Output**: Path to `learning_package_snapshot.md`

## Steps

1. **Wakeup & Debrief**:
   // turbo
   python3 plugins/sanctuary-guardian/scripts/learning_debrief.py --hours 24

2. **Read Truth Anchor**:
   The output of the previous command provided a path to `learning_package_snapshot.md`.
   You MUST read this file now using `view_file`.

3. **Guardian Check**:
   // turbo
   python3 plugins/sanctuary-guardian/scripts/guardian_wakeup.py --mode TELEMETRY

