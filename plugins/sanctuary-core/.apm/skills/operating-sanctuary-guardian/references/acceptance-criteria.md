# Acceptance Criteria: Operating Sanctuary Guardian

## Functional Acceptance Criteria
1. `guardian_wakeup.py --mode HOLISTIC` generates `.agent/learning/guardian_boot_digest.md` and outputs JSON success.
2. `capture_snapshot.py --type audit` executes without crashing.
3. `learning_debrief.py --hours 24` successfully extracts recent commit and file modifications.
4. `persist_soul.py` executes without unhandled exceptions.
5. All lifecycle operations execute interactively in the main conversation thread.
