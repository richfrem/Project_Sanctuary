# Acceptance Criteria: Operating Sanctuary Guardian

## Functional Acceptance Criteria
1. `guardian_wakeup.py --mode HOLISTIC` generates `.agent/learning/guardian_boot_digest.md` and outputs JSON success with GREEN health.
2. `capture_snapshot.py --type audit` executes without crashing.
3. `learning_debrief.py --hours 24` successfully extracts recent commit and file modifications.
4. `persist_soul.py` executes without unhandled exceptions.
5. `forge_llm.py verify` executes an offline Protocol 128 verification test returning valid JSON with `status: "PASS"`.
6. `forge_llm.py server status` and `mlx_server_manager.py status` report accurate process PID, port, and RSS memory telemetry.
7. All lifecycle operations execute interactively in the main conversation thread.
