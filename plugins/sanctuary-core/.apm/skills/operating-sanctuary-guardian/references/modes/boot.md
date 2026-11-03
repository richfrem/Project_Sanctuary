# Mode: Boot (`boot`)

Orient the agent session using the Protocol 128 Bootloader sequence and verify sovereign intelligence readiness.

## Execution Steps

1. Run the wakeup script:
```bash
python3 scripts/guardian_wakeup.py --mode HOLISTIC
```
2. Verify local sovereign model readiness:
   - Check persistent MLX server endpoint: `curl -s http://127.0.0.1:8080/v1/models`
   - Or audit cached weights and engine readiness: `python3 scripts/forge_llm.py status`
3. Inspect the generated `.agent/learning/guardian_boot_digest.md` and top priority incomplete mission.
4. Check for open tasks in `tasks/in-progress/` or backlog.
5. Verify environment status is GREEN before proceeding with work.
