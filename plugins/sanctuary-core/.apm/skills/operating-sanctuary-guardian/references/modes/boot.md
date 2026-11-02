# Mode: Boot (`boot`)

Orient the agent session using the Protocol 128 Bootloader sequence.

## Execution Steps

1. Run the wakeup script:
```bash
python3 scripts/guardian_wakeup.py --mode HOLISTIC
```
2. Inspect the generated `.agent/learning/guardian_boot_digest.md`.
3. Check for open tasks in `tasks/in-progress/` or backlog.
4. Verify environment status is GREEN before proceeding with work.
