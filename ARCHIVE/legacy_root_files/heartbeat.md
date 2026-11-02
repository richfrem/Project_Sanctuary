# Heartbeat Tasks

<!-- Scheduled tasks for the /loop scheduler (Claude Code CLI). -->
<!-- Source: https://github.com/richfrem/agent-plugins-skills/tree/main/plugins/agent-agentic-os -->
<!-- Start hourly with: /loop "Read heartbeat.md and execute the items under Every Hour" --interval 1h -->

## Every Hour
- Check `context/memory/hook-errors.log` for any hook failures
- Read `context/events.jsonl` (last 20 lines) for friction signals
- Run the `os-learning-loop` agent if friction count > 3 in the last hour
- Write a brief status update to `context/status.md`

## Every 24 Hours
- Run `session-memory-manager` to promote durable facts from today's log to `context/memory.md`
- Check `context/memory.md` line count (500-line limit - archive if exceeded)
- Run `rlm-distill-agent` on any new .py/.md files added today
- Write session summary to `context/memory/YYYY-MM-DD.md`

## On Session Start
- Load `START_HERE.md`
- Read `context/status.md` for open blockers
- Emit: `python3 context/kernel.py emit_event --agent system --type agent_start --action session_start --status success`

## Continuous Improvement Metrics (os-learning-loop feed)

The `os-learning-loop` agent reads these sources to generate improvement proposals:
- `context/events.jsonl` - all kernel events (friction = error events or failed actions)
- `context/memory/hook-errors.log` - hook execution failures (primary friction signal)
- `context/memory/YYYY-MM-DD.md` - session logs (lessons learned, repeated failures)

When friction count threshold is exceeded, `os-learning-loop`:
1. Acquires lock: `python3 context/kernel.py acquire_lock kernel`
2. Reads friction sources above
3. Proposes skill/SKILL.md update (requires `skill-improvement-eval` PASS gate)
4. Validates proposed change via `skill-improvement-eval` agent
5. If PASS: writes to disk with Loop Recovery Snapshot backup
6. Releases lock: `python3 context/kernel.py release_lock kernel`
