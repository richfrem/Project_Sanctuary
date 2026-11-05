# Living Chronicle - Entry 377

**Title:** Reinstalling the plugin exposed the held-out evaluation generator to the local-agent sandbox; fixed by hiding it by name
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

After apm install at the project level (needed because the installed skill copies were stale), a local-agent test failed: the MISSION-022 sandbox hid agent_eval_tasks.py (the held-out task generator) only by source path prefix, but APM copies whole skill folders including tests into .claude/skills, .agents/skills and apm_modules/_local, so the generator and the guardian tests that import it became findable and readable in three more places. The previous lockfile shows only test_coordination.py had ever been deployed from that tests folder, so the 3 November held-out runs (MISSION-022, FALSIFIED) were not exposed; the exposure starts with this reinstall and would affect any future held-out run. Fix with tests first: the sandbox now denies by file name (agent_eval_tasks.py, heldout_results*.json) and by folder glob (the guardian skill tests folder in any copy) by default; other skills tests stay visible; the old assertion now checks that no hit path points into a denied file. Also in this change: the wakeup installed-skills drift check no longer reports APM link rewriting as drift (a false positive found right after the install), apm install generated an untracked .github/hooks folder for the copilot target (now gitignored) and refreshed apm.lock.yaml (no machine paths or secrets), and the reinstall registered the SessionStart and PreCompact hooks in .claude/settings.json, .codex/hooks.json and .github/hooks, which activates the MISSION-030 snapshot hook in this project. Lesson: a deny-list written as path prefixes of the source tree silently stops protecting once the tree is copied.
