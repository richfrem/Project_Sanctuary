# Living Chronicle - Entry 364

**Title:** MISSION-030 corroborated: PreCompact snapshot hook, first measured Codex record, codex token split fix
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

A deterministic PreCompact hook (pre_compact_snapshot.py) writes a provisional snapshot of uncommitted files, last assistant conclusions and friction lines, with secrets redacted, before context compaction; SessionStart adds a pointer to it after a compaction. Median 61 ms, max 67 ms on a 20.6 MB transcript (ceiling 2500 ms). It fired on a real manual /compact in a throwaway Claude Code session started with a temporary --settings file; the auto trigger is unverified, and the hook only runs where the plugin hooks are installed. Codex (gpt-6-luna, isolated dispatch, 28 s) wrote the unit tests from a spec: 11 passed on the first run after I trimmed log lines my own extraction had copied in and changed one line. That dispatch is the first measured Codex record for MISSION-035: 27,943 tokens, 26,893 in / 1,050 out, 0.0032 USD. Recording it exposed two defects, fixed: codex_tokens_between returned only a total so output was priced at the input rate, and my first finish call used the wrong lease key. Found along the way: the Claude CLI warns that this sign-in ends between 2026-11-10 and 2026-11-13. Files: .apm/hooks/pre_compact_snapshot.py, hooks.json, session_start_wakeup.py, hooks/tests/, references/precompact-hook.md.
