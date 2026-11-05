# Living Chronicle - Entry 376

**Title:** Stale-reference sweep: installed skill copies were stale; navigation docs, context and .agent corrected; drift check added
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

Asked whether plugin references match the new locations and whether context/, .agent/, INSTALL.md, START_HERE.md, AGENTS.md and architecture.md are current. Source plugins/ was clean; the real gap was the installed copies: APM project installs are real copies (.claude/skills, .agents/skills, apm_modules/_local), dated 1-2 November and 13/21/9/2/3/1 files behind per skill, so agents running a skill from there used old scripts with old paths. Fix: apm install at the repo root (dry run touches only ./plugins/sanctuary-core) plus an installed-skills line in the session wakeup that warns on drift. context/memory.md and soul.md had wrong statements (a backlog.json that never existed, concepts/topics, a wrong promotion target, 6 skills instead of 7, a dead path); two durable entries were added. .agent/ had a README listing an archived manifest and a wiki path, a rule file pointing at a renamed skill, and a Feb 2026 friction log about a retired tool (archived). AGENTS.md gained rule 9 (leased worktrees, untracked board, merge_prep, immediate resync, update plugins with every path move) and a reinstall sentence in rule 4; architecture.md, INSTALL.md and README.md were corrected; START_HERE.md was current. Guards: navigation docs must have no dangling repo path, must not describe topics as current, must say to reinstall. Found but not changed: 31 plugin-doc references to retired trees (many are skill-relative paths, not errors), 13 in canonical protocols (human-gated), INSTALL.md wanting APM 0.35 while the machine runs 0.33, and a January cortex.json from a retired subsystem.
