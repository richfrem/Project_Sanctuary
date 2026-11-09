# Living Chronicle - Entry 390

**Title:** Sanctuary Bot: local, approved, scheduled automations
**Date:** 2026-11-08
**Author:** claude-code
**Status:** COMPLETE
**Classification:** architecture

---

Sanctuary gained its own background bot, built on MISSION-001's zero-token daemon. The two interim skills (running-sanctuary-daemon, orchestrating-daemon-campaigns) were merged into one, operating-sanctuary-bot, with one interface (bot.py). Design, after comparing the industry (xAI Grok Automations, OpenAI ChatGPT scheduled tasks, Google Gemini Scheduled Actions, Claude Code routines; branded agents Grok Bot, Dots, Duet, Gemini Agent, Meta Muse): the product is the bot; the scheduled unit is an automation; each firing is a run; a run is one step or several (job or campaign). Choices: the scheduler lives inside the daemon in plain Python, so it is identical on macOS, Linux and Windows (only start-at-login is OS-specific); definitions are tracked in .agent/automations and reviewed by PR; approval is bound to a SHA-256 of the definition, so any edit stops it until re-approved; every run goes through the conductor (triage, git-trail and previous-run prompts, commit/allowed-path/verify checks, worktree-only revert and retry, breakers); no overlap, catch-up once or skip, auto-pause after repeated failed runs, history, inbox and an optional notify command. Deliberately unlike cloud bots: it runs only on this machine, has no identity, email or payments, and acts only through reviewable git commits. Bootstrap never starts it. Evidence: 34 tests (11 bot incl. a real end-to-end scheduled run, 14 conductor, 9 daemon), skill audits PASS, installed and preserved. Known gaps, registered as MISSION-069 (not executed): any process can run bot.py approve --by <anyone>, and one automation's failures can trip the daemon's global breaker.
