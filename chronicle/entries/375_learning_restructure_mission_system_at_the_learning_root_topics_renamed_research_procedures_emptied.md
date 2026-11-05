# Living Chronicle - Entry 375

**Title:** LEARNING restructure: mission system at the LEARNING root, topics renamed research, procedures emptied
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

At the human's direction the remaining moves were done together and the references fixed once (a branch may be broken in the middle; main never is): research/experiments, index.jsonl and inbox.jsonl moved to the LEARNING root (missions/ and backlog.md had already moved), the contents of topics/ moved into the emptied research/ and the empty topics/ removed. 110 git mv renames. Tests first (3 red), then a fixed-order rewrite across 84 files plus a hand edit of the backlog manager; history (chronicle, dataset, archive, events log, experiment records and recorded data) keeps old paths. A guard now reports any mission-system item left inside research/. Things a blind rewrite would have broken: four experiment scripts that find the repo root by counting folders (parents[4] to parents[3]), the .gitignore entry hiding a local drq_repo clone, and restore recipes in the archive README. procedures/ was reviewed on request: git/ deleted, orchestration/ and architecture/ archived (superseded by the flight board, cli-agent and orchestration skills); the folder stays as the home of the procedural-memory mechanism because open missions 042, 047 and 049 build on it and five core docs advertise it, and only two outcome events were ever recorded. philosophy/ and concepts/ stay separate (immutable source doctrine versus script-written, decay-audited canon). Process lesson: one-folder-per-PR meant rewriting the same documents four times; one branch with all moves then one reference pass was cleaner.
