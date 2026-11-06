# MISSION-058 successor handoff

- **What happened:** a frozen read-only discovery pass over 20 public Moltbook posts (3 submolts, `sort=hot`, first page) produced 3 eligible candidates; one question was selected (physical cost of remembering/forgetting, from posts #9/#12 by `aporiathinks`) and investigated with independent sources and a script. Nothing was posted, voted or registered; no credentials were read.
- **Read first:** `findings.md` (conclusion and gaps), then `selection.md`, `investigation-protocol.md` (the question, hypothesis and failure conditions as stated before the investigation), `landauer_check.py` / `landauer-results.json`, `LEARNING/research/physics_of_forgetting/README.md`.
- **Conclusion:** the posts are partly right (Landauer floor is real) and wrong on three points (floor vs actual energy 10^10+; deletion is not erasure; flash retains data unpowered for years).
- **Do not redo:** the discovery pass (rerunning `discover.py` would give a different `hot` ranking and break reproducibility; use `discovery-log.json`); the SQLite and arithmetic checks (rerunning `landauer_check.py` is cheap and deterministic).
- **Next step:** the one next question in `findings.md`; first action = read the PRL (arXiv 2505.23087) efficiency figure.
- **Optional outward contribution:** a draft reply for human review is in `proposed-moltbook-contribution.md`. It has NOT been posted; posting is not authorized.
