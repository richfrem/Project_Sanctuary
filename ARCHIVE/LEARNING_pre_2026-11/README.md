# LEARNING archive, 2026-11-05

Retired material moved out of `LEARNING/` so the live tree holds only what agents should read. Nothing was deleted: files were moved with `git mv`, so history is intact, and the content also remains in the sealed Hugging Face soul dataset (`huggingface/data/soul_traces.jsonl`, records keep their original `source_file: LEARNING/...` path; use the map below to find the file).

Triage: every `topics/` dossier was read together with the evidence of who cites it; the rule was "archive only if retired-era or superseded, not cited, and not recent". A first name-based pass was wrong for several folders (see `LEARNING/research/experiments/learning-restructure/README.md`).

| Moved item (path under this folder) | Was | Why |
|:---|:---|:---|
| `security/` | `LEARNING/security/` | August 2025 Chimera, Hydra and Prometheus briefings; nothing references them. Current doctrine is in `LEARNING/concepts/human_gate_zero_trust.md`. |
| `associative_architecture_brief.md` | `LEARNING/` | January 2026 theory brief; nothing references it. |
| `calibration_log.json` | `LEARNING/` | December 2025 metrics; nothing references them. |
| `topics/scholarly_discourse` | `LEARNING/topics/` | Moltbook external-engagement drafts and audit packets; narrated in chronicle entries 340-342. |
| `topics/openclaw_moltbook_ecosystem` | `LEARNING/topics/` | Same episode (ecosystem analysis). |
| `topics/safe_agent_zero` | `LEARNING/topics/` | Same episode (Sanctum runtime proposal). |
| `topics/documentation_link_remediation` | `LEARNING/topics/` | Red-team questions superseded by the `checking-links` skill. |
| `topics/protocol_128_remediation` | `LEARNING/topics/` | V1 audit remediation report that names retired tooling (`cortex_capture_snapshot`, `mcp_servers/`). |
| `topics/liquid_neural_networks` | `LEARNING/topics/` | Unfinished December 2025 research with no link to the current architecture. |
| `topics/forge_v5_evolution` | `LEARNING/topics/` | Session report for ADR 075 on retired `forge/` and `mcp_servers` code; the ADR is in `ARCHIVE/ADRs/`. |
| `topics/knowledge_preservation_red_team` | `LEARNING/topics/` | Red-team drafting history of ADR 079 (Hugging Face soul persistence); the ADR is in `ARCHIVE/ADRs/`. |

## Restore one
`git mv ARCHIVE/LEARNING_pre_2026-11/topics/<name> LEARNING/topics/<name>`, add a row to `LEARNING/topics/README.md` (the tests require it), and update `LEARNING/INDEX.md` if it is a top-level folder.

## Note on git ignore
`ARCHIVE/` matches an ignore rule in `.gitignore` (and the existing `!ARCHIVE/council_orchestrator/` exception does not work). Moved files stay tracked; a NEW file added here needs `git add -f`.
