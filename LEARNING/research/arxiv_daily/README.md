---
id: res-arxiv-daily
title: "arXiv Daily Digest (cs.AI)"
status: KEEP
tags: ["arxiv", "cs.AI", "daily-digest", "sanctuary-bot", "automation"]
related_missions: ["MISSION-001", "MISSION-069"]
source_reading_depth: "abstract_inspection"
last_substantive_review: "2026-11-09"
---

# arXiv Daily Digest (cs.AI)

**Status:** KEEP (live; updated by the `arxiv-daily` automation)
**Produced by:** Sanctuary Bot automation `.agent/automations/arxiv-daily.json` (skill `operating-sanctuary-bot`).

## What is here
| Path | Contents |
|---|---|
| `latest.md` | The newest digest |
| `digests/YYYY-MM-DD.md` | One digest per day (archive) |
| `incoming.json` | The latest raw fetch: id, title and abstract of the 5 newest cs.AI submissions |
| `pipeline/fetch_arxiv.py` | Fetch from the arXiv API (stdlib only) |
| `pipeline/make_digest.py` | Build the digest (deterministic; no model) |
| `analysis/YYYY-MM-DD.md` | Daily analysis: relevance to existing missions, project impact, improvement ideas, mission seeds (Claude Haiku, abstracts only) |
| `seeds.jsonl` | Candidate mission seeds from the analyses, all `unreviewed`; promote to `LEARNING/inbox.jsonl` only through review |
| `pipeline/save_analysis.py` | Saves the analysis and appends seeds; rejects output missing required sections |
| `pipeline/test_pipeline.py` | Output checks, used as the automation's verify command |

## How it flows
1. Daily at 07:00 the bot runs four steps in its runtime worktree `.worktrees/bot-arxiv-daily` (branch `bot/arxiv-daily`): fetch; digest (only if the papers changed; otherwise the run ends with nothing to commit); analyze (Claude Haiku, isolated: no tools, inputs passed as data, treated as untrusted); save (writes the analysis and seeds, commits).
2. The bot never pushes or merges. An agent or the human opens a PR from `bot/arxiv-daily` to `main` to bring the digests into the knowledge base and the soul dataset.

## Reading depth and limits
- Summaries are the **first two sentences of each abstract**, not written summaries; nothing here is a reading of the full paper.
- "Candidate Mission Ideas" are templated from titles. They are unreviewed seeds, not proposals; register a mission only after reading the paper.
- The analysis sees only the digest and Sanctuary's indexes, not full papers or code; its mission links and seeds are suggestions to review.
- arXiv text is untrusted input to a model: the analyze step has no tools or file access, and its output reaches main only through PR review.
- Only the 5 newest submissions per day.
