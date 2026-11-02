# Acceptance Criteria: Managing Sovereign Memory

## Functional Acceptance Criteria
1. `disclose.py --level summary` extracts frontmatter and summary in < 15ms.
2. `record_insight.py` writes insights to `context/memory.md` and `wiki/concepts/` without crashing.
3. `audit_decay.py --check` identifies `CONFIRMED` entries older than 30 days.
4. `query_graph.py` maps outbound links and inbound backlinks across markdown files in < 20ms.
5. All scripts operate with zero third-party dependencies using standard Python 3.
