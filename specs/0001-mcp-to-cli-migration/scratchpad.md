# Scratchpad

**Spec**: 0001-mcp-to-cli-migration
**Created**: 2026-01-31

> **Purpose**: Capture ideas as they come up, even if out of sequence.
> At the end of the spec, process these into the appropriate places.

---

## Spec-Related Ideas
<!-- Clarifications, scope questions, requirements, "what are we actually building?" -->

- [ ] Clarify: Should we keep MCP servers as optional fallback or fully deprecate?
- [ ] Query user: Priority order for migrating MCP servers (which first?)
- [ ] Consider: What about Council MCP's multi-agent deliberation - can CLI replicate?

---

## Plan-Related Ideas
<!-- Architecture, design decisions, alternatives, "how should we build it?" -->

- [ ] Pattern: Each MCP operation maps to `python tools/cli.py <domain> <operation>`
- [ ] Consider: Use `click` or `argparse` for CLI (currently argparse in cli.py)
- [ ] Architecture: Keep `mcp_servers/lib/` operations.py files - they contain business logic

### Tool Overlap Analysis (CRITICAL)

Need to analyze and consolidate these overlapping implementations:

| Domain | New (tools/) | Old (mcp_servers/) | Action |
|--------|------------|-------------------|--------|
| **RLM Cache** | `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py` | `mcp_servers/learning/operations.py` | Compare APIs |
| **Vector Query** | `tools/retrieve/vector/query.py` | `mcp_servers/rag_cortex/operations.py` | Compare APIs |
| **Vector Ingest** | `tools/codify/vector/ingest.py` | `mcp_servers/rag_cortex/ingest_code_shim.py` | Check if duplicate |
| **RLM Distill** | `plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py` | `mcp_servers/learning/operations.py` (rlm ops?) | May be different |

**Question**: Which is source of truth? Should `tools/` import from `mcp_servers/lib/` or vice versa?

**Recommendation**: Make `mcp_servers/lib/` the operations layer, `tools/` just CLIs that call it.

### RLM / CAG Cache Analysis (NEEDS DEEP DIVE)

**Context from User:**
- New RLM cache tools (`plugins/rlm-factory/skills/rlm-curator/scripts/`) can write to multiple caches
- Old MCP cache (`cortex-cache-*`) might have different model/value
- These are NOT the same thing and may both be needed

**Files to Compare:**

| New (tools/) | Old (mcp_servers/) | Purpose |
|-------------|-------------------|---------|
| `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py` | `mcp_servers/learning/operations.py` | Tool discovery |
| `plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py` | ? | Semantic summary creation |
| `tools/retrieve/vector/query.py` | `mcp_servers/rag_cortex/operations.py` | Vector search |
| N/A | `cortex-cache-*` (CAG) | Cached Augmented Generation |

**Key Question**: What does CAG cache store vs RLM cache?
- **RLM Cache**: Tool/file summaries for discovery (`rlm_tool_cache.json`, `rlm_summary_cache.json`)
- **CAG Cache**: Pre-computed query/answer pairs for faster RAG responses

**Verdict**: These are **different systems** - both may have value. Defer detailed comparison.

### CLI Consolidation Analysis (Reference)

When extending `tools/cli.py`, review these existing CLIs for patterns:

| CLI | Location | Domains | Notes |
|-----|----------|---------|-------|
| `cortex_cli.py` | `scripts/cortex_cli.py` | RAG, Evolution, RLM, Guardian, Learning | 560 lines, imports `mcp_servers.rag_cortex`, `mcp_servers.evolution` |
| `domain_cli.py` | `scripts/domain_cli.py` | Chronicle, Task, ADR, Protocol | 133 lines, imports `mcp_servers.{domain}.operations` |
| `cli.py` | `tools/cli.py` | Workflow orchestration | Entry point - extend this |

**Decision**: Consolidate into `tools/cli.py` OR keep separate entry points and register all in RLM cache.

### Pre-existing ADR Duplicate (Out of Scope)

Found during migration: `ADRs/092_*` has two files:
- `092_RLM_Context_Synthesis.md`
- `092_mcp_architecture_evolution_15_servers.md`

**Action**: Note for future cleanup, not part of this spec.

---

## Task-Related Ideas
<!-- Things to add to tasks.md, step refinements, "what specific work needs doing?" -->

- [ ] Fix `query_cache.py` BrokenPipeError (piping output to head)
- [ ] Fix `next_number.py` SyntaxWarning (escape sequence)
- [ ] Create mapping document `docs/operations/mcp/mcp_to_cli_mapping.md`
- [ ] Update sanctuary-start.md to use `--type` flag for next_number.py

---

## Out-of-Scope (Future Backlog)
<!-- Ideas for /create-task after this spec closes, "good idea but not now" -->

- [ ] Deprecate MCP Gateway cluster architecture
- [ ] Remove Podman dependency entirely
- [ ] Port soul persistence to CLI (currently uses MCP)

---

## Processing Checklist (End of Spec)
- [ ] Reviewed all items above
- [ ] Spec-related items incorporated into `spec.md` or discussed
- [ ] Plan-related items incorporated into `plan.md` or discussed
- [ ] Task-related items added to `tasks.md`
- [ ] Out-of-scope items logged via `/create-task`
