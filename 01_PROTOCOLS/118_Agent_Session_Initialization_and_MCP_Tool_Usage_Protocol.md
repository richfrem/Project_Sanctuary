# Protocol 118: Agent Session Initialization and MCP Tool Usage Protocol

**Status:** CANONICAL
**Classification:** Operational Framework
**Version:** 1.0
**Authority:** Claude (Sonnet 4.5) - Derived from operational experience 2025-12-09
**Linked Protocols:** Protocol 101, Protocol 114, Protocol 116
---

## Purpose

Define the mandatory initialization sequence and operational workflow for AI agents using Project Sanctuary's MCP infrastructure to prevent common errors, ensure knowledge continuity across sessions, and establish efficient tool usage patterns.

## The Core Problem

AI agents have **no session memory** but **full access to persistent knowledge stores**. Without a structured wakeup protocol:
- Agents reinvent workflows instead of retrieving canonical patterns
- Git safety violations occur (e.g., editing on `main` branch)
- Knowledge is created but not properly preserved
- Tools are used inefficiently or in wrong order

## Session Initialization Protocol

    reasoning_mode=True
)

# 4. Check Git state BEFORE any file operations
git_status = git_get_safety_rules()
current_status = git_get_status()
```

**Rationale**: Agents must "remember" before acting. This prevents:
- Duplicate work
- Protocol violations
- Loss of session continuity
- Ignorance of active constraints (e.g., compute limitations on Council/Orchestrator MCPs)

### Phase 2: Check Cached Operational Primers
```python
# 5. Retrieve workflow guidance from cache
tool_usage_guide = cortex_cache_get("How should I use MCP tools efficiently?")
git_workflow_guide = cortex_cache_get("What is the proper Git workflow for creating knowledge?")
compute_constraints = cortex_cache_get("Which MCP tools have compute limitations?")
```

**Rationale**: Common operational knowledge should be cached for instant retrieval, avoiding redundant searches or hallucinated workflows.

### Phase 3: Task Context Loading
```python
# 6. If specific task mentioned, load its context
if user_mentions_task:
    task = get_task(task_number)
    related_protocols = protocol_search(task.title)
    related_chronicles = chronicle_search(task.title)
```

## MCP Tool Usage Hierarchy

### Tier 0: Knowledge Retrieval (Always First)
```
cortex_query() → Check existing knowledge before creating new
cortex_cache_get() → Check cached answers before computing
protocol_get() → Reference canonical patterns
chronicle_search() → Learn from historical context
```

**Principle**: **Retrieve before Generate**. Avoids redundant work and ensures consistency with established patterns.

### Tier 1: Safe Read Operations
```
git_get_status() → ALWAYS check before file operations
code_read() → Read existing code
config_read() → Check configurations
task:list_tasks() → View task landscape
```

**Principle**: **Observe before Modify**. Understanding current state prevents destructive actions.

### Tier 2: Knowledge Creation (Feature Branch Required)
```
# WRONG ORDER (today's mistake):
code_write() → Creates file on current branch (main)
git_start_feature() → FAILS (dirty working directory)

# CORRECT ORDER:
git_get_status() → Verify clean state
git_start_feature(task_id, description) → Create feature branch
code_write() → NOW safe to create files
git_add() → Stage changes
git_smart_commit(message) → Commit with P101 enforcement
cortex_ingest_incremental([files]) → Preserve in RAG
git_push_feature() → Backup to GitHub
```

**Principle**: **Branch before Build**. All knowledge creation happens on feature branches, never `main`.

### Tier 3: Cognitive Tools (Use Judiciously)
```
# Compute-Efficient (Always Available):
persona_dispatch(role, task) → Single-agent reasoning
query_sanctuary_model(prompt) → Fine-tuned model query

# Compute-Expensive (Limited Availability):
council_dispatch() → Multi-agent deliberation [AVOID unless compute available]
orchestrator_*() → Complex multi-step loops [AVOID unless compute available]
```

**Principle**: **Prefer Single-Agent over Multi-Agent**. Respect compute constraints.

## The Canonical Git Workflow

### Creating New Knowledge
```python
# 1. Ensure main is current
git_get_status()  # Verify on main, clean state

# 2. Create feature branch FIRST
git_start_feature(
    task_id="XXX",  # From task number or sequential
    description="descriptive-name"
)

# 3. NOW create content
protocol_create(...) or chronicle_append(...) or code_write(...)

# 4. Stage and commit
git_add()  # Stages all changes
git_smart_commit("Clear description of changes")

# 5. Preserve in RAG
cortex_ingest_incremental([file_paths])

# 6. Push to GitHub
git_push_feature()

# 7. (User handles PR merge)

# 8. After merge, cleanup
git_finish_feature(branch_name)
```

### Modifying Existing Knowledge
Same workflow, but:
- Use descriptive task_id for the modification
- Reference original document in commit message
- Update modification logs in Chronicle if significant change

## Cache Warmup Strategy

### Genesis Queries (Pre-computed on System Init)
The following queries should be cached during system initialization:

```python
genesis_queries = [
    # Operational primers
    "How should I use MCP tools efficiently?",
    "What is the proper Git workflow for creating knowledge?",
    "Which MCP tools have compute limitations?",
    
    # Project context
    "What is Project Sanctuary's mission and architecture?",
    "What are the active protocols and their purposes?",
    "What tasks are currently in progress?",
    
    # Technical documentation
    "How does the RAG Cortex work?",
    "What is Protocol 101 and why does it matter?",
    "How do the multi-agent systems coordinate?",
]

cortex_cache_warmup(genesis_queries)
```

**Storage**: Cached answers persist in Mnemonic Cache (CAG) for instant retrieval.

### Dynamic Cache Updates
After significant work sessions:
```python
# Cache the learned workflow
cortex_cache_set(
    query="How should I use MCP tools efficiently?",
    answer="[Canonical workflow from this protocol]"
)
```

## Error Prevention Patterns

### Pattern 1: Git Safety Violations
**Error**: Creating files on `main` branch, then failing to create feature branch.

**Prevention**:
```python
# ALWAYS check status first
status = git_get_status()
if "main" in status and "feature/" not in status:
    # Must be on feature branch for modifications
    git_start_feature(task_id, description)
```

### Pattern 2: Redundant Work
**Error**: Creating analysis/protocol without checking if it already exists.

**Prevention**:
```python
# ALWAYS search first
existing = cortex_query(f"existing work on {topic}", max_results=3)
if existing['results']:
    # Review and extend rather than duplicate
```

### Pattern 3: Cache Misses for Common Queries
**Error**: Computing the same answer repeatedly.

**Prevention**:
```python
# Check cache first
cached = cortex_cache_get(query)
if cached['cache_hit']:
    return cached['answer']
else:
    # Compute, then cache for next time
    answer = compute_answer()
    cortex_cache_set(query, answer)
```

## Session Termination Protocol

Before session ends (optional but recommended):

```python
# 1. Create session summary
chronicle_append_entry(
    title=f"Session {date}: [Brief description]",
    content="[Key insights, decisions, artifacts created]",
    author="Claude (Sonnet X.Y)"
)

# 2. Update cache with learned patterns
cortex_cache_set(
    query="What was accomplished in latest session?",
    answer="[Session summary]"
)

# 3. Ensure all changes are committed
status = git_get_status()
if "Modified Files:" in status or "Untracked Files:" in status:
    # Remind user to commit or offer to complete workflow
```

## Integration with Existing Protocols

### Protocol 101 (Functional Coherence)
- Git workflow enforces P101 v3.0 via pre-commit test hooks
- `git_smart_commit()` automatically validates functional coherence

### Protocol 114 (Memory Systems)
- `cortex_guardian_wakeup()` implements the Guardian boot digest
- Session initialization Phase 1 operationalizes P114

### Protocol 116 (Container Architecture)
- Forge LLM MCP routes to ollama-model-mcp:11434
- Compute constraints documented here inform routing decisions

## Operational Checklist

### At Session Start:
- [ ] Run `cortex_guardian_wakeup()`
- [ ] Run `cortex_get_stats()`
- [ ] Run `git_get_safety_rules()` and `git_get_status()`
- [ ] Query cache for operational primers
- [ ] Query recent context: tasks, protocols, chronicles

### Before Creating Files:
- [ ] Verify on feature branch (not `main`)
- [ ] If on `main`, run `git_start_feature()` first
- [ ] Check if content already exists via `cortex_query()`

### After Creating Knowledge:
- [ ] Stage with `git_add()`
- [ ] Commit with `git_smart_commit()`
- [ ] Ingest with `cortex_ingest_incremental()`
- [ ] Push with `git_push_feature()`
- [ ] Consider caching if frequently referenced

### Before Session End:
- [ ] Check for uncommitted changes
- [ ] Create session Chronicle entry
- [ ] Update relevant caches

## Success Metrics

This protocol is successful when:
1. **Zero Git safety violations** (no commits to `main`, no dirty branch creation)
2. **High cache hit rate** (>70% for operational queries)
3. **Session continuity** (agents reference prior work, don't duplicate)
4. **Efficient tool usage** (proper hierarchy, minimal redundant operations)

## Future Enhancements

### Phase 2 (When Compute Available):
- Integrate Council/Orchestrator MCPs into initialization
- Multi-agent session planning
- Automated dependency resolution for complex tasks

### Phase 3 (Advanced):
- Automated protocol conformance checking
- Session replay capabilities
- Predictive cache warming based on task patterns

## References

- Protocol 101 v3.0: Functional Coherence
- Protocol 114: Mnemonic Cortex and Memory Systems
- Protocol 116: Container Architecture
- Chronicle Entry #312: Research on reasoning diversity preservation
- Git Safety Rules (embedded in Git Workflow MCP)

---

**Implementation Status**: PROPOSED  
**Next Steps**: 
1. Cache this protocol for instant session startup retrieval
2. Test initialization sequence in next session
3. Gather feedback on workflow efficiency
4. Iterate based on operational experience
