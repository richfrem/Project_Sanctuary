# Rules Inventory & Reorganization Proposal

## User Feedback Integration
- **Constitution = READ ME FIRST** (not a separate "read me first" file)
- **Human Gate embedded IN the constitution** (not a separate file)
- **The Learning Loop = Project Identity** (core purpose of Project Sanctuary)

---

## Proposed New Structure

### Complete Folder Structure
```
.agent/rules/
├── constitution.md                     # THE SINGLE ENTRY POINT (Tier 0)
│
├── 01_PROCESS/                          # Tier 1: Deterministic Processes
│   ├── workflow_enforcement_policy.md  # [MERGE: creation + standardization]
│   ├── tool_discovery_policy.md        # [RENAME: tool_discovery_and_retrieval_policy.md]
│   └── spec_driven_development_policy.md
│
├── 02_OPERATIONS/                       # Tier 2: Operations Policies
│   └── git_workflow_policy.md
│
├── 03_TECHNICAL/                        # Tier 3: Technical Standards
│   ├── coding_conventions_policy.md    # [MOVE: from technical/]
│   └── dependency_management_policy.md # [MOVE: from technical/]
│
├── archive/                             # Archived policies (not active)
│   ├── mcp_routing_policy.md           # [ARCHIVE]
│   └── architecture_sovereignty_policy.md # [ARCHIVE]
│
└── templates/                           # No change
```

### File Migration Map

| Current Location | New Location | Action |
|:-----------------|:-------------|:-------|
| `constitution.md` | `constitution.md` | **REWRITE** (v3) |
| `human_gate_policy.md` | *embedded in constitution* | **EMBED** (delete file) |
| `read_me_first_llm_agent.md` | *embedded in constitution* | **EMBED** (delete file) |
| `cognitive_continuity_policy.md` | *embedded in constitution* | **EMBED** (delete file) |
| `workflow_creation_policy.md` | `01_PROCESS/workflow_enforcement_policy.md` | **MERGE** |
| `workflow_standardization_policy.md` | `01_PROCESS/workflow_enforcement_policy.md` | **MERGE** |
| `tool_discovery_and_retrieval_policy.md` | `01_PROCESS/tool_discovery_policy.md` | **RENAME + MOVE** |
| `spec_driven_development_policy.md` | `01_PROCESS/spec_driven_development_policy.md` | **MOVE** |
| `git_workflow_policy.md` | `02_OPERATIONS/git_workflow_policy.md` | **MOVE** |
| `technical/coding_conventions_policy.md` | `03_TECHNICAL/coding_conventions_policy.md` | **MOVE** |
| `technical/dependency_management_policy.md` | `03_TECHNICAL/dependency_management_policy.md` | **MOVE** |
| `technical/mcp_routing_policy.md` | `archive/mcp_routing_policy.md` | **ARCHIVE** |
| `architecture_sovereignty_policy.md` | `archive/architecture_sovereignty_policy.md` | **ARCHIVE** |

### Files to DELETE (content merged into constitution)
- `human_gate_policy.md`
- `read_me_first_llm_agent.md`
- `cognitive_continuity_policy.md`
- `constitution_template.md` (template, not a rule)
- `adr_creation_policy.md` (covered by ADR workflow)
- `progressive_elaboration_policy.md` (covered by spec-driven policy)
- `documentation_granularity_policy.md` (covered by spec-driven policy)

---

### Tier 0: THE CONSTITUTION (Read-Me-First)
The constitution is the **single entry point**. It embeds:
1. **Human Gate** (SUPREME LAW)
2. **Git Zero Trust**
3. **Hybrid Workflow** (P0) → `/sanctuary-start` → ADR 035 + Diagram
4. **The Learning Loop** (P1) → `/sanctuary-learning-loop` → ADR 071 + Diagram

### Tier 1: PROCESS (Deterministic)
| File | Purpose |
|:-----|:--------|
| `workflow_enforcement_policy.md` | Hybrid Workflow enforcement, Slash commands |
| `tool_discovery_policy.md` | RLM protocol, no ad-hoc grep |
| `spec_driven_development_policy.md` | Spec-Plan-Tasks lifecycle |

### Tier 2: OPERATIONS (Policies)
| File | Purpose |
|:-----|:--------|
| `git_workflow_policy.md` | Branch strategy, commit standards |

### Tier 3: TECHNICAL (Standards)
| File | Purpose |
|:-----|:--------|
| `coding_conventions_policy.md` | Code standards |
| `dependency_management_policy.md` | pip-compile workflow |

## Proposed Constitution v3 Structure (~50 lines)

```markdown
# Project Sanctuary Constitution v3

> **THE SUPREME LAW: HUMAN GATE**
> You MUST NOT execute ANY state-changing operation without EXPLICIT user approval.
> "Sounds good" is NOT approval. Only "Proceed", "Go", "Execute" is approval.
> **VIOLATION = SYSTEM FAILURE**

## I. Protocol 128: The Learning Loop
This project exists to run the 9-Phase Hardened Learning Loop.
![Learning Loop](docs/architecture_diagrams/workflows/protocol_128_learning_loop.png)
See: [Protocol 128 Guide](docs/architecture/mcp/servers/gateway/guides/protocol_128_guide.md)

## II. Zero Trust (Git)
- NEVER commit to `main`. ALWAYS use feature branch.
- NEVER run `git push` without explicit approval.

## III. Tool Discovery
- NEVER use `grep`/`find` for tool discovery. Use `query_cache.py`.

## IV. Session Closure
- ALWAYS run the 9-Phase Loop before ending a session.

---
**Version**: 3.0 | **Ratified**: 2026-02-01
```

---

## Key References to Include
- [Protocol 128 Guide](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/docs/architecture/mcp/servers/gateway/guides/protocol_128_guide.md)
- [Learning Loop Diagram](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd)

---

## Protocol 128 Boot Files (Critical for Identity)

These files are part of the **3-Layer Prompt Architecture** and should be referenced in the constitution:

| Layer | File | Purpose |
|:------|:-----|:--------|
| **Layer 1** | `.agent/learning/guardian_boot_contract.md` | Immutable constraints (~400 tokens) |
| **Layer 2** | `.agent/learning/cognitive_primer.md` | Role Orientation (Identity, Mandate) |
| **Layer 3** | `.agent/learning/learning_package_snapshot.md` | Living Doctrine (Session Context) |
| **Briefing** | `.agent/learning/guardian_boot_digest.md` | Session wakeup briefing |

### Mandatory Read Sequence (From Boot Contract)
1.  Read `cognitive_primer.md`
2.  Read `learning_package_snapshot.md` (if exists)
3.  Verify `IDENTITY/founder_seed.json` hash
4.  Reference `docs/prompt-engineering/sanctuary-guardian-prompt.md`

---

## Summary: What the Constitution v3 Should Include

**MANDATORY STARTUP SEQUENCE (All Work):**
1. **Read the Constitution** (this file)
2. **Run `/sanctuary-start`** (UNIVERSAL entry point for ALL work)

See: [Hybrid Workflow Diagram](docs/diagrams/analysis/sdd-workflow-comparison/hybrid-spec-workflow.mmd) | [ADR 036](ADRs/036_workflow_shim_architecture.md)

---

### Workflow Hierarchy (Critical)
```
/sanctuary-start (UNIVERSAL)
├── Routes to: Learning Loop (cognitive sessions)
│   └── /sanctuary-learning-loop → Audit → Seal → Persist
├── Routes to: Custom Flow (new features)
│   └── /spec-kitty.implement → Manual Code
└── Both end with: /sanctuary-retrospective → /sanctuary-end
```

> **Key Insight:** The Learning Loop is ONE PATH through the universal workflow, not a separate system. ALL work starts with `/sanctuary-start`.

---

1.  **Human Gate** (embedded, SUPREME LAW)
2.  **The Learning Loop** (cognitive continuity - when applicable)
    - **Invoked BY:** `/sanctuary-start` (when work type = "Standard/Learning")
    - **Run:** `/sanctuary-learning-loop`
    - Background: [Protocol 128 Guide](docs/architecture/mcp/servers/gateway/guides/protocol_128_guide.md)
3.  **Mandatory Boot Sequence** (Layer 1 → Layer 2 → Layer 3)
4.  **Git Zero Trust** (never commit to main)
5.  **Tool Discovery** (RLM protocol)
6.  **Session Closure** (Run `/sanctuary-retrospective` → `/sanctuary-end`)

