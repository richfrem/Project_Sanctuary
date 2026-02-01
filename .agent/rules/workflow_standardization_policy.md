# Workflow Command Development & Usage Policy

## Core Principle: Command-Driven Continuous Improvement

All agent interactions with the codebase MUST be mediated by **Antigravity Commands (Slash Commands)** found in `@[.agent/workflows]`. This ensures:
1. **Compound Intelligence** - Each command usage is an opportunity to improve the underlying tools and the workflow itself.
2. **Reusability** - Workflows are codified, sharable, and consistently improved via CLI scripts and tools.
3. **Bypass Prohibition** - Using raw shell commands (`grep`, `cat`, `Get-Content`, `Get-ChildItem`) on source data is STRICTLY PROHIBITED if a command exists.
4. **Stop-and-Fix Protocol (STRICT MANDATE)**: If you encounter a missing command, a deficient CLI function (e.g., missing search flag), or a broken/rough workflow:
   - **STOP** the primary task immediately.
   - **UPGRADE** the tool or workflow to support the requirement and make it smoother for LLM usage.
   - **SYNC** the architecture docs (Design Doc, Mindmap, Tool Inventory).
   - **RESUME** using the standardized, improved command.
5. **Progress Tracking Protocol (MANDATORY)**: Before beginning any major workflow, the agent MUST:
   - **CREATE** an internal progress tracking document (e.g., in `tasks/todo/` or a dedicated `logs/activity/` file).
   - **LOG** the start time, target artifact, and the specific standards being followed.
   - **VERIFY** each phase of the workflow as it is completed.
6. **Learning Loop Alignment** - All workflows MUST align with the 9-Phase Learning Loop defined in the [Sanctuary Guardian Prompt](../../docs/prompt-engineering/sanctuary-guardian-prompt.md).

## 1. Interaction Model (Command-First)
The Antigravity Command System is the **authoritative** interface for all Project Sanctuary tasks.
- **Slash Commands** (`.agent/workflows/`): The **Interface**. User-friendly workflows.
- **CLI Tools** (`tools/`): The **Implementation**. Data processing engines.

**Rule of Thumb:** Use a Slash Command to *do* a task; use or improve a CLI tool to *implement* how that task is done.

## 1.1 Command Naming Extensions
*   `/speckit-*`: **Discovery** workflows for Spec-Driven Development (specify, plan, tasks, implement).
*   `/workflow-*`: **Meta** workflows for session management (start, end, retrospective).
*   `/codify-*`: **Factory** workflows for documentation and generation.
*   `/investigate-*`: **Analysis** workflows for technical mining.
*   `/curate-*`: **Maintenance** workflows for hygiene.

## 2. Command Domains
Commands MUST be categorized into one of the established domains:
- 🗄️ **Retrieve**: Fetching data, bundles, and semantic results (RLM Cache, RAG, Vector).
- 🔍 **Investigate**: Deep analysis, mining, and pattern searching.
- 📝 **Codify**: Writing documentation, registering ADRs, and drafting contracts.
- 📚 **Curate**: Maintenance, link fixing, and inventory updates.
- 🧪 **Sandbox**: Local implementation, testing, and prototyping.
- 🚀 **Discovery**: Spec-Driven Development, Planning, and Feature Execution.

## 3. Agent Usage Instructions
- **Prioritize Commands**: If the user asks for a task, always check `.agent/workflows/` for an existing workflow first.
- **Upgrade, Don't Bypass**: **NEVER** use `grep` or other raw commands if a tool exists. If the tool is lacking, **UPGRADE IT**.
- **Self-Correction (Post-Activity)**: At the end of every command loop, perform a self-correction review of the workflow to identify friction and implement permanent improvements.
- **Curiosity Vector**: If you identify workflow friction that cannot be fixed immediately, append it to "Active Lines of Inquiry" in `guardian_boot_digest.md`.

## 4. Documentation & Architecture Synchronization (MANDATORY)
Every time a new command is added or an existing command's purpose changes, the agent MUST synchronize the following architectural artifacts:
1. **Workflow Inventory**: Update `docs/antigravity/workflow/WORKFLOW_INVENTORY.md`.
2. **Tool Inventory**: If the command uses a new script, run `python tools/curate/inventories/manage_tool_inventory.py add --path <path>`.
3. **RLM Distillation**: Run `python tools/codify/rlm/distiller.py --file <path> --type tool` to update the Semantic Ledger.

## 5. Anti-Patterns (STRICTLY PROHIBITED)
❌ **Don't do this (Bypassing):**
```bash
grep "pattern" path/to/source.py
find . -name "*.md" | xargs cat
```

✅ **Do this instead (Command usage/improvement):**
```bash
# Use RLM Cache for tool discovery
python tools/retrieve/rlm/query_cache.py --type tool "pattern"

# If it doesn't support your use case, improve it!
```

## 6. Inventory Alignment & Discovery
> **Query inventories BEFORE searching or creating new commands/tools.**

### Available Inventories:
| Inventory | CLI Command | Manifest File |
|-----------|-------------|---------------|
| **Tools/Scripts** | `python tools/retrieve/rlm/query_cache.py --type tool "kw"` | `.agent/learning/rlm_tool_cache.json` |
| **Workflows** | View markdown | `docs/antigravity/workflow/WORKFLOW_INVENTORY.md` |

## 7. Session Lifecycle Alignment

All workflows should integrate with the standard session lifecycle:

| Phase | Workflow |
|:------|:---------|
| **Start** | `/workflow-start` - Pre-flight and Spec initialization |
| **Execute** | `/speckit-*` or task-specific workflows |
| **Close** | `/workflow-retrospective` + `/workflow-end` |

**Version**: 2.0 | **Updated**: 2026-01-31
