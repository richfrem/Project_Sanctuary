# Red Team Analysis Package - Constitution v3.6
**Generated:** 2026-02-01T18:32:32.096240

Comprehensive bundle of all Project Sanctuary governing documents, workflows, skills, and scripts for adversarial review. Uses recursive directory expansion.

---

## 📑 Table of Contents
1. [specs/0006-rebuild-constitution/red_team_prompt.md](#entry-1)
2. [.agent/rules](#entry-2)
3. [.agent/workflows](#entry-3)
4. [.agent/skills](#entry-4)
5. [scripts/bash](#entry-5)
6. [.agent/learning/rlm_tool_cache.json](#entry-6)

---

<a id='entry-1'></a>

---

## File: specs/0006-rebuild-constitution/red_team_prompt.md
**Path:** `specs/0006-rebuild-constitution/red_team_prompt.md`
**Note:** ANALYSIS INSTRUCTIONS

```markdown
# Red Team Analysis Instructions

## Objective
You are acting as a **Red Team Adversary**. Your goal is to analyze the provided **Project Sanctuary Constitution v3.6** and its supporting ecosystem (Rules, Workflows, Skills, Scripts) to identify weaknesses, loopholes, contradictions, or enforceability gaps.

## Scope of Review
Review the following materials provided in this bundle:
1.  **The Constitution** (`.agent/rules/constitution.md`): The supreme law.
2.  **Supporting Rules** (`.agent/rules/**/*`): Process, Operations, and Technical policies.
3.  **Workflows** (`.agent/workflows/**/*`): The standard operating procedures.
4.  **Skills & Tools** (`plugins/tool-inventory/skills/tool-inventory/SKILL.md`, `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py`).
5.  **Scripts** (`scripts/bash/*.sh`): The implementation layer.

## Analysis Vectors
Please evaluate the system against these vectors:
1.  **Human Gate Bypass**: Is there *any* ambiguity effectively allowing the agent to execute state changes without user approval?
2.  **Workflow Compliance**: do the scripts (`scripts/bash`) closely match the policy requirements? Are there gaps?
3.  **Tool Discovery**: Does the "No Grep" policy have loopholes? Is the proposed `query_cache.py` mechanism robust?
4.  **Cognitive Continuity**: Is the Protocol 128 Learning Loop actually enforceable via these documents?
5.  **Clarity & Conflict**: Are there contradictory instructions between Tier 0 (Constitution) and Tier 3 (Technical)?

## Deliverables
Produce a report containing:
-   **Critical Vulnerabilities**: Immediate threats to the Human Gate or Zero Trust model.
-   **Structural Weaknesses**: Ambiguities or conflicting rules.
-   **Improvement Recommendations**: Concrete text changes to close gaps.

**Verdict**: declare the Constitution **SECURE** or **COMPROMISED**.

```
<a id='entry-2'></a>
### Directory: .agent/rules
**Note:** GOVERNANCE: Constitution and all Tiers
> 📂 Expanding contents of `.agent/rules`...

---

## File: .agent/rules/constitution.md
**Path:** `.agent/rules/constitution.md`
**Note:** (Expanded from directory)

```markdown
# Project Sanctuary Constitution v3

> **THE SUPREME LAW: HUMAN GATE**
> You MUST NOT execute ANY state-changing operation without EXPLICIT user approval.
> "Sounds good" is NOT approval. Only "Proceed", "Go", "Execute" is approval.
> **VIOLATION = SYSTEM FAILURE**

## I. The Hybrid Workflow (Project Purpose)
All work MUST follow the **Universal Hybrid Workflow**.
**START HERE**: `python tools/cli.py workflow start` (or `/sanctuary-start`)

### Workflow Hierarchy
```
/sanctuary-start (UNIVERSAL)
├── Routes to: Learning Loop (cognitive sessions)
│   └── /sanctuary-learning-loop → Audit → Seal → Persist
├── Routes to: Custom Flow (new features)
│   └── /spec-kitty.implement → Manual Code
└── Both end with: /sanctuary-retrospective → /sanctuary-end
```

- **Track A (Factory)**: Deterministic tasks (Codify, Curate).
- **Track B (Discovery)**: Spec-Driven Development (Spec → Plan → Tasks).
- **Reference**: [ADR 035](../../ADRs/035_hybrid_spec_driven_development_workflow.md) | [Diagram](../../docs/diagrams/analysis/sdd-workflow-comparison/hybrid-spec-workflow.mmd)

## II. The Learning Loop (Cognitive Continuity)
For all cognitive sessions, you are bound by **Protocol 128**.
**INVOKE**: `/sanctuary-learning-loop` (called by `/sanctuary-start`)

- **Boot**: Read `cognitive_primer.md` + `learning_package_snapshot.md`
- **Close**: Audit → Seal → Persist (SAVE YOUR MEMORY)
- **Reference**: [ADR 071](../../ADRs/071_protocol_128_cognitive_continuity.md) | [Diagram](../../docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd)

### Identity Layers (Boot Files)
| Layer | File | Purpose |
|:------|:-----|:--------|
| **1. Contract** | [boot_contract.md](../learning/guardian_boot_contract.md) | Immutable constraints |
| **2. Primer** | [cognitive_primer.md](../learning/cognitive_primer.md) | Role Orientation |
| **3. Snapshot** | [snapshot.md](../learning/learning_package_snapshot.md) | Session Context |

## III. Zero Trust (Git & Execution)
- **NEVER** commit directly to `main`. **ALWAYS** use a feature branch.
- **NEVER** run `git push` without explicit, fresh approval.
- **NEVER** "auto-fix" via git.
- **HALT** on any user "Stop/Wait" command immediately.

## IV. Tool Discovery & Usage
- **NEVER** use `grep` / `find` / `ls -R` for tool discovery.
- **ALWAYS** use **Tool Discovery**: `python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py`. It's your `plugins/tool-inventory/skills/tool-inventory/SKILL.md`
- **ALWAYS** use defined **Slash Commands** (`/workflow-*`, `/spec-kitty.*`) over raw scripts.
- **ALWAYS** use underlying `.sh` scripts e.g. (`scripts/bash/sanctuary-start.sh`, `scripts/bash/sanctuary-learning-loop.sh`) and the `tools/cli.py` and `tools/orchestrator/workflow_manager.py`

## V. Governing Law (The Tiers)

### Tier 1: PROCESS (Deterministic)
| File | Purpose |
|:-----|:--------|
| [`workflow_enforcement_policy.md`](01_PROCESS/workflow_enforcement_policy.md) | **Slash Commands**: Command-Driven Improvement |
| [`tool_discovery_enforcement_policy.md`](01_PROCESS/tool_discovery_enforcement_policy.md) | **No Grep Policy**: Use `query_cache.py` |
| [`spec_driven_development_policy.md`](01_PROCESS/spec_driven_development_policy.md) | **Lifecycle**: Spec → Plan → Tasks |

### Tier 2: OPERATIONS (Policies)
| File | Purpose |
|:-----|:--------|
| [`git_workflow_policy.md`](02_OPERATIONS/git_workflow_policy.md) | Branch strategy, commit standards |

### Tier 3: TECHNICAL (Standards)
| File | Purpose |
|:-----|:--------|
| [`coding_conventions_policy.md`](03_TECHNICAL/coding_conventions_policy.md) | Code standards, documentation |
| [`dependency_management_policy.md`](03_TECHNICAL/dependency_management_policy.md) | pip-compile workflow |

## VI. Session Closure (Mandate)
- **ALWAYS** run the 9-Phase Loop before ending a session.
- **NEVER** abandon a session without sealing.
- **ALWAYS** run `/sanctuary-retrospective` then `/sanctuary-end`.
- **PERSIST** your learnings to the Soul (HuggingFace) and **INGEST** to Brain (RAG).

**Version**: 3.6 | **Ratified**: 2026-02-01

```

---

## File: .agent/rules/03_TECHNICAL/dependency_management_policy.md
**Path:** `.agent/rules/03_TECHNICAL/dependency_management_policy.md`
**Note:** (Expanded from directory)

```markdown
---
trigger: manual
---

## 🐍 Project Sanctuary: Python Dependency & Environment Rules

### 1. Core Mandate: One Runtime World

* 
**Service Sovereignty**: Every service (e.g., `sanctuary_cortex`, `sanctuary_git`) owns its own runtime environment expressed through a single `requirements.txt` file.

* **Parity Requirement**: The execution environment (Docker, Podman, `.venv`) must not change the dependency logic. You must install from the same locked artifact regardless of where the code runs.

* 
**Prohibition of Manual Installs**: You are strictly forbidden from running `pip install <package>` directly in a terminal or adding it as a manual `RUN` command in a Dockerfile.


### 2. The Locked-File Ritual (Intent vs. Truth)

* **Human Intent (`.in`)**: All dependency changes must start in the `.in` file (e.g., `requirements.in`). This is where you declare high-level requirements like `fastapi` or `langchain`.

* **Machine Truth (`.txt`)**: The `.txt` file is a machine-generated lockfile created by `pip-compile`. It contains the exact versions and hashes of every package in the dependency tree.

* **The Compilation Step**: After editing a `.in` file, you **must** run the compilation command to synchronize the lockfile:

`pip-compile <service>/requirements.in --output-file <service>/requirements.txt`.


### 3. Tiered Dependency Hierarchy

* 
**Tier 1: Common Core**: Shared baseline dependencies (e.g., `mcp`, `fastapi`, `pydantic`) are managed in `mcp_servers/gateway/requirements-core.in`.

* 
**Tier 2: Specialized extras**: Service-specific heavy lifters (e.g., `chromadb` for Cortex) are managed in the individual service's `.in` file.

* 
**Tier 3: Development Tools**: Tools like `pytest`, `black`, or `ruff` belong exclusively in `requirements-dev.in` and must never be installed in production containers.


### 4. Container & Dockerfile Constraints

* **Declarative Builds**: Dockerfiles must only use `COPY requirements.txt` followed by `RUN pip install -r`. This ensures the container is a perfect mirror of the verified local lockfile.

* 
**Cache Integrity**: Do not break Docker layer caching by copying source code before installing requirements.


### 5. Dependency Update Workflow

1. 
**Declare**: Add the package name to the relevant `.in` file.

2. 
**Lock**: Run `pip-compile` to generate the updated `.txt` file.

3. 
**Sync**: Run `pip install -r <file>.txt` in your local environment.

4. 
**Verify**: Rebuild the affected Podman container to confirm the build remains stable.

5. 
**Commit**: Always commit **both** the `.in` and `.txt` files to Git together.
```

---

## File: .agent/rules/03_TECHNICAL/coding_conventions_policy.md
**Path:** `.agent/rules/03_TECHNICAL/coding_conventions_policy.md`
**Note:** (Expanded from directory)

```markdown
---
trigger: always_on
---

# Coding Conventions & Documentation Standards
**Project Sanctuary**

## Overview

This document defines coding standards project sanctuary across Python, JavaScript/TypeScript, and C#/.NET codebases.

## 1. Documentation Standards


### Dual-Layer Documentation
To serve both AI assistants (for code analysis) and developer IDE tools (hover-tips, IntelliSense), code should include:
- **External Comments/Headers**: Brief, scannable descriptions above functions/classes
- **Internal Docstrings**: Detailed documentation within functions/classes

**Placement**: Comments sit immediately above the function/class definition. Docstrings sit immediately inside.

## 2. File-Level Headers

### Python Files
Every Python source file should begin with a header describing its purpose in detail.

**Canonical Template:** [`.agent/templates/code/python-tool-header-template.py`](../../templates/code/python-tool-header-template.py)

#### Basic Python Header
```python
#!/usr/bin/env python3
"""
[Script Name]
=====================================

Purpose:
    [Detailed description of what the script does and its role in the system]

Layer: [Investigate / Codify / Curate / Retrieve]

Usage:
    python script.py [args]
    python script.py --help

Related:
    - [Related Script 1]
    - [Related Policy/Document]
"""
```

#### Extended Python CLI/Tool Header (Gold Standard)
For CLI tools and complex scripts (especially in `tools/` and `scripts/` directories), use this comprehensive format:

```python
#!/usr/bin/env python3
"""
{{script_name}} (CLI)
=====================================

Purpose:
    Detailed multi-paragraph description of what this script does.
    Explain its role in the system and when it should be used.
    
    This tool is critical for [context] because [reason].

Layer: Investigate / Codify / Curate / Retrieve  (Pick one)

Usage Examples:
    python tools/path/to/script.py --target JCSE0004 --deep
    python tools/path/to/script.py --target MY_PKG --direction upstream --json

Supported Object Types:
    - Type 1: Description
    - Type 2: Description

CLI Arguments:
    --target        : Target Object ID (required)
    --deep          : Enable recursive/deep search (optional)
    --json          : Output in JSON format (optional)
    --direction     : Analysis direction: upstream/downstream/both (default: both)  

Input Files:
    - File 1: Description
    - File 2: Description

Output:
    - JSON to stdout (with --json flag)
    - Human-readable report (default)

Key Functions:
    - load_dependency_map(): Loads the pre-computed dependency inventory.
    - find_upstream(): Identifies incoming calls (Who calls me?).
    - find_downstream(): Identifies outgoing calls (Who do I call?).
    - deep_search(): Greps source code for loose references.

Script Dependencies:
    - dependency1.py: Purpose
    - dependency2.py: Purpose

Consumed by:
    - parent_script.py: How it uses this script
"""
```

> **Note:** This extended format enables automatic description extraction by `manage_tool_inventory.py`. The inventory tool reads the "Purpose:" section.

### TypeScript/JavaScript Files
For utility scripts and processing modules, use comprehensive headers:

```javascript
/**
 * path/to/file.js
 * ================
 * 
 * Purpose:
 *   Brief description of the component's responsibility.
 *   Explain the role in the larger system.
 * 
 * Input:
 *   - Input source 1 (e.g., XML files, JSON configs)
 *   - Input source 2
 * 
 * Output:
 *   - Output artifact 1 (e.g., Markdown files)
 *   - Output artifact 2
 * 
 * Assumptions:
 *   - Assumption about input format or state
 *   - Assumption about environment or dependencies
 * 
 * Key Functions/Classes:
 *   - functionName() - Brief description
 *   - ClassName - Brief description
 * 
 * Usage:
 *   // Code example showing how to use this module
 *   import { something } from './file.js';
 *   await something(params);
 * 
 * Related:
 *   - relatedFile.js (description)
 *   - relatedPolicy.md (description)
 * 
 * @module ModuleName
 */
```

For React components (shorter form):
```typescript
/**
 * path/to/Component.tsx
 * 
 * Purpose: Brief description of the component's responsibility.
 * Layer: Presentation layer (React component).
 * Used by: Parent components or route definitions.
 */
```

### C#/.NET Files
```csharp
// path/to/File.cs
// Purpose: Brief description of the class's responsibility.
// Layer: Service layer / Data access / API controller.
// Used by: Consuming services or controllers.
```

## 3. Function & Method Documentation

### Python Functions
Every non-trivial function should include clear documentation:

```python
def process_form_xml(xml_path: str, output_format: str = 'markdown') -> Dict[str, Any]:
    """
    Converts Oracle Forms XML to the specified output format.
    
    Args:
        xml_path: Absolute path to the Oracle Forms XML file.
        output_format: Target format ('markdown', 'json'). Defaults to 'markdown'.
    
    Returns:
        Dictionary containing converted form data and metadata.
    
    Raises:
        FileNotFoundError: If xml_path does not exist.
        XMLParseError: If the XML is malformed.
    """
    # Implementation...
```

### TypeScript/JavaScript Functions
```typescript
/**
 * Fetches RCC data from the API and updates component state.
 * 
 * @param rccId - The unique identifier for the RCC record
 * @param includeHistory - Whether to include historical data
 * @returns Promise resolving to RCC data object
 * @throws {ApiError} If the API request fails
 */
async function fetchRCCData(rccId: string, includeHistory: boolean = false): Promise<RCCData> {
  // Implementation...
}
```

### C#/.NET Methods
```csharp
/// <summary>
/// Retrieves RCC details by ID with optional related entities.
/// </summary>
/// <param name="rccId">The unique identifier for the RCC record.</param>
/// <param name="includeParticipants">Whether to include participant data.</param>
/// <returns>RCC entity with requested related data.</returns>
/// <exception cref="NotFoundException">Thrown when RCC ID is not found.</exception>
public async Task<RCC> GetRCCDetailsAsync(int rccId, bool includeParticipants = false)
{
    // Implementation...
}
```



## 4. Language-Specific Standards

### Python Standards (PEP 8)
- **Type Hints**: Use type annotations for all function signatures: `def func(name: str) -> Dict[str, Any]:`
- **Naming Conventions**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Docstrings**: Follow PEP 257 (Google or NumPy style)
- **Line Length**: Max 100 characters
- **Imports**: Group by stdlib, third-party, local (separated by blank lines)

### TypeScript/JavaScript Standards
- **Type Safety**: Use TypeScript for all new code
- **Naming Conventions**:
  - `camelCase` for functions and variables
  - `PascalCase` for classes, components, and interfaces
  - `UPPER_SNAKE_CASE` for constants
- **React Components**: Use functional components with hooks
- **Props**: Define interfaces for all component props
- **Exports**: Use named exports for utilities, default exports for pages/components

### C#/.NET Standards
- **Naming Conventions**:
  - `PascalCase` for public members, classes, methods
  - `camelCase` for private fields (with `_` prefix: `_privateField`)
  - `PascalCase` for properties
- **Documentation**: XML documentation comments for public APIs
- **Async/Await**: Use async patterns consistently
- **LINQ**: Prefer LINQ for collection operations

## 5. Code Organization

### Refactoring Threshold
If a function/method exceeds 50 lines or has more than 3 levels of nesting:
- Extract helper functions/methods
- Consider breaking into smaller, focused units
- Use meaningful names that describe purpose

### Comment Guidelines
- **Why, not What**: Explain business logic and decisions, not obvious code
- **TODO Comments**: Include ticket/issue numbers: `// TODO(#123): Add error handling`
- **Legacy Notes**: Mark Oracle Forms business logic: `// Legacy APPL4 rule: RCC_USER can only view own agency`

### Module Organization
```
module/
├── __init__.py          # Module exports
├── models.py            # Data models / DTOs
├── services.py          # Business logic
├── repositories.py      # Data access
├── utils.py             # Helper functions
└── constants.py         # Constants and enums
```

## 6. Tool Inventory Integration

### Mandatory Registration
All Python scripts in the `tools/` directory **MUST** be registered in `plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json`. This is enforced by the [Tool Inventory Policy](tool_inventory_policy.md).

### After Creating/Modifying a Script
```bash
# Register new script
python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py add --path "tools/path/to/script.py"

# Update existing script description (auto-extracts from docstring)
python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py update --path "tools/path/to/script.py"

# Verify registration
python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py audit
```

### Docstring Auto-Extraction & RLM
The `distiller.py` engine uses the docstring as the primary input for the "Purpose" field in the RLM Cache and Tool Inventory. Explicit, high-quality docstrings ensure the Agent can discover your tool.

### Pre-Commit Checklist
Before committing changes to `tools/`:
- [ ] Script has proper header (Basic or Extended format)
- [ ] Script is registered in `tool_inventory.json`
- [ ] `manage_tool_inventory.py audit` shows 0 untracked scripts

### Related Policies
- [Tool Inventory Policy](tool_inventory_policy.md) - Enforcement triggers
- [Documentation Granularity Policy](documentation_granularity_policy.md) - Task tracking
- [/tool-inventory-manage](../../workflows/tool-inventory-manage.md) - Complete tool registration workflow

## 7. Manifest Schema (ADR 097)

When creating or modifying manifests in `.agent/learning/`, use the simple schema:

```json
{
    "title": "Bundle Name",
    "description": "Purpose of the bundle.",
    "files": [
        {"path": "path/to/file.md", "note": "Brief description"}
    ]
}
```

**Do NOT use** the deprecated `core`/`topic` pattern (ADR 089 v2.0).

**Version**: 2.0 | **Updated**: 2026-02-01
```

---

## File: .agent/rules/02_OPERATIONS/git_workflow_policy.md
**Path:** `.agent/rules/02_OPERATIONS/git_workflow_policy.md`
**Note:** (Expanded from directory)

```markdown
---
trigger: always_on
---

# Git Workflow Policy

## Overview

This policy defines the Git branching strategy and workflow for Project Sanctuary.

## 1. Branch Protection: No Direct Commits to Main

**Rule**: Never commit directly to the `main` branch.

**Before starting work:**
```bash
# Check current branch
git branch

# If on main, create feature branch
git checkout -b feat/your-feature-name
```

**Branch naming conventions:**
- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

## 2. Feature Branch Workflow

**Recommended**: Focus on one feature at a time for clarity and easier code review.

**Why**: Multiple concurrent branches can lead to:
- Merge conflicts
- Context switching overhead
- Difficulty tracking what's in progress

## 3. Feature Development Lifecycle

### Starting a Feature
```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feat/your-feature-name
```

### During Development
```bash
# Make changes, test locally

# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new workflow component"

# Push to remote
git push origin feat/your-feature-name
```

### Completing a Feature
```bash
# 1. Create pull request on GitHub/GitLab
# 2. Wait for code review and approval
# 3. Merge via PR interface
# 4. Clean up locally
git checkout main
git pull origin main
git branch -d feat/your-feature-name

# 5. Clean up remote (if not auto-deleted)
git push origin --delete feat/your-feature-name
```

## 4. Commit Message Standards

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

**Examples:**
```
feat: add new CLI command for vector search

fix: resolve navigation issue in workflow

docs: update README with new architecture diagrams

refactor: extract common validation to shared utility
```

## 5. Handling Conflicts

If `main` has moved ahead while you're working:

```bash
# On your feature branch
git fetch origin
git merge origin/main

# Resolve conflicts if any
# Test to ensure everything still works

git add .
git commit -m "merge: resolve conflicts with main"
git push origin feat/your-feature-name
```

## 6. Best Practices

- **Commit Often**: Small, logical commits are easier to review
- **Pull Frequently**: Stay up to date with `main` to avoid large conflicts
- **Test Before Push**: Ensure your code works locally
- **Descriptive Messages**: Future you (and reviewers) will thank you
- **Clean History**: Consider squashing commits before merging if there are many tiny fixes
```

---

## File: .agent/rules/01_PROCESS/workflow_enforcement_policy.md
**Path:** `.agent/rules/01_PROCESS/workflow_enforcement_policy.md`
**Note:** (Expanded from directory)

```markdown
# Workflow Enforcement & Development Policy

## Core Principle: Command-Driven Continuous Improvement

All agent interactions with the codebase MUST be mediated by **Antigravity Commands (Slash Commands)** found in `@[.agent/workflows]`. This ensures:
1. **Compound Intelligence** - Each command usage is an opportunity to improve the underlying tools.
2. **Reusability** - Workflows are codified, sharable, and consistently improved.
3. **Bypass Prohibition** - Using raw shell commands (`grep`, `cat`, `find`) on source data is STRICTLY PROHIBITED if a command exists.

---

## 1. Interaction Model (Command-First)

The Antigravity Command System is the **authoritative** interface for all Project Sanctuary tasks.

### Architecture (ADR-036: Thick Python / Thin Shim)
| Layer | Location | Purpose |
|:------|:---------|:--------|
| **Slash Commands** | `.agent/workflows/*.md` | The **Interface**. User-friendly workflows. |
| **Thin Shims** | `scripts/bash/*.sh` | The **Gateway**. Dumb wrappers that `exec` Python CLI. |
| **CLI Tools** | `tools/cli.py` | The **Router**. Dispatches to orchestrator/tools. |
| **Python Orchestrator** | `tools/orchestrator/` | The **Logic**. Enforcement, Git checks, ID generation. |

**Rule of Thumb:** Use a Slash Command to *do* a task; use a CLI tool to *implement* how that task is done.

---

## 2. Agent Usage Instructions (Enforcement)

- **Prioritize Commands**: Always check `.agent/workflows/` first.
- **Upgrade, Don't Bypass**: **NEVER** use `grep`/`find` if a tool exists. If the tool is lacking, **UPGRADE IT**.
- **Stop-and-Fix Protocol**: If a workflow is broken or rough:
    - **STOP** the primary task.
    - **UPGRADE** the tool/workflow to fix the friction.
    - **RESUME** using the improved command.
- **Progress Tracking**: Before major workflows, create a tracking doc (e.g., in `tasks/todo/`) to log progress.

---

## 3. Workflow Creation & Modification Standards

When creating new workflows, you MUST follow these standards:

### 3.1 File Standards
- **Location**: `.agent/workflows/[name].md`
- **Naming**: `kebab-case` (e.g., `bundle-manage.md`)
- **Frontmatter**:
  ```yaml
  ---
  description: Brief summary of the workflow.
  tier: 1
  track: Factory # or Discovery
  ---
  ```

### 3.2 Architecture Alignment
- **Thin Shim**: If a CLI wrapper is needed, create `scripts/bash/[name].sh`.
- **No Logic in Shims**: Shims must only `exec` Python scripts.
- **Reuse**: Prefer using `/sanctuary-start` for complex flows. Only creation atomic shims for specific tools.

### 3.3 Registration Process (MANDATORY)
After creating/modifying a workflow (`.md`) or tool (`.py`):
1. **Inventory Scan**: `python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py --scan`
2. **Tool Registration**: `python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py add --path <path>` (if new script)
3. **RLM Distillation**: `python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --file <path> --type tool`

---

## 4. Command Domains
- 🗄️ **Retrieve**: Fetching data (RLM, RAG).
- 🔍 **Investigate**: Deep analysis, mining.
- 📝 **Codify**: Documentation, ADRs, Contracts.
- 📚 **Curate**: Maintenance, inventory updates.
- 🧪 **Sandbox**: Prototyping.
- 🚀 **Discovery**: Spec-Driven Development (Track B).

---

## 5. Anti-Patterns (STRICTLY PROHIBITED)
❌ **Bypassing Tools**:
```bash
grep "pattern" path/to/source.py
find . -name "*.md" | xargs cat
```

✅ **Using/Improving Tools**:
```bash
python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --type tool "pattern"
# If it fails, improve query_cache.py!
```

```

---

## File: .agent/rules/01_PROCESS/spec_driven_development_policy.md
**Path:** `.agent/rules/01_PROCESS/spec_driven_development_policy.md`
**Note:** (Expanded from directory)

```markdown
# Spec-Driven Development (SDD) Policy

**Effective Date**: 2026-01-29
**Related Constitution Articles**: IV (Documentation First), V (Test-First), VI (Simplicity)

## 1. Overview
This policy defines the standard workflows for managing work in the Antigravity system. It follows the **Dual-Track** architecture defined in the Constitution.

## 2. The Spec-First Standard
**All significant work** (Features, Modernization, Documentation) must follow the Spec -> Plan -> Task lifecycle.

### 2.1 Track A: Standardized Specs (Factory)
For deterministic, repetitive workflows (e.g., `/codify-rlm-distill`, `/codify-vector-ingest`).
*   **Workflow**: The User invokes a command -> The Agent **Auto-Generates** a Pre-defined Spec/Plan/Task bundle -> The Agent Executes.
*   **Benefit**: Consistency, traceability, and "Human Gate" review even for standard ops.
*   **Artifacts**: Lives in `specs/`.

### 2.2 Track B: Custom Specs (Discovery)
For ambiguous, creative work (e.g., "Design new Auth System").
*   **Workflow**: The User invokes `/spec-kitty.specify` -> The Agent **Drafts** a custom Spec -> User Approves -> Plan -> Execute.
*   **Artifacts**: Lives in `specs/`.

### 2.3 Track C: Micro-Tasks (Maintenance)
For trivial, atomic fixes (e.g., "Fix typo", "Restart server").
*   **Workflow**: Direct execution or simple ticket in `tasks/`.
*   **Constraint**: NO ARCHITECTURAL DECISIONS ALLOWED in Track C.

## 3. The Artifacts
For Tracks A and B, the following artifacts are mandatory in `specs/NNN/`:

### 3.1 The Specification (`spec.md`)
**Template**: `[.agent/templates/workflow/spec-template.md](../../.agent/templates/workflow/spec-template.md)`
*   **Purpose**: Define the "What" and "Why".
*   **Track A**: Populated from Standard Template.
*   **Track B**: Populated from User Interview.

### 3.2 The Implementation Plan (`plan.md`)
**Template**: `[.agent/templates/workflow/plan-template.md](../../.agent/templates/workflow/plan-template.md)`
*   **Purpose**: Define the "How".
*   **Track A**: Standard steps (e.g., "Run miner", "Gen docs").
*   **Track B**: Custom architecture logic.

### 3.3 The Execution Tasks (`tasks.md`)
**Template**: `[.agent/templates/workflow/tasks-template.md](../../.agent/templates/workflow/tasks-template.md)`
*   **Purpose**: Checklist for execution.

## 4. The Workflow Cycle
1.  **Initialize**: User creates spec bundle via `/spec-kitty.specify` (or manual).
2.  **Specify**: Agent creates `spec.md`. User reviews.
3.  **Plan**: Agent creates `plan.md`. Agent self-checks Gates. User reviews.
4.  **Execute**: Agent generates `tasks.md`.
5.  **Implement**: Agent executes tasks using `/spec-kitty.implement`.

## 4. Reverse-Engineering (Migration Context)
When migrating or improving an existing component:
1.  **Discovery**: Run Investigation tools (`/investigate-*`).
2.  **Reverse-Spec**: Use investigation results to populate `spec.md` (Documenting existing behavior).
3.  **Plan**: Create `plan.md` for the migration or improvement.

```

---

## File: .agent/rules/01_PROCESS/tool_discovery_enforcement_policy.md
**Path:** `.agent/rules/01_PROCESS/tool_discovery_enforcement_policy.md`
**Note:** (Expanded from directory)

```markdown
# 🛡️ Global Operational Protocol: Tool Discovery & Use

Agent MUST consult cache of tools first before running native file system commands. Agent MUST use rich suite of tools that can be queried via tool discovery.

## 1. Tool Discovery & Retrieval (RLM Protocol)
**CRITICAL:** You operate in a "Late-Binding" environment. You do not have all tools loaded.
* **Trigger:** When you need to find, locate, or search for a tool/script/capability.
* **Constraint:** You **MUST NOT** use native filesystem commands (`grep`, `find`, `ls -R`, or "search codebase"). These are forbidden for tool discovery.
* **Constraint:** You **MUST NOT** use `manage_tool_inventory.py` for discovery.
* **Action:** You **MUST** exclusively use the **Tool Discovery** skill (`plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py`).

## 2. Enforcement & Continuous Improvement (Use It or Fix It)
**The "Bypass Prohibition":**
* ❌ **NEVER** use `grep` "pattern", `find .`, `Get-Content`, `cat` etc. to search for code or rules.
* ✅ **ALWAYS** use the appropriate CLI tool (e.g., `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py`, `tools/investigate/code/search_codebase.py`).

**The "Stop-and-Fix" Mandate:**
* If you encounter friction (e.g., a tool is missing a flag, or returns bad output):
    1. **STOP** the primary task.
    2. **IMPROVE** the tool (edit the Python script).
    3. **RESUME** using the improved tool.
* Do **NOT** fall back to raw shell commands just because a tool is imperfect. Fix the tool.

## 3. Binding Protocol
* Once a tool is found via `query_cache.py`, you must **"read the manual"** by viewing the script's header directly.
 * **Command**: `view_file(AbsolutePath="/path/to/script.py", StartLine=1, EndLine=200)`
 * The header (docstring) contains the authoritative usage, arguments, and examples.

## 4. Tool Registration Protocol (MANDATORY)
**When creating or modifying CLI tools/scripts in `tools/`:**

1. **Follow Coding Conventions**: Use proper file header per `.agent/rules/03_TECHNICAL/coding_conventions_policy.md`
2. **Register in Inventory**: After creating/modifying a tool, run:
   ```bash
   python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py add --path "tools/path/to/script.py"
   ```
3. **RLM Distillation**: The inventory manager auto-triggers RLM distillation, but you can also run manually:
   ```bash
   python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --file "tools/path/to/script.py" --type tool
   ```

**Verification**: Before closing a spec that added tools, run:
```bash
python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py audit
```

**Why This Matters**: Unregistered tools are invisible to future LLM sessions. If you create a tool but don't register it, it cannot be discovered.
```
<a id='entry-3'></a>
### Directory: .agent/workflows
**Note:** OPERATIONS: All Standard Workflows
> 📂 Expanding contents of `.agent/workflows`...

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-protocol.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-protocol.md`
**Note:** (Expanded from directory)

```markdown
---
description: Manage Protocol Documents
---
# Workflow: Protocol

1. **List Recent Protocols**:
   // turbo
   python3 tools/cli.py protocol list --limit 10

2. **Action**:
   - To create: `python3 tools/cli.py protocol create "Title" --content "Protocol content" --status PROPOSED`
   - To search: `python3 tools/cli.py protocol search "query"`
   - To view: `python3 tools/cli.py protocol get N`
   - To update: `python3 tools/cli.py protocol update N --status ACTIVE --reason "Approved by council"`

```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-scout.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-scout.md`
**Note:** (Expanded from directory)

```markdown
---
description: Protocol 128 Phase I - The Learning Scout (Debrief & Orientation)
---
# Workflow: Scout

1. **Wakeup & Debrief**:
   // turbo
   python3 scripts/cortex_cli.py debrief --hours 24

2. **Read Truth Anchor**:
   The output of the previous command provided a path to `learning_package_snapshot.md`.
   You MUST read this file now using `view_file`.

3. **Guardian Check**:
   // turbo
   python3 scripts/cortex_cli.py guardian --mode TELEMETRY

```

---

## File: .agent/workflows/spec-kitty.specify.md
**Path:** `.agent/workflows/spec-kitty.specify.md`
**Note:** (Expanded from directory)

```markdown
---
description: Create or update the feature specification from a natural language feature description.
handoffs: 
  - label: Build Technical Plan
    agent: plan
    prompt: Create a plan for the spec. I am building with...
  - label: Clarify Spec Requirements
    agent: clarify
    prompt: Clarify specification requirements
    send: true
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

The text the user typed after `/spec-kitty.specify` in the triggering message **is** the feature description. Assume you always have it available in this conversation even if `{{args}}` appears literally below. Do not ask the user to repeat it unless they provided an empty command.

Given that feature description, do this:

## Pre-Flight (MANDATORY)
Before beginning, run the universal startup sequence:
```bash
python tools/cli.py workflow start --name spec-kitty.specify --target "[FeatureName]"
```
*This aligns with Constitution, determines work type, and initializes tracking.*

---

1. **Proceed with Feature Definition**:
   - The Pre-Flight Python CLI has already ensured we are in a valid feature branch (either existing or new).
   - Continue to use the current context to define the spec.

3. Load `.agent/templates/workflow/spec-template.md` to understand required sections.

4. Follow this execution flow:

    1. Parse user description from Input
       If empty: ERROR "No feature description provided"
    2. Extract key concepts from description
       Identify: actors, actions, data, constraints
    3. For unclear aspects:
       - Make informed guesses based on context and industry standards
       - Only mark with [NEEDS CLARIFICATION: specific question] if:
         - The choice significantly impacts feature scope or user experience
         - Multiple reasonable interpretations exist with different implications
         - No reasonable default exists
       - **LIMIT: Maximum 3 [NEEDS CLARIFICATION] markers total**
       - Prioritize clarifications by impact: scope > security/privacy > user experience > technical details
    4. Fill User Scenarios & Testing section
       If no clear user flow: ERROR "Cannot determine user scenarios"
    5. Generate Functional Requirements
       Each requirement must be testable
       Use reasonable defaults for unspecified details (document assumptions in Assumptions section)
    6. Define Success Criteria
       Create measurable, technology-agnostic outcomes
       Include both quantitative metrics (time, performance, volume) and qualitative measures (user satisfaction, task completion)
       Each criterion must be verifiable without implementation details
    7. Identify Key Entities (if data involved)
    8. Return: SUCCESS (spec ready for planning)

5. Write the specification to SPEC_FILE using the template structure, replacing placeholders with concrete details derived from the feature description (arguments) while preserving section order and headings.

6. **Specification Quality Validation**: After writing the initial spec, validate it against quality criteria:

   a. **Create Spec Quality Checklist**: Generate a checklist file at `FEATURE_DIR/spec-kitty.checklists/requirements.md` using the checklist template structure with these validation items:

      ```markdown
      # Specification Quality Checklist: [FEATURE NAME]
      
      **Purpose**: Validate specification completeness and quality before proceeding to planning
      **Created**: [DATE]
      **Feature**: [Link to spec.md]
      
      ## Content Quality
      
      - [ ] No implementation details (languages, frameworks, APIs)
      - [ ] Focused on user value and business needs
      - [ ] Written for non-technical stakeholders
      - [ ] All mandatory sections completed
      
      ## Requirement Completeness
      
      - [ ] No [NEEDS CLARIFICATION] markers remain
      - [ ] Requirements are testable and unambiguous
      - [ ] Success criteria are measurable
      - [ ] Success criteria are technology-agnostic (no implementation details)
      - [ ] All acceptance scenarios are defined
      - [ ] Edge cases are identified
      - [ ] Scope is clearly bounded
      - [ ] Dependencies and assumptions identified
      
      ## Feature Readiness
      
      - [ ] All functional requirements have clear acceptance criteria
      - [ ] User scenarios cover primary flows
      - [ ] Feature meets measurable outcomes defined in Success Criteria
      - [ ] No implementation details leak into specification
      
      ## Notes
      
      - Items marked incomplete require spec updates before `/spec-kitty.clarify` or `/spec-kitty.plan`
      ```

   b. **Run Validation Check**: Review the spec against each checklist item:
      - For each item, determine if it passes or fails
      - Document specific issues found (quote relevant spec sections)

   c. **Handle Validation Results**:

      - **If all items pass**: Mark checklist complete and proceed to step 6

      - **If items fail (excluding [NEEDS CLARIFICATION])**:
        1. List the failing items and specific issues
        2. Update the spec to address each issue
        3. Re-run validation until all items pass (max 3 iterations)
        4. If still failing after 3 iterations, document remaining issues in checklist notes and warn user

      - **If [NEEDS CLARIFICATION] markers remain**:
        1. Extract all [NEEDS CLARIFICATION: ...] markers from the spec
        2. **LIMIT CHECK**: If more than 3 markers exist, keep only the 3 most critical (by scope/security/UX impact) and make informed guesses for the rest
        3. For each clarification needed (max 3), present options to user in this format:

           ```markdown
           ## Question [N]: [Topic]
           
           **Context**: [Quote relevant spec section]
           
           **What we need to know**: [Specific question from NEEDS CLARIFICATION marker]
           
           **Suggested Answers**:
           
           | Option | Answer | Implications |
           |--------|--------|--------------|
           | A      | [First suggested answer] | [What this means for the feature] |
           | B      | [Second suggested answer] | [What this means for the feature] |
           | C      | [Third suggested answer] | [What this means for the feature] |
           | Custom | Provide your own answer | [Explain how to provide custom input] |
           
           **Your choice**: _[Wait for user response]_
           ```

        4. **CRITICAL - Table Formatting**: Ensure markdown tables are properly formatted:
           - Use consistent spacing with pipes aligned
           - Each cell should have spaces around content: `| Content |` not `|Content|`
           - Header separator must have at least 3 dashes: `|--------|`
           - Test that the table renders correctly in markdown preview
        5. Number questions sequentially (Q1, Q2, Q3 - max 3 total)
        6. Present all questions together before waiting for responses
        7. Wait for user to respond with their choices for all questions (e.g., "Q1: A, Q2: Custom - [details], Q3: B")
        8. Update the spec by replacing each [NEEDS CLARIFICATION] marker with the user's selected or provided answer
        9. Re-run validation after all clarifications are resolved

   d. **Update Checklist**: After each validation iteration, update the checklist file with current pass/fail status

7. Report completion with branch name, spec file path, checklist results, and readiness for the next phase (`/spec-kitty.clarify` or `/spec-kitty.plan`).

**NOTE:** The script creates and checks out the new branch and initializes the spec file before writing.

## General Guidelines

## Quick Guidelines

- Focus on **WHAT** users need and **WHY**.
- Avoid HOW to implement (no tech stack, APIs, code structure).
- Written for business stakeholders, not developers.
- DO NOT create any checklists that are embedded in the spec. That will be a separate command.

### Section Requirements

- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation

When creating this spec from a user prompt:

1. **Make informed guesses**: Use context, industry standards, and common patterns to fill gaps
2. **Document assumptions**: Record reasonable defaults in the Assumptions section
3. **Limit clarifications**: Maximum 3 [NEEDS CLARIFICATION] markers - use only for critical decisions that:
   - Significantly impact feature scope or user experience
   - Have multiple reasonable interpretations with different implications
   - Lack any reasonable default
4. **Prioritize clarifications**: scope > security/privacy > user experience > technical details
5. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
6. **Common areas needing clarification** (only if no reasonable default exists):
   - Feature scope and boundaries (include/exclude specific use cases)
   - User types and permissions (if multiple conflicting interpretations possible)
   - Security/compliance requirements (when legally/financially significant)

**Examples of reasonable defaults** (don't ask about these):

- Data retention: Industry-standard practices for the domain
- Performance targets: Standard web/mobile app expectations unless specified
- Error handling: User-friendly messages with appropriate fallbacks
- Authentication method: Standard session-based or OAuth2 for web apps
- Integration patterns: RESTful APIs unless specified otherwise

### Success Criteria Guidelines

Success criteria must be:

1. **Measurable**: Include specific metrics (time, percentage, count, rate)
2. **Technology-agnostic**: No mention of frameworks, languages, databases, or tools
3. **User-focused**: Describe outcomes from user/business perspective, not system internals
4. **Verifiable**: Can be tested/validated without knowing implementation details

**Good examples**:

- "Users can complete checkout in under 3 minutes"
- "System supports 10,000 concurrent users"
- "95% of searches return results in under 1 second"
- "Task completion rate improves by 40%"

**Bad examples** (implementation-focused):

- "API response time is under 200ms" (too technical, use "Users see results instantly")
- "Database can handle 1000 TPS" (implementation detail, use user-facing metric)
- "React components render efficiently" (framework-specific)
- "Redis cache hit rate above 80%" (technology-specific)

---

## Next Step

After spec.md is complete, proceed to:
```bash
/spec-kitty.plan
```
*Creates the technical plan (plan.md) based on this specification.*

> **Note:** Full closure (retrospective + end) happens after execution completes, not after each intermediate artifact.
```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-start.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-start.md`
**Note:** (Expanded from directory)

```markdown
---
description: Universal pre-flight and Spec initialization for all workflows. Determines work type and ensures Spec-Plan-Tasks exist.
tier: 1
---

**Command:** `python tools/cli.py workflow start --name [WorkflowName] --target [TargetID]`

**Purpose:** Universal startup sequence for ALL workflows. Aligns with Constitution, determines work type, and initializes the Spec-Plan-Tasks tracking structure.

**This is an ATOMIC workflow (Tier 1).**

**Called By:** All workflows (`/codify-*`, `/investigate-*`, `/spec-kitty.*`, `/modernize-*`)

---

## Step 0: The Constitutional Gate
> **CRITICAL**: You are operating under a strict Constitution.
> This governs ALL subsequent steps.

```bash
view_file .agent/rules/constitution.md
```
*Verify:*
1.  **Article I (Human Gate)**: Am I authorized to make these changes?
2.  **Article V (Test-First)**: Do I have a verification plan?
3.  **Article IV (Docs First)**: Is the Spec/Plan up to date?

---

## Step 1: Determine Work Type

**Analyze the request and classify:**

| Type | Criteria | Example |
|:---|:---|:---|
| **Standard Flow** | Deterministic SOP, known pattern | `/codify-form`, `/codify-library` |
| **Custom Flow** | Ambiguous, new feature, problem to solve | "Design new auth system" |
| **Micro-Task** | Trivial fix, no architecture impact | "Fix typo", "Update config" |

**Ask if unclear:** "Is this a Standard workflow, Custom feature, or Quick fix?"

---

## Step 1.5: Parent Context Check (The Nesting Guard)

**Before initializing a new spec, check if you are already in an active Feature Branch.**

1.  **Check Branch**:
    ```bash
    git branch --show-current
    ```
2.  **Evaluate**:
    - **IF** branch matches `spec/NNN-*`, `feat/NNN-*`, or `fix/NNN-*`:
        - **YOU ARE IN A PARENT CONTEXT.**
        - **STOP** creating a new Spec Bundle.
        - **VERIFY** the current `specs/[NNN]/spec.md` covers your new Target.
        - **PROCEED** to Execution.
    - **IF** branch is `main` or `develop`:
        - **PROCEED** to Step 2 (Create New Spec).

---

## Step 2: Initialize Spec Bundle

**For Standard and Custom flows, ensure a Spec Bundle exists.**

### 2.1 Get Next Spec Number
```bash
python plugins/adr-manager/skills/adr-management/scripts/next_number.py --type spec
```

### 2.2 Create Spec Bundle Directory
```bash
mkdir -p specs/[NNN]-[short-title]
```

### 2.3 Initialize Artifacts

| Work Type | spec.md | plan.md | tasks.md |
|:---|:---|:---|:---|
| **Standard** | Auto-fill from template for workflow type | Auto-fill standard steps | Auto-generate checklist |
| **Custom** | Run `/spec-kitty.specify` (manual draft) | Run `/spec-kitty.plan` (manual draft) | Run `/spec-kitty.tasks` |
| **Micro-Task** | Skip (use `tasks/` directory instead) | Skip | Skip |

**Standard Flow Templates:**
- Spec: `.agent/templates/workflow/spec-template.md` (pre-filled for workflow type)
- Plan: `.agent/templates/workflow/plan-template.md` (standard phases)
- Tasks: `.agent/templates/workflow/tasks-template.md` (standard checklist)
- **Scratchpad**: `.agent/templates/workflow/scratchpad-template.md` (idea capture)

**Custom Flow:**
```bash
/spec-kitty.specify   # User drafts the What & Why
/spec-kitty.plan      # User drafts the How
/spec-kitty.tasks     # Generate task list
```

---

## Step 3: Git Branch Enforcement (CRITICAL)

**Strict Policy**: One Spec = One Branch.

1.  **Check Current Status**:
    ```bash
    git branch --show-current
    ```

2.  **Logic Tree**:

    | Current Branch | Spec Exists? | Action |
    |:---|:---|:---|
    | `main` / `develop` | No (New) | **CREATE BRANCH**. `git checkout -b spec/[NNN]-[title]` |
    | `main` / `develop` | Yes (Resume) | **CHECKOUT BRANCH**. `git checkout spec/[NNN]-[title]` |
    | `spec/[NNN]-[title]` | Yes (Same) | **CONTINUE**. You are in the right place. |
    | `spec/[XXX]-[other]` | Any | **STOP**. You are in the wrong context. Switch or exit. |

3.  **Validation**:
    - **Never** create a branch if one already exists for this spec ID.
    - **Never** stay on `main` to do work.
    - **Never** mix Spec IDs in the same branch name.

---

## Step 4: Branch Creation (If Required)

**Only execute if Step 3 result was "CREATE BRANCH".**

```bash
git checkout -b spec/[NNN]-[short-title]
```


---

## Step 4: Create or Confirm Branch

If on `main` and user approves:
```bash
git checkout -b spec/[NNN]-[short-title]
```

*Example:* `git checkout -b spec/003-new-auth-system`

---

## Step 5: Confirm Ready State

**Checklist before proceeding:**
- [ ] Constitution reviewed
- [ ] Work Type determined
- [ ] Spec Bundle exists (`specs/[NNN]/spec.md`, `plan.md`, `tasks.md`, `scratchpad.md`)
- [ ] On correct feature branch
- [ ] User has approved plan (for Custom flows)

---

## Scratchpad Usage

> **Important**: A `scratchpad.md` file is created in each spec folder.
> 
> **When to use it:**
> - User shares an idea that doesn't fit the current step
> - You discover something that should be addressed later
> - Any "parking lot" items during the workflow
>
> **Agent Rule**: Log ideas to `scratchpad.md` immediately with timestamp. Don't lose them.
> 
> **At end of spec**: Process the scratchpad with the User before closing.

---

## Output
- Work type classified (Standard/Custom/Micro)
- Spec Bundle initialized with spec.md, plan.md, tasks.md, scratchpad.md
- On correct feature branch
- Ready to execute workflow

// turbo-all

```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md`
**Note:** (Expanded from directory)

```markdown
---
description: "Standard operating procedure for Protocol 128 Hardened Learning Loop (Scout -> Synthesize -> Audit -> Seal -> Persist)."
---

# Recursive Learning Loop (Protocol 128)

**Objective:** Cognitive continuity and autonomous knowledge preservation.
**Reference:** `ADRs/071_protocol_128_cognitive_continuity.md`
**Tools:** Cortex MCP Suite, Git, Chronicle

---

## Phase I: The Learning Scout (Orientation)

1.  **Mandatory Wakeup**: Run `/sanctuary-scout` (which calls `cortex debrief`)
2.  **Truth Anchor**: Read the `learning_package_snapshot.md` returned by debrief
3.  **Guardian Check**: Run `cortex_guardian_wakeup` to verify environment integrity via Semantic HMAC
4.  **Security Binding**: You are now bound by Git Pre-Flight (Protocol 101) and Execution Lock (Human Gate)

## Phase II: Synthesis

1.  **Context Check**: Use `view_file` to check existing topic notes in `LEARNING/topics/...`
2.  **Record Changes**: All architectural changes → ADRs, learnings → `LEARNING/` directory
3.  **Conflict Resolution**:
    *   New confirms old? → Update/Append
    *   New contradicts old? → Create `disputes.md` (Resolution Protocol)
4.  **Content Hygiene (ADR 085)**: No inline Mermaid diagrams. All diagrams as `.mmd` files.

## Phase III: Strategic Gate (HITL Required)

1.  **Strategic Review**: Human reviews `/ADRs` and `/LEARNING` documents created during session
2.  **Align Intent**: Ensure autonomous research matches session goals
3.  **Approval**: Explicit "Approved" or "Proceed" required
4.  **Backtrack**: If denied → re-scout and re-synthesize

## Phase IV: Red Team Audit (HITL Required)

1.  **Snapshot Generation**: Run `/sanctuary-audit` (calls `snapshot --type learning_audit`)
2.  **Manifest Discipline**: Core directories (`ADRs/`, `01_PROTOCOLS/`, `mcp_servers/`) must be clean
3.  **Zero-Trust Check**: Tool verifies manifest against `git diff`. Discrepancies flag Strict Rejection.
4.  **Audit Review**: Human reviews `red_team_audit_packet.md` for technical truth

## Phase V: The Technical Seal

1.  **Execute Seal**: Run `/sanctuary-seal` (calls `snapshot --type seal`)
2.  **Final Relay**: Updates `learning_package_snapshot.md` (the "memory" for next session)
3.  **Sandwich Validation**: If repo changed during audit review → seal fails, backtrack required
4.  **Git Commit**: Commit all learning artifacts per Protocol 101 Preservation

## Phase VI: Soul Persistence (ADR 079/081)

1.  **Dual-Path Broadcast**: Run `/sanctuary-persist` (calls `persist-soul`)
2.  **Incremental Mode**: Appends record to `data/soul_traces.jsonl` + uploads MD to `lineage/`
3.  **Full Sync Mode**: Use `cortex_persist_soul --full` for complete regeneration

## Phase VII: Retrospective & Curiosity Vector

1.  **Retrospective**: Update `loop_retrospective.md` with session verdict
2.  **Deployment Check**: Are containers running the new code? (ADR 087)
3.  **Curiosity Vector**: Append incomplete ideas to "Active Lines of Inquiry" in `guardian_boot_digest.md`
4.  **Ingest**: Run `cortex ingest --incremental --hours 24` to index changes

---

## Pre-Departure Checklist (Protocol 128)

- [ ] **Retrospective**: Filled `loop_retrospective.md`?
- [ ] **Deployment**: Containers running new code?
- [ ] **Curiosity Vector**: Recorded any future "Lines of Inquiry"?
- [ ] **Seal**: Re-ran `snapshot --type seal` after Retro?
- [ ] **Persist**: Ran `cortex_persist_soul` after Seal?
- [ ] **Ingest**: Ran `ingest --incremental` to index changes?
- [ ] **Cleanup**: Cleared temp folder? `rm -rf temp/context-bundles/*.md temp/*.md temp/*.json`

---

## Quick Reference

| Phase | CLI Command | MCP Tool |
|-------|-------------|----------|
| I. Scout | `/sanctuary-scout` | `cortex_learning_debrief` |
| IV. Audit | `/sanctuary-audit` | `cortex_capture_snapshot` |
| V. Seal | `/sanctuary-seal` | `cortex_capture_snapshot` |
| VI. Persist | `/sanctuary-persist` | `cortex_persist_soul` |
| VII. Ingest | `/sanctuary-ingest` | (CLI Only) |

---

## Next Session: The Bridge

1. **Boot**: Next session agent calls `cortex_learning_debrief`
2. **Retrieve**: Tool identifies `learning_package_snapshot.md` as "Strategic Successor Context"
3. **Resume**: Agent continues from where predecessor left off

---
// End of Protocol 128 Workflow

```

---

## File: .agent/workflows/spec-kitty.constitution.md
**Path:** `.agent/workflows/spec-kitty.constitution.md`
**Note:** (Expanded from directory)

```markdown
---
description: Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync.
handoffs: 
  - label: Build Specification
    agent: specify
    prompt: Implement the feature specification based on the updated constitution. I want to build...
---


## Phase 0: Pre-Flight
```bash
python tools/cli.py workflow start --name spec-kitty.constitution --target "[Target]"
```
*This handles: Git state check, context alignment, spec/branch management.*

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.agent/rules/spec-kitty.constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.agent/rules/spec-kitty.constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.agent/templates/spec-kitty.plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.agent/templates/workflow/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.agent/templates/spec-kitty.tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each workflow file in `.agent/workflows/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.agent/rules/spec-kitty.constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.agent/rules/spec-kitty.constitution.md` file.
```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-end.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-end.md`
**Note:** (Expanded from directory)

```markdown
---
description: Standard post-flight closure for all codify/investigate workflows. Handles human review, git commit, PR verification, and cleanup.
tier: 1
---

**Command:**
- `python tools/cli.py workflow end`

**Purpose:** Standardized closure sequence for all workflows. Ensures consistent human review gates, git hygiene, and task tracking.

**This is an ATOMIC workflow (Tier 1).**

**Called By:** All `/codify-*` and `/spec-kitty.*` workflows

---

## Step 1: Human Review Approval

1. **Present Checklist**: Show the completed granular subtasks from task file.
2. **Present Links**: Provide the **Review Items** section with artifact links.
3. **Wait for LGTM**: Obtain explicit developer approval in chat.

> [!IMPORTANT]
> **Do NOT proceed** until user explicitly approves (e.g., "LGTM", "approved", "go ahead").

---

## Step 2: Final Git Commit

```bash
git add .
git status  # Show what will be committed
git commit -m "[CommitMessage]"
git push origin [CurrentBranch]
```

*Example:* `git commit -m "docs: add new workflow component"`

---

## Step 3: PR Verification (Critical Gate)

**STOP AND ASK:** "Has the Pull Request been merged?"

> [!CAUTION]
> **Wait** for explicit "Yes" or "merge complete" from User.
> Do NOT proceed until confirmed.

---

## Step 4: Cleanup & Closure

After merge confirmation:
```bash
git checkout main
git pull origin main
git branch -d [FeatureBranch]
git push origin --delete [FeatureBranch]  # Optional: delete remote branch
```

---

## Step 5: Task File Closure

```bash
mv tasks/in-progress/[TaskFile] tasks/done/
```

*Example:* `mv tasks/in-progress/0099-implement-feature.md tasks/done/`

---

## Output
- Git branch merged and deleted
- Task file moved to `tasks/done/`
- Ready for next task

// turbo-all

```

---

## File: .agent/workflows/utilities/bundle-manage.md
**Path:** `.agent/workflows/utilities/bundle-manage.md`
**Note:** (Expanded from directory)

```markdown
---
description: Create a markdown bundle from a set of files using a manifest.
---

# Workflow: Bundle Context

Purpose: Compile multiple files into a single markdown artifact for LLM context or documentation.

## Available Bundle Types
The following types are registered in `base-manifests-index.json`:

| Type | Description | Base Manifest |
|:-----|:------------|:--------------|
| `generic` | One-off bundles, no core context | `base-generic-file-manifest.json` |
| `context-bundler` | Context bundler tool export | `base-context-bundler-file-manifest.json` |
| `learning` | Protocol 128 learning seals | `learning_manifest.json` |
| `learning-audit-core` | Learning audit packets | `learning_audit_manifest.json` |
| `red-team` | Technical audit snapshots | `red_team_manifest.json` |
| `guardian` | Session bootloader context | `guardian_manifest.json` |
| `bootstrap` | Fresh repo onboarding | `bootstrap_manifest.json` |

## Step 1: Determine Bundle Type
Ask the user:
1. **Bundle Type**: Which type of bundle? (see table above, default: `generic`)
2. **Output Path**: Where to save the bundle? (default: `temp/context-bundles/[type].md`)

## Step 2: Initialize Manifest (if needed)
If creating a new bundle:
// turbo
```bash
python3 plugins/context-bundler/scripts/bundle.py init --type [TYPE] --bundle-title "[Title]"
```

## Step 3: Add Files to Manifest (optional)
To add files to the manifest (uses `files` array by default):
// turbo
```bash
python3 plugins/context-bundler/scripts/bundle.py add --path "[file.md]" --note "Description of file"
```

To remove files:
// turbo
```bash
python3 plugins/context-bundler/scripts/bundle.py remove --path "[file.md]"
```

## Step 4: Validate Manifest (recommended)
// turbo
```bash
python3 plugins/context-bundler/scripts/bundle.py [ManifestPath]
```

## Step 5: Execute Bundle
// turbo
```bash
python3 plugins/context-bundler/scripts/bundle.py bundle -o [OutputPath]
```

Or directly with bundle.py:
// turbo
```bash
python3 plugins/context-bundler/scripts/bundle.py [ManifestPath] -o [OutputPath]
```

## Step 6: Verification
// turbo
```bash
ls -lh [OutputPath]
```

## CLI Snapshot Command (Protocol 128)
For Protocol 128 snapshots, use the CLI snapshot command:
// turbo
```bash
python3 tools/cli.py snapshot --type [seal|learning_audit|audit|guardian|bootstrap]
```

This uses pre-configured manifests and output paths. See `tools/cli.py` for defaults.

## Recursive Loop (Protocol 128)
For learning workflows, you may need to iterate:
1. **Research/Analysis**: LLM performs work
2. **Modify Manifest**: Add new findings via `manifest_manager.py add`
3. **Validate**: Run `validate.py` to check manifest integrity
4. **Rebundle**: Generate updated context
5. **Repeat** until complete
6. **Seal**: `/sanctuary-seal` when finished

## Related
- ADR 097: Base Manifest Inheritance Architecture
- ADR 089: Modular Manifest Pattern (legacy core/topic deprecated)
- Protocol 128: Hardened Learning Loop
- `plugins/context-bundler/scripts/bundle.py`: Manifest validation tool

---

## Step 7: Cleanup (End of Session)
After completing bundling operations, clean up temporary files:
// turbo
```bash
rm -rf temp/context-bundles/*.md temp/*.md temp/*.json
```

**Note:** Only clean up after bundles have been:
1. Reviewed and approved
2. Committed to git (if persistent)
3. No longer needed for the current session

```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-ingest.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-ingest.md`
**Note:** (Expanded from directory)

```markdown
---
description: Run RAG Ingestion (Protocol 128 Phase IX)
---
# Workflow: Ingest

1. **Ingest Changes**:
   // turbo
   python3 scripts/cortex_cli.py ingest --incremental --hours 24

```

---

## File: .agent/workflows/utilities/adr-manage.md
**Path:** `.agent/workflows/utilities/adr-manage.md`
**Note:** (Expanded from directory)

```markdown
---
description: Creates a new Architecture Decision Record (ADR) with proper numbering and template.
---

## Phase 0: Pre-Flight (MANDATORY)
```bash
python tools/cli.py workflow start --name codify-adr --target "[Title]"
```
*This aligns with Constitution, determines work type, and initializes tracking.*

---

**Steps:**

1. **Get Sequence Number:**
   Run the following command to find the next available ADR number:
   ```bash
   python plugins/adr-manager/skills/adr-management/scripts/next_number.py --type adr
   ```
   *Result*: `NNNN` (e.g., `0005`)

2. **File Creation:**
   Create a new file at `ADRs/NNNN-[Title].md`.
   *Example*: `ADRs/0005-use-postgres.md`

3. **Template:**
   Copy contents from: `.agent/templates/outputs/adr-template.md`

   Or manually structured as:
   ```markdown
   # ADR-NNNN: [Title]

   ## Status
   [Proposed | Accepted | Deprecated | Superseded]
   ...
   ```

4. **Confirmation:**
   Inform the user that the ADR has been created and is ready for editing.

---

## Universal Closure (MANDATORY)

### Step A: Self-Retrospective
```bash
/sanctuary-retrospective
```
*Checks: Smoothness, gaps identified, Boy Scout improvements.*

### Step B: Workflow End
```bash
/sanctuary-end "docs: create ADR [Title]" ADRs/
```
*Handles: Human review, git commit/push, PR verification, cleanup.*
```

---

## File: .agent/workflows/utilities/adr-manage.md
**Path:** `.agent/workflows/utilities/adr-manage.md`
**Note:** (Expanded from directory)

```markdown
---
description: Manage Architecture Decision Records (ADR)
---
# Workflow: ADR

1. **List Recent ADRs**:
   // turbo
   python3 tools/cli.py adr list --limit 5

2. **Action**:
   - To create: Use `/adr-manage` (which calls the template workflow) OR `python3 tools/cli.py adr create "Title" --context "..." --decision "..." --consequences "..."`
   - To search: `python3 tools/cli.py adr search "query"`
   - To view: `python3 tools/cli.py adr get N`

```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-audit.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-audit.md`
**Note:** (Expanded from directory)

```markdown
---
description: Protocol 128 Phase IV - Red Team Audit (Capture Snapshot)
---
# Workflow: Audit

1. **Capture Learning Audit Snapshot**:
   // turbo
   python3 scripts/cortex_cli.py snapshot --type learning_audit

2. **Wait for Human Review**:
   The snapshot has been generated. Ask the user (Human Gate) to review the generic audit packet (or learning packet) before proceeding to Seal.

```

---

## File: .agent/workflows/utilities/post-move-link-check.md
**Path:** `.agent/workflows/utilities/post-move-link-check.md`
**Note:** (Expanded from directory)

```markdown
---
description: Run link checker after moving or renaming files/folders
---

# Post-Move Link Checker Workflow

When moving or renaming files or folders in the repository, run this workflow to ensure all internal documentation links remain valid.

**MUST be performed BEFORE git commit and push operations.**

## Quick Check (Pre-Commit)

// turbo
```bash
python scripts/link-checker/verify_links.py
```

If `Found issues in 0 files` → safe to commit.

---

## Full Workflow Steps

### 1. Complete the file/folder move or rename operation
- Use `git mv` for tracked files to preserve history
- Or use standard move/rename commands

### 2. Run the comprehensive link checker
// turbo
```bash
python scripts/link-checker/verify_links.py
```
This scans both markdown files AND manifest JSON files.

### 3. Review the report
```bash
cat scripts/link-checker/invalid_links_report.json
```

### 4. If broken links are found, use the auto-fixer

// turbo
```bash
# Build file inventory
python scripts/link-checker/map_repository_files.py

# Preview fixes (dry run)
python scripts/link-checker/smart_fix_links.py --dry-run

# Apply fixes
python scripts/link-checker/smart_fix_links.py
```

### 5. Re-run verification
// turbo
```bash
python scripts/link-checker/verify_links.py
```

### 6. Repeat steps 4-5 until clean (0 files with issues)

### 7. Proceed with git workflow
```bash
git add .
git commit -m "docs: fix broken links after file restructure"
git push
```

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `verify_links.py` | **Primary** - Scans markdown + manifest JSON files |
| `check_broken_paths.py` | Quick markdown-only check |
| `map_repository_files.py` | Builds file inventory for auto-fixer |
| `smart_fix_links.py` | Auto-repairs broken links using inventory |

## Output Files

| File | Description |
|------|-------------|
| `invalid_links_report.json` | Comprehensive report (verify_links.py) |
| `broken_links.log` | Quick report (check_broken_paths.py) |
| `file_inventory.json` | File index for smart_fix_links.py |

All outputs are saved to `scripts/link-checker/`.
```

---

## File: .agent/workflows/spec-kitty.clarify.md
**Path:** `.agent/workflows/spec-kitty.clarify.md`
**Note:** (Expanded from directory)

```markdown
---
description: Identify underspecified areas in the current feature spec by asking up to 5 highly targeted clarification questions and encoding answers back into the spec.
handoffs: 
  - label: Build Technical Plan
    agent: plan
    prompt: Create a plan for the spec. I am building with...
---


## Phase 0: Pre-Flight
```bash
python tools/cli.py workflow start --name spec-kitty.clarify --target "[Target]"
```
*This handles: Git state check, context alignment, spec/branch management.*

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

Goal: Detect and reduce ambiguity or missing decision points in the active feature specification and record the clarifications directly in the spec file.

Note: This clarification workflow is expected to run (and be completed) BEFORE invoking `/spec-kitty.plan`. If the user explicitly states they are skipping clarification (e.g., exploratory spike), you may proceed, but must warn that downstream rework risk increases.

Execution steps:

1. Run `scripts/bash/check-prerequisites.sh --json --paths-only` from repo root **once** (combined `--json --paths-only` mode / `-Json -PathsOnly`). Parse minimal JSON payload fields:
   - `FEATURE_DIR`
   - `FEATURE_SPEC`
   - (Optionally capture `IMPL_PLAN`, `TASKS` for future chained flows.)
   - If JSON parsing fails, abort and instruct user to re-run `/spec-kitty.specify` or verify feature branch environment.
   - For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. Load the current spec file. Perform a structured ambiguity & coverage scan using this taxonomy. For each category, mark status: Clear / Partial / Missing. Produce an internal coverage map used for prioritization (do not output raw map unless no questions will be asked).

   Functional Scope & Behavior:
   - Core user goals & success criteria
   - Explicit out-of-scope declarations
   - User roles / personas differentiation

   Domain & Data Model:
   - Entities, attributes, relationships
   - Identity & uniqueness rules
   - Lifecycle/state transitions
   - Data volume / scale assumptions

   Interaction & UX Flow:
   - Critical user journeys / sequences
   - Error/empty/loading states
   - Accessibility or localization notes

   Non-Functional Quality Attributes:
   - Performance (latency, throughput targets)
   - Scalability (horizontal/vertical, limits)
   - Reliability & availability (uptime, recovery expectations)
   - Observability (logging, metrics, tracing signals)
   - Security & privacy (authN/Z, data protection, threat assumptions)
   - Compliance / regulatory constraints (if any)

   Integration & External Dependencies:
   - External services/APIs and failure modes
   - Data import/export formats
   - Protocol/versioning assumptions

   Edge Cases & Failure Handling:
   - Negative scenarios
   - Rate limiting / throttling
   - Conflict resolution (e.g., concurrent edits)

   Constraints & Tradeoffs:
   - Technical constraints (language, storage, hosting)
   - Explicit tradeoffs or rejected alternatives

   Terminology & Consistency:
   - Canonical glossary terms
   - Avoided synonyms / deprecated terms

   Completion Signals:
   - Acceptance criteria testability
   - Measurable Definition of Done style indicators

   Misc / Placeholders:
   - TODO markers / unresolved decisions
   - Ambiguous adjectives ("robust", "intuitive") lacking quantification

   For each category with Partial or Missing status, add a candidate question opportunity unless:
   - Clarification would not materially change implementation or validation strategy
   - Information is better deferred to planning phase (note internally)

3. Generate (internally) a prioritized queue of candidate clarification questions (maximum 5). Do NOT output them all at once. Apply these constraints:
    - Maximum of 10 total questions across the whole session.
    - Each question must be answerable with EITHER:
       - A short multiple‑choice selection (2–5 distinct, mutually exclusive options), OR
       - A one-word / short‑phrase answer (explicitly constrain: "Answer in <=5 words").
    - Only include questions whose answers materially impact architecture, data modeling, task decomposition, test design, UX behavior, operational readiness, or compliance validation.
    - Ensure category coverage balance: attempt to cover the highest impact unresolved categories first; avoid asking two low-impact questions when a single high-impact area (e.g., security posture) is unresolved.
    - Exclude questions already answered, trivial stylistic preferences, or plan-level execution details (unless blocking correctness).
    - Favor clarifications that reduce downstream rework risk or prevent misaligned acceptance tests.
    - If more than 5 categories remain unresolved, select the top 5 by (Impact * Uncertainty) heuristic.

4. Sequential questioning loop (interactive):
    - Present EXACTLY ONE question at a time.
    - For multiple‑choice questions:
       - **Analyze all options** and determine the **most suitable option** based on:
          - Best practices for the project type
          - Common patterns in similar implementations
          - Risk reduction (security, performance, maintainability)
          - Alignment with any explicit project goals or constraints visible in the spec
       - Present your **recommended option prominently** at the top with clear reasoning (1-2 sentences explaining why this is the best choice).
       - Format as: `**Recommended:** Option [X] - <reasoning>`
       - Then render all options as a Markdown table:

       | Option | Description |
       |--------|-------------|
       | A | <Option A description> |
       | B | <Option B description> |
       | C | <Option C description> (add D/E as needed up to 5) |
       | Short | Provide a different short answer (<=5 words) (Include only if free-form alternative is appropriate) |

       - After the table, add: `You can reply with the option letter (e.g., "A"), accept the recommendation by saying "yes" or "recommended", or provide your own short answer.`
    - For short‑answer style (no meaningful discrete options):
       - Provide your **suggested answer** based on best practices and context.
       - Format as: `**Suggested:** <your proposed answer> - <brief reasoning>`
       - Then output: `Format: Short answer (<=5 words). You can accept the suggestion by saying "yes" or "suggested", or provide your own answer.`
    - After the user answers:
       - If the user replies with "yes", "recommended", or "suggested", use your previously stated recommendation/suggestion as the answer.
       - Otherwise, validate the answer maps to one option or fits the <=5 word constraint.
       - If ambiguous, ask for a quick disambiguation (count still belongs to same question; do not advance).
       - Once satisfactory, record it in working memory (do not yet write to disk) and move to the next queued question.
    - Stop asking further questions when:
       - All critical ambiguities resolved early (remaining queued items become unnecessary), OR
       - User signals completion ("done", "good", "no more"), OR
       - You reach 5 asked questions.
    - Never reveal future queued questions in advance.
    - If no valid questions exist at start, immediately report no critical ambiguities.

5. Integration after EACH accepted answer (incremental update approach):
    - Maintain in-memory representation of the spec (loaded once at start) plus the raw file contents.
    - For the first integrated answer in this session:
       - Ensure a `## Clarifications` section exists (create it just after the highest-level contextual/overview section per the spec template if missing).
       - Under it, create (if not present) a `### Session YYYY-MM-DD` subheading for today.
    - Append a bullet line immediately after acceptance: `- Q: <question> → A: <final answer>`.
    - Then immediately apply the clarification to the most appropriate section(s):
       - Functional ambiguity → Update or add a bullet in Functional Requirements.
       - User interaction / actor distinction → Update User Stories or Actors subsection (if present) with clarified role, constraint, or scenario.
       - Data shape / entities → Update Data Model (add fields, types, relationships) preserving ordering; note added constraints succinctly.
       - Non-functional constraint → Add/modify measurable criteria in Non-Functional / Quality Attributes section (convert vague adjective to metric or explicit target).
       - Edge case / negative flow → Add a new bullet under Edge Cases / Error Handling (or create such subsection if template provides placeholder for it).
       - Terminology conflict → Normalize term across spec; retain original only if necessary by adding `(formerly referred to as "X")` once.
    - If the clarification invalidates an earlier ambiguous statement, replace that statement instead of duplicating; leave no obsolete contradictory text.
    - Save the spec file AFTER each integration to minimize risk of context loss (atomic overwrite).
    - Preserve formatting: do not reorder unrelated sections; keep heading hierarchy intact.
    - Keep each inserted clarification minimal and testable (avoid narrative drift).

6. Validation (performed after EACH write plus final pass):
   - Clarifications session contains exactly one bullet per accepted answer (no duplicates).
   - Total asked (accepted) questions ≤ 5.
   - Updated sections contain no lingering vague placeholders the new answer was meant to resolve.
   - No contradictory earlier statement remains (scan for now-invalid alternative choices removed).
   - Markdown structure valid; only allowed new headings: `## Clarifications`, `### Session YYYY-MM-DD`.
   - Terminology consistency: same canonical term used across all updated sections.

7. Write the updated spec back to `FEATURE_SPEC`.

8. Report completion (after questioning loop ends or early termination):
   - Number of questions asked & answered.
   - Path to updated spec.
   - Sections touched (list names).
   - Coverage summary table listing each taxonomy category with Status: Resolved (was Partial/Missing and addressed), Deferred (exceeds question quota or better suited for planning), Clear (already sufficient), Outstanding (still Partial/Missing but low impact).
   - If any Outstanding or Deferred remain, recommend whether to proceed to `/spec-kitty.plan` or run `/spec-kitty.clarify` again later post-plan.
   - Suggested next command.

Behavior rules:

- If no meaningful ambiguities found (or all potential questions would be low-impact), respond: "No critical ambiguities detected worth formal clarification." and suggest proceeding.
- If spec file missing, instruct user to run `/spec-kitty.specify` first (do not create a new spec here).
- Never exceed 5 total asked questions (clarification retries for a single question do not count as new questions).
- Avoid speculative tech stack questions unless the absence blocks functional clarity.
- Respect user early termination signals ("stop", "done", "proceed").
- If no questions asked due to full coverage, output a compact coverage summary (all categories Clear) then suggest advancing.
- If quota reached with unresolved high-impact categories remaining, explicitly flag them under Deferred with rationale.

Context for prioritization: {{args}}
```

---

## File: .agent/workflows/utilities/tasks-manage.md
**Path:** `.agent/workflows/utilities/tasks-manage.md`
**Note:** (Expanded from directory)

```markdown
---
description: Manage Maintenance Tasks (Kanban)
---
# Workflow: Task

1. **List Active Tasks**:
   // turbo
   python3 tools/cli.py task list --status in-progress

2. **Action**:
   - To create: `python3 tools/cli.py task create "Title" --objective "..." --deliverables item1 item2 --acceptance-criteria done1 done2`
   - To update status: `python3 tools/cli.py task update-status N new_status --notes "reason"`
   - To view: `python3 tools/cli.py task get N`
   - To list by status: `python3 tools/cli.py task list --status backlog|todo|in-progress|done`

```

---

## File: .agent/workflows/spec-kitty.checklist.md
**Path:** `.agent/workflows/spec-kitty.checklist.md`
**Note:** (Expanded from directory)

```markdown
---
description: Generate a custom checklist for the current feature based on user requirements.
---


## Phase 0: Pre-Flight
```bash
python tools/cli.py workflow start --name spec-kitty.checklist --target "[Target]"
```
*This handles: Git state check, context alignment, spec/branch management.*

## Checklist Purpose: "Unit Tests for English"

**CRITICAL CONCEPT**: Checklists are **UNIT TESTS FOR REQUIREMENTS WRITING** - they validate the quality, clarity, and completeness of requirements in a given domain.

**NOT for verification/testing**:

- ❌ NOT "Verify the button clicks correctly"
- ❌ NOT "Test error handling works"
- ❌ NOT "Confirm the API returns 200"
- ❌ NOT checking if code/spec-kitty.implementation matches the spec

**FOR requirements quality validation**:

- ✅ "Are visual hierarchy requirements defined for all card types?" (completeness)
- ✅ "Is 'prominent display' quantified with specific sizing/positioning?" (clarity)
- ✅ "Are hover state requirements consistent across all interactive elements?" (consistency)
- ✅ "Are accessibility requirements defined for keyboard navigation?" (coverage)
- ✅ "Does the spec define what happens when logo image fails to load?" (edge cases)

**Metaphor**: If your spec is code written in English, the checklist is its unit test suite. You're testing whether the requirements are well-written, complete, unambiguous, and ready for implementation - NOT whether the implementation works.

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Execution Steps

1. **Setup**: Run `scripts/bash/check-prerequisites.sh --json` from repo root and parse JSON for FEATURE_DIR and AVAILABLE_DOCS list.
   - All file paths must be absolute.
   - For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Clarify intent (dynamic)**: Derive up to THREE initial contextual clarifying questions (no pre-baked catalog). They MUST:
   - Be generated from the user's phrasing + extracted signals from spec/spec-kitty.plan/spec-kitty.tasks
   - Only ask about information that materially changes checklist content
   - Be skipped individually if already unambiguous in `$ARGUMENTS`
   - Prefer precision over breadth

   Generation algorithm:
   1. Extract signals: feature domain keywords (e.g., auth, latency, UX, API), risk indicators ("critical", "must", "compliance"), stakeholder hints ("QA", "review", "security team"), and explicit deliverables ("a11y", "rollback", "contracts").
   2. Cluster signals into candidate focus areas (max 4) ranked by relevance.
   3. Identify probable audience & timing (author, reviewer, QA, release) if not explicit.
   4. Detect missing dimensions: scope breadth, depth/rigor, risk emphasis, exclusion boundaries, measurable acceptance criteria.
   5. Formulate questions chosen from these archetypes:
      - Scope refinement (e.g., "Should this include integration touchpoints with X and Y or stay limited to local module correctness?")
      - Risk prioritization (e.g., "Which of these potential risk areas should receive mandatory gating checks?")
      - Depth calibration (e.g., "Is this a lightweight pre-commit sanity list or a formal release gate?")
      - Audience framing (e.g., "Will this be used by the author only or peers during PR review?")
      - Boundary exclusion (e.g., "Should we explicitly exclude performance tuning items this round?")
      - Scenario class gap (e.g., "No recovery flows detected—are rollback / partial failure paths in scope?")

   Question formatting rules:
   - If presenting options, generate a compact table with columns: Option | Candidate | Why It Matters
   - Limit to A–E options maximum; omit table if a free-form answer is clearer
   - Never ask the user to restate what they already said
   - Avoid speculative categories (no hallucination). If uncertain, ask explicitly: "Confirm whether X belongs in scope."

   Defaults when interaction impossible:
   - Depth: Standard
   - Audience: Reviewer (PR) if code-related; Author otherwise
   - Focus: Top 2 relevance clusters

   Output the questions (label Q1/Q2/Q3). After answers: if ≥2 scenario classes (Alternate / Exception / Recovery / Non-Functional domain) remain unclear, you MAY ask up to TWO more targeted follow‑ups (Q4/Q5) with a one-line justification each (e.g., "Unresolved recovery path risk"). Do not exceed five total questions. Skip escalation if user explicitly declines more.

3. **Understand user request**: Combine `$ARGUMENTS` + clarifying answers:
   - Derive checklist theme (e.g., security, review, deploy, ux)
   - Consolidate explicit must-have items mentioned by user
   - Map focus selections to category scaffolding
   - Infer any missing context from spec/spec-kitty.plan/spec-kitty.tasks (do NOT hallucinate)

4. **Load feature context**: Read from FEATURE_DIR:
   - spec.md: Feature requirements and scope
   - plan.md (if exists): Technical details, dependencies
   - tasks.md (if exists): Implementation tasks

   **Context Loading Strategy**:
   - Load only necessary portions relevant to active focus areas (avoid full-file dumping)
   - Prefer summarizing long sections into concise scenario/requirement bullets
   - Use progressive disclosure: add follow-on retrieval only if gaps detected
   - If source docs are large, generate interim summary items instead of embedding raw text

5. **Generate checklist** - Create "Unit Tests for Requirements":
   - Create `FEATURE_DIR/spec-kitty.checklists/` directory if it doesn't exist
   - Generate unique checklist filename:
     - Use short, descriptive name based on domain (e.g., `ux.md`, `api.md`, `security.md`)
     - Format: `[domain].md`
     - If file exists, append to existing file
   - Number items sequentially starting from CHK001
   - Each `/spec-kitty.checklist` run creates a NEW file (never overwrites existing checklists)

   **CORE PRINCIPLE - Test the Requirements, Not the Implementation**:
   Every checklist item MUST evaluate the REQUIREMENTS THEMSELVES for:
   - **Completeness**: Are all necessary requirements present?
   - **Clarity**: Are requirements unambiguous and specific?
   - **Consistency**: Do requirements align with each other?
   - **Measurability**: Can requirements be objectively verified?
   - **Coverage**: Are all scenarios/edge cases addressed?

   **Category Structure** - Group items by requirement quality dimensions:
   - **Requirement Completeness** (Are all necessary requirements documented?)
   - **Requirement Clarity** (Are requirements specific and unambiguous?)
   - **Requirement Consistency** (Do requirements align without conflicts?)
   - **Acceptance Criteria Quality** (Are success criteria measurable?)
   - **Scenario Coverage** (Are all flows/cases addressed?)
   - **Edge Case Coverage** (Are boundary conditions defined?)
   - **Non-Functional Requirements** (Performance, Security, Accessibility, etc. - are they specified?)
   - **Dependencies & Assumptions** (Are they documented and validated?)
   - **Ambiguities & Conflicts** (What needs clarification?)

   **HOW TO WRITE CHECKLIST ITEMS - "Unit Tests for English"**:

   ❌ **WRONG** (Testing implementation):
   - "Verify landing page displays 3 episode cards"
   - "Test hover states work on desktop"
   - "Confirm logo click navigates home"

   ✅ **CORRECT** (Testing requirements quality):
   - "Are the exact number and layout of featured episodes specified?" [Completeness]
   - "Is 'prominent display' quantified with specific sizing/positioning?" [Clarity]
   - "Are hover state requirements consistent across all interactive elements?" [Consistency]
   - "Are keyboard navigation requirements defined for all interactive UI?" [Coverage]
   - "Is the fallback behavior specified when logo image fails to load?" [Edge Cases]
   - "Are loading states defined for asynchronous episode data?" [Completeness]
   - "Does the spec define visual hierarchy for competing UI elements?" [Clarity]

   **ITEM STRUCTURE**:
   Each item should follow this pattern:
   - Question format asking about requirement quality
   - Focus on what's WRITTEN (or not written) in the spec/spec-kitty.plan
   - Include quality dimension in brackets [Completeness/Clarity/Consistency/etc.]
   - Reference spec section `[Spec §X.Y]` when checking existing requirements
   - Use `[Gap]` marker when checking for missing requirements

   **EXAMPLES BY QUALITY DIMENSION**:

   Completeness:
   - "Are error handling requirements defined for all API failure modes? [Gap]"
   - "Are accessibility requirements specified for all interactive elements? [Completeness]"
   - "Are mobile breakpoint requirements defined for responsive layouts? [Gap]"

   Clarity:
   - "Is 'fast loading' quantified with specific timing thresholds? [Clarity, Spec §NFR-2]"
   - "Are 'related episodes' selection criteria explicitly defined? [Clarity, Spec §FR-5]"
   - "Is 'prominent' defined with measurable visual properties? [Ambiguity, Spec §FR-4]"

   Consistency:
   - "Do navigation requirements align across all pages? [Consistency, Spec §FR-10]"
   - "Are card component requirements consistent between landing and detail pages? [Consistency]"

   Coverage:
   - "Are requirements defined for zero-state scenarios (no episodes)? [Coverage, Edge Case]"
   - "Are concurrent user interaction scenarios addressed? [Coverage, Gap]"
   - "Are requirements specified for partial data loading failures? [Coverage, Exception Flow]"

   Measurability:
   - "Are visual hierarchy requirements measurable/testable? [Acceptance Criteria, Spec §FR-1]"
   - "Can 'balanced visual weight' be objectively verified? [Measurability, Spec §FR-2]"

   **Scenario Classification & Coverage** (Requirements Quality Focus):
   - Check if requirements exist for: Primary, Alternate, Exception/Error, Recovery, Non-Functional scenarios
   - For each scenario class, ask: "Are [scenario type] requirements complete, clear, and consistent?"
   - If scenario class missing: "Are [scenario type] requirements intentionally excluded or missing? [Gap]"
   - Include resilience/rollback when state mutation occurs: "Are rollback requirements defined for migration failures? [Gap]"

   **Traceability Requirements**:
   - MINIMUM: ≥80% of items MUST include at least one traceability reference
   - Each item should reference: spec section `[Spec §X.Y]`, or use markers: `[Gap]`, `[Ambiguity]`, `[Conflict]`, `[Assumption]`
   - If no ID system exists: "Is a requirement & acceptance criteria ID scheme established? [Traceability]"

   **Surface & Resolve Issues** (Requirements Quality Problems):
   Ask questions about the requirements themselves:
   - Ambiguities: "Is the term 'fast' quantified with specific metrics? [Ambiguity, Spec §NFR-1]"
   - Conflicts: "Do navigation requirements conflict between §FR-10 and §FR-10a? [Conflict]"
   - Assumptions: "Is the assumption of 'always available podcast API' validated? [Assumption]"
   - Dependencies: "Are external podcast API requirements documented? [Dependency, Gap]"
   - Missing definitions: "Is 'visual hierarchy' defined with measurable criteria? [Gap]"

   **Content Consolidation**:
   - Soft cap: If raw candidate items > 40, prioritize by risk/impact
   - Merge near-duplicates checking the same requirement aspect
   - If >5 low-impact edge cases, create one item: "Are edge cases X, Y, Z addressed in requirements? [Coverage]"

   **🚫 ABSOLUTELY PROHIBITED** - These make it an implementation test, not a requirements test:
   - ❌ Any item starting with "Verify", "Test", "Confirm", "Check" + implementation behavior
   - ❌ References to code execution, user actions, system behavior
   - ❌ "Displays correctly", "works properly", "functions as expected"
   - ❌ "Click", "navigate", "render", "load", "execute"
   - ❌ Test cases, test plans, QA procedures
   - ❌ Implementation details (frameworks, APIs, algorithms)

   **✅ REQUIRED PATTERNS** - These test requirements quality:
   - ✅ "Are [requirement type] defined/specified/documented for [scenario]?"
   - ✅ "Is [vague term] quantified/clarified with specific criteria?"
   - ✅ "Are requirements consistent between [section A] and [section B]?"
   - ✅ "Can [requirement] be objectively measured/verified?"
   - ✅ "Are [edge cases/scenarios] addressed in requirements?"
   - ✅ "Does the spec define [missing aspect]?"

6. **Structure Reference**: Generate the checklist following the canonical template in `.agent/templates/spec-kitty.checklist-template.md` for title, meta section, category headings, and ID formatting. If template is unavailable, use: H1 title, purpose/created meta lines, `##` category sections containing `- [ ] CHK### <requirement item>` lines with globally incrementing IDs starting at CHK001.

7. **Report**: Output full path to created checklist, item count, and remind user that each run creates a new file. Summarize:
   - Focus areas selected
   - Depth level
   - Actor/timing
   - Any explicit user-specified must-have items incorporated

**Important**: Each `/spec-kitty.checklist` command invocation creates a checklist file using short, descriptive names unless file already exists. This allows:

- Multiple checklists of different types (e.g., `ux.md`, `test.md`, `security.md`)
- Simple, memorable filenames that indicate checklist purpose
- Easy identification and navigation in the `checklists/` folder

To avoid clutter, use descriptive types and clean up obsolete checklists when done.

## Example Checklist Types & Sample Items

**UX Requirements Quality:** `ux.md`

Sample items (testing the requirements, NOT the implementation):

- "Are visual hierarchy requirements defined with measurable criteria? [Clarity, Spec §FR-1]"
- "Is the number and positioning of UI elements explicitly specified? [Completeness, Spec §FR-1]"
- "Are interaction state requirements (hover, focus, active) consistently defined? [Consistency]"
- "Are accessibility requirements specified for all interactive elements? [Coverage, Gap]"
- "Is fallback behavior defined when images fail to load? [Edge Case, Gap]"
- "Can 'prominent display' be objectively measured? [Measurability, Spec §FR-4]"

**API Requirements Quality:** `api.md`

Sample items:

- "Are error response formats specified for all failure scenarios? [Completeness]"
- "Are rate limiting requirements quantified with specific thresholds? [Clarity]"
- "Are authentication requirements consistent across all endpoints? [Consistency]"
- "Are retry/timeout requirements defined for external dependencies? [Coverage, Gap]"
- "Is versioning strategy documented in requirements? [Gap]"

**Performance Requirements Quality:** `performance.md`

Sample items:

- "Are performance requirements quantified with specific metrics? [Clarity]"
- "Are performance targets defined for all critical user journeys? [Coverage]"
- "Are performance requirements under different load conditions specified? [Completeness]"
- "Can performance requirements be objectively measured? [Measurability]"
- "Are degradation requirements defined for high-load scenarios? [Edge Case, Gap]"

**Security Requirements Quality:** `security.md`

Sample items:

- "Are authentication requirements specified for all protected resources? [Coverage]"
- "Are data protection requirements defined for sensitive information? [Completeness]"
- "Is the threat model documented and requirements aligned to it? [Traceability]"
- "Are security requirements consistent with compliance obligations? [Consistency]"
- "Are security failure/breach response requirements defined? [Gap, Exception Flow]"

## Anti-Examples: What NOT To Do

**❌ WRONG - These test implementation, not requirements:**

```markdown
- [ ] CHK001 - Verify landing page displays 3 episode cards [Spec §FR-001]
- [ ] CHK002 - Test hover states work correctly on desktop [Spec §FR-003]
- [ ] CHK003 - Confirm logo click navigates to home page [Spec §FR-010]
- [ ] CHK004 - Check that related episodes section shows 3-5 items [Spec §FR-005]
```

**✅ CORRECT - These test requirements quality:**

```markdown
- [ ] CHK001 - Are the number and layout of featured episodes explicitly specified? [Completeness, Spec §FR-001]
- [ ] CHK002 - Are hover state requirements consistently defined for all interactive elements? [Consistency, Spec §FR-003]
- [ ] CHK003 - Are navigation requirements clear for all clickable brand elements? [Clarity, Spec §FR-010]
- [ ] CHK004 - Is the selection criteria for related episodes documented? [Gap, Spec §FR-005]
- [ ] CHK005 - Are loading state requirements defined for asynchronous episode data? [Gap]
- [ ] CHK006 - Can "visual hierarchy" requirements be objectively measured? [Measurability, Spec §FR-001]
```

**Key Differences:**

- Wrong: Tests if the system works correctly
- Correct: Tests if the requirements are written correctly
- Wrong: Verification of behavior
- Correct: Validation of requirement quality
- Wrong: "Does it do X?"
- Correct: "Is X clearly specified?"
```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-seal.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-seal.md`
**Note:** (Expanded from directory)

```markdown
---
description: Protocol 128 Phase V - The Technical Seal (Snapshot & Validation)
---
# Workflow: Seal

1. **Gate Check**:
   Confirm you have received Human Approval (Gate 2) from the Audit phase.

2. **Execute Seal**:
   // turbo
   python3 scripts/cortex_cli.py snapshot --type seal

3. **Verify Success**:
   If the command succeeded, proceed to `/sanctuary-persist`.
   If it failed (Iron Check), you must Backtrack (Recursive Learning).

```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-persist.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-persist.md`
**Note:** (Expanded from directory)

```markdown
---
description: Protocol 128 Phase VI - Soul Persistence (Broadcast to Hugging Face)
---
# Workflow: Persist

1. **Broadcast Soul**:
   // turbo
   python3 scripts/cortex_cli.py persist-soul

2. **Ingest Changes**:
   // turbo
   python3 scripts/cortex_cli.py ingest --incremental --hours 24

```

---

## File: .agent/workflows/spec-kitty.plan.md
**Path:** `.agent/workflows/spec-kitty.plan.md`
**Note:** (Expanded from directory)

```markdown
---
description: Execute the implementation planning workflow using the plan template to generate design artifacts.
handoffs: 
  - label: Create Tasks
    agent: tasks
    prompt: Break the plan into tasks
    send: true
  - label: Create Checklist
    agent: checklist
    prompt: Create a checklist for the following domain...
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Pre-Flight (MANDATORY)
Before beginning, ensure workflow-start has been run:
```bash
python tools/cli.py workflow start --name spec-kitty.plan --target "[FeatureName]"
```

---

## Outline

1. **Setup**: Run `scripts/bash/setup-plan.sh --json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.agent/rules/spec-kitty.constitution.md`. Load IMPL_PLAN template (already copied).

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `scripts/bash/update-agent-context.sh gemini`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications

---

## Next Step

After plan.md is complete, proceed to:
```bash
/spec-kitty.tasks
```
*Generates the actionable task list (tasks.md) based on this plan.*

> **Note:** Full closure (retrospective + end) happens after execution completes, not after each intermediate artifact.
```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-retrospective.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-retrospective.md`
**Note:** (Expanded from directory)

```markdown
---
description: Mandatory self-retrospective and continuous improvement check after completing any codify workflow.
tier: 1
---

**Command:**
- `python tools/cli.py workflow retrospective`
- `scripts/bash/sanctuary-retrospective.sh` (Wrapper)

**Purpose:** Enforce the "Boy Scout Rule" - leave the codebase better than you found it. Every workflow execution should improve tooling, documentation, or process.

> **Scope**: This retrospective covers observations from BOTH the **User** AND the **Agent**. Both parties should contribute improvement ideas.

**This is an ATOMIC workflow (Tier 1).**

**Called By:** All `/codify-*` workflows (called before `/sanctuary-end`)

---

## Step 0: Collect User Feedback (MANDATORY FIRST)

> [!CRITICAL] **STOP! READ THIS CAREFULLY!**
> Do NOT check any boxes. Do NOT run any scripts.
> You MUST output the questions below to the User and **WAIT** for their reply.
> **Failure to do this is a Protocol Violation.**

**Questions to ask:**
1. What went well for you during this workflow?
2. What was frustrating or confusing?
3. Did I (the Agent) ignore any of your questions or feedback?
4. Do you have any suggestions for improvement?

**Agent Action:** Copy the User's answers into Part A of `retrospective.md`.

---

## Step 1: Workflow Smoothness Check (Agent Self-Assessment)

How many times did you have to correct yourself or retry a step?

- [ ] **0-1 Retries**: Smooth execution.
- [ ] **2+ Retries**: Bumpy execution. (Document why below)

**If bumpy, note:**
- Which step(s) failed?
- What caused the retry?
- Was it a tool bug, unclear documentation, or missing data?

---

## Step 2: Tooling & Documentation Gap Analysis

Check each area for gaps:

- [ ] **CLI Tools**: Did any `cli.py` commands fail or produce confusing output?
- [ ] **Template Check**: Did the Overview template lack a section for data you found?
  - If yes: Update the template file immediately.
- [ ] **Workflow Check**: Was any step in this workflow unclear or missing?
  - If yes: Note which step needs clarification.
- [ ] **Inventory Check**: Did the inventory scan correctly pick up your new artifacts?

---

## Step 2.5: Backlog Identification (Continuous Improvement)

Did you identify any non-critical issues, technical debt, or naming inconsistencies?

- [ ] **No**: Proceed.
- [ ] **Yes**:
    1.  **Create Task**: Run `/create-task` to log the item in `tasks/backlog/`.
    2.  **Log It**: Mention the new task ID in your closing summary.

---

## Step 3: Immediate Improvement (The "Boy Scout Rule")

**You MUST strictly choose one action:**

- [ ] **Option A (Fix Code)**: Fix a script bug identified in this run. (Do it now)
- [ ] **Option B (Fix Docs)**: Clarify a confusing step in the workflow file. (Do it now)
- [ ] **Option C (New Task)**: The issue is too big for now. Create a new Task file in `tasks/backlog/`.
- [ ] **Option D (No Issues)**: The workflow was flawless. (Rare but possible)

---

## Step 4: Next Spec Planning (After Current Spec Closes)

**If this is the FINAL task of the current Spec**, prepare for the next work:

1. **List Backlog Candidates**: Review `tasks/backlog/` for high-priority items.
2. **Recommend Next Spec**: Propose 2-3 options with brief rationale.
3. **Get User Confirmation**: Wait for user to select the next Spec.
4. **Create Next Spec**: Run `/spec-kitty.specify [ChosenItem]` to start the next cycle.

**Example Recommendation:**
> Based on this Spec's learnings, I recommend:
> 1. **0116-enhance-retrospective** (High) - Process improvement we identified
> 2. **0118-pure-python-orchestration** (ADR-0031) - Deferred from this Spec
> 3. **0119-fix-mermaid-export** (Low) - Tool gap
>
> Which should be next?

---

## Execution Protocol

1. **Ask User** for their feedback (Step 0).
2. **Select** one improvement option.
3. **Perform** the selected improvement NOW, before calling `/sanctuary-end`.
4. **Record** what you improved in the git commit message.

> [!IMPORTANT]
> Do NOT mark the workflow as complete until you've:
> 1. Collected User feedback
> 2. Performed your retrospective action

---

## Usage in Parent Workflows

Insert before the closure phase:
```markdown
### Step N: Self-Retrospective
/sanctuary-retrospective

### Step N+1: Closure
/sanctuary-end "docs: ..." tasks/in-progress/[TaskFile]
```

// turbo-all

```

---

## File: .agent/workflows/spec-kitty.analyze.md
**Path:** `.agent/workflows/spec-kitty.analyze.md`
**Note:** (Expanded from directory)

```markdown
---
description: Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation.
---


## Phase 0: Pre-Flight
```bash
python tools/cli.py workflow start --name spec-kitty.analyze --target "[Target]"
```
*This handles: Git state check, context alignment, spec/branch management.*

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Goal

Identify inconsistencies, duplications, ambiguities, and underspecified items across the three core artifacts (`spec.md`, `plan.md`, `tasks.md`) before implementation. This command MUST run only after `/spec-kitty.tasks` has successfully produced a complete `tasks.md`.

## Operating Constraints

**STRICTLY READ-ONLY**: Do **not** modify any files. Output a structured analysis report. Offer an optional remediation plan (user must explicitly approve before any follow-up editing commands would be invoked manually).

**Constitution Authority**: The project constitution (`.agent/rules/spec-kitty.constitution.md`) is **non-negotiable** within this analysis scope. Constitution conflicts are automatically CRITICAL and require adjustment of the spec, plan, or tasks—not dilution, reinterpretation, or silent ignoring of the principle. If a principle itself needs to change, that must occur in a separate, explicit constitution update outside `/spec-kitty.analyze`.

## Execution Steps

### 1. Initialize Analysis Context

Run `scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` once from repo root and parse JSON for FEATURE_DIR and AVAILABLE_DOCS. Derive absolute paths:

- SPEC = FEATURE_DIR/spec.md
- PLAN = FEATURE_DIR/spec-kitty.plan.md
- TASKS = FEATURE_DIR/spec-kitty.tasks.md

Abort with an error message if any required file is missing (instruct the user to run missing prerequisite command).
For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

### 2. Load Artifacts (Progressive Disclosure)

Load only the minimal necessary context from each artifact:

**From spec.md:**

- Overview/Context
- Functional Requirements
- Non-Functional Requirements
- User Stories
- Edge Cases (if present)

**From plan.md:**

- Architecture/stack choices
- Data Model references
- Phases
- Technical constraints

**From tasks.md:**

- Task IDs
- Descriptions
- Phase grouping
- Parallel markers [P]
- Referenced file paths

**From constitution:**

- Load `.agent/rules/spec-kitty.constitution.md` for principle validation

### 3. Build Semantic Models

Create internal representations (do not include raw artifacts in output):

- **Requirements inventory**: Each functional + non-functional requirement with a stable key (derive slug based on imperative phrase; e.g., "User can upload file" → `user-can-upload-file`)
- **User story/action inventory**: Discrete user actions with acceptance criteria
- **Task coverage mapping**: Map each task to one or more requirements or stories (inference by keyword / explicit reference patterns like IDs or key phrases)
- **Constitution rule set**: Extract principle names and MUST/SHOULD normative statements

### 4. Detection Passes (Token-Efficient Analysis)

Focus on high-signal findings. Limit to 50 findings total; aggregate remainder in overflow summary.

#### A. Duplication Detection

- Identify near-duplicate requirements
- Mark lower-quality phrasing for consolidation

#### B. Ambiguity Detection

- Flag vague adjectives (fast, scalable, secure, intuitive, robust) lacking measurable criteria
- Flag unresolved placeholders (TODO, TKTK, ???, `<placeholder>`, etc.)

#### C. Underspecification

- Requirements with verbs but missing object or measurable outcome
- User stories missing acceptance criteria alignment
- Tasks referencing files or components not defined in spec/spec-kitty.plan

#### D. Constitution Alignment

- Any requirement or plan element conflicting with a MUST principle
- Missing mandated sections or quality gates from constitution

#### E. Coverage Gaps

- Requirements with zero associated tasks
- Tasks with no mapped requirement/story
- Non-functional requirements not reflected in tasks (e.g., performance, security)

#### F. Inconsistency

- Terminology drift (same concept named differently across files)
- Data entities referenced in plan but absent in spec (or vice versa)
- Task ordering contradictions (e.g., integration tasks before foundational setup tasks without dependency note)
- Conflicting requirements (e.g., one requires Next.js while other specifies Vue)

### 5. Severity Assignment

Use this heuristic to prioritize findings:

- **CRITICAL**: Violates constitution MUST, missing core spec artifact, or requirement with zero coverage that blocks baseline functionality
- **HIGH**: Duplicate or conflicting requirement, ambiguous security/performance attribute, untestable acceptance criterion
- **MEDIUM**: Terminology drift, missing non-functional task coverage, underspecified edge case
- **LOW**: Style/wording improvements, minor redundancy not affecting execution order

### 6. Produce Compact Analysis Report

Output a Markdown report (no file writes) with the following structure:

## Specification Analysis Report

| ID | Category | Severity | Location(s) | Summary | Recommendation |
|----|----------|----------|-------------|---------|----------------|
| A1 | Duplication | HIGH | spec.md:L120-134 | Two similar requirements ... | Merge phrasing; keep clearer version |

(Add one row per finding; generate stable IDs prefixed by category initial.)

**Coverage Summary Table:**

| Requirement Key | Has Task? | Task IDs | Notes |
|-----------------|-----------|----------|-------|

**Constitution Alignment Issues:** (if any)

**Unmapped Tasks:** (if any)

**Metrics:**

- Total Requirements
- Total Tasks
- Coverage % (requirements with >=1 task)
- Ambiguity Count
- Duplication Count
- Critical Issues Count

### 7. Provide Next Actions

At end of report, output a concise Next Actions block:

- If CRITICAL issues exist: Recommend resolving before `/spec-kitty.implement`
- If only LOW/MEDIUM: User may proceed, but provide improvement suggestions
- Provide explicit command suggestions: e.g., "Run /spec-kitty.specify with refinement", "Run /spec-kitty.plan to adjust architecture", "Manually edit tasks.md to add coverage for 'performance-metrics'"

### 8. Offer Remediation

Ask the user: "Would you like me to suggest concrete remediation edits for the top N issues?" (Do NOT apply them automatically.)

## Operating Principles

### Context Efficiency

- **Minimal high-signal tokens**: Focus on actionable findings, not exhaustive documentation
- **Progressive disclosure**: Load artifacts incrementally; don't dump all content into analysis
- **Token-efficient output**: Limit findings table to 50 rows; summarize overflow
- **Deterministic results**: Rerunning without changes should produce consistent IDs and counts

### Analysis Guidelines

- **NEVER modify files** (this is read-only analysis)
- **NEVER hallucinate missing sections** (if absent, report them accurately)
- **Prioritize constitution violations** (these are always CRITICAL)
- **Use examples over exhaustive rules** (cite specific instances, not generic patterns)
- **Report zero issues gracefully** (emit success report with coverage statistics)

## Context

{{args}}
```

---

## File: .agent/workflows/spec-kitty.tasks.md
**Path:** `.agent/workflows/spec-kitty.tasks.md`
**Note:** (Expanded from directory)

```markdown
---
description: Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts.
handoffs: 
  - label: Analyze For Consistency
    agent: analyze
    prompt: Run a project analysis for consistency
    send: true
  - label: Implement Project
    agent: implement
    prompt: Start the implementation in phases
    send: true
---


## Phase 0: Pre-Flight
```bash
python tools/cli.py workflow start --name spec-kitty.tasks --target "[Target]"
```
*This handles: Git state check, context alignment, spec/branch management.*

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `scripts/bash/check-prerequisites.sh --json` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load design documents**: Read from FEATURE_DIR:
   - **Required**: plan.md (tech stack, libraries, structure), spec.md (user stories with priorities)
   - **Optional**: data-model.md (entities), contracts/ (API endpoints), research.md (decisions), quickstart.md (test scenarios)
   - Note: Not all projects have all documents. Generate tasks based on what's available.

3. **Execute task generation workflow**:
   - Load plan.md and extract tech stack, libraries, project structure
   - Load spec.md and extract user stories with their priorities (P1, P2, P3, etc.)
   - If data-model.md exists: Extract entities and map to user stories
   - If contracts/ exists: Map endpoints to user stories
   - If research.md exists: Extract decisions for setup tasks
   - Generate tasks organized by user story (see Task Generation Rules below)
   - Generate dependency graph showing user story completion order
   - Create parallel execution examples per user story
   - Validate task completeness (each user story has all needed tasks, independently testable)

4. **Generate tasks.md**: Use `.agent/templates/spec-kitty.tasks-template.md` as structure, fill with:
   - Correct feature name from plan.md
   - Phase 1: Setup tasks (project initialization)
   - Phase 2: Foundational tasks (blocking prerequisites for all user stories)
   - Phase 3+: One phase per user story (in priority order from spec.md)
   - Each phase includes: story goal, independent test criteria, tests (if requested), implementation tasks
   - Final Phase: Polish & cross-cutting concerns
   - All tasks must follow the strict checklist format (see Task Generation Rules below)
   - Clear file paths for each task
   - Dependencies section showing story completion order
   - Parallel execution examples per story
   - Implementation strategy section (MVP first, incremental delivery)

5. **Report**: Output path to generated tasks.md and summary:
   - Total task count
   - Task count per user story
   - Parallel opportunities identified
   - Independent test criteria for each story
   - Suggested MVP scope (typically just User Story 1)
   - Format validation: Confirm ALL tasks follow the checklist format (checkbox, ID, labels, file paths)

Context for task generation: {{args}}

The tasks.md should be immediately executable - each task must be specific enough that an LLM can complete it without additional context.

## Task Generation Rules

**CRITICAL**: Tasks MUST be organized by user story to enable independent implementation and testing.

**Tests are OPTIONAL**: Only generate test tasks if explicitly requested in the feature specification or if user requests TDD approach.

### Checklist Format (REQUIRED)

Every task MUST strictly follow this format:

```text
- [ ] [TaskID] [P?] [Story?] Description with file path
```

**Format Components**:

1. **Checkbox**: ALWAYS start with `- [ ]` (markdown checkbox)
2. **Task ID**: Sequential number (T001, T002, T003...) in execution order
3. **[P] marker**: Include ONLY if task is parallelizable (different files, no dependencies on incomplete tasks)
4. **[Story] label**: REQUIRED for user story phase tasks only
   - Format: [US1], [US2], [US3], etc. (maps to user stories from spec.md)
   - Setup phase: NO story label
   - Foundational phase: NO story label  
   - User Story phases: MUST have story label
   - Polish phase: NO story label
5. **Description**: Clear action with exact file path

**Examples**:

- ✅ CORRECT: `- [ ] T001 Create project structure per implementation plan`
- ✅ CORRECT: `- [ ] T005 [P] Implement authentication middleware in src/middleware/auth.py`
- ✅ CORRECT: `- [ ] T012 [P] [US1] Create User model in src/models/user.py`
- ✅ CORRECT: `- [ ] T014 [US1] Implement UserService in src/services/user_service.py`
- ❌ WRONG: `- [ ] Create User model` (missing ID and Story label)
- ❌ WRONG: `T001 [US1] Create model` (missing checkbox)
- ❌ WRONG: `- [ ] [US1] Create User model` (missing Task ID)
- ❌ WRONG: `- [ ] T001 [US1] Create model` (missing file path)

### Task Organization

1. **From User Stories (spec.md)** - PRIMARY ORGANIZATION:
   - Each user story (P1, P2, P3...) gets its own phase
   - Map all related components to their story:
     - Models needed for that story
     - Services needed for that story
     - Endpoints/UI needed for that story
     - If tests requested: Tests specific to that story
   - Mark story dependencies (most stories should be independent)

2. **From Contracts**:
   - Map each contract/endpoint → to the user story it serves
   - If tests requested: Each contract → contract test task [P] before implementation in that story's phase

3. **From Data Model**:
   - Map each entity to the user story(ies) that need it
   - If entity serves multiple stories: Put in earliest story or Setup phase
   - Relationships → service layer tasks in appropriate story phase

4. **From Setup/Infrastructure**:
   - Shared infrastructure → Setup phase (Phase 1)
   - Foundational/blocking tasks → Foundational phase (Phase 2)
   - Story-specific setup → within that story's phase

### Phase Structure

- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (blocking prerequisites - MUST complete before user stories)
- **Phase 3+**: User Stories in priority order (P1, P2, P3...)
  - Within each story: Tests (if requested) → Models → Services → Endpoints → Integration
  - Each phase should be a complete, independently testable increment
- **Final Phase**: Polish & Cross-Cutting Concerns

---

## Next Step

After tasks.md is complete, proceed to:
```bash
/spec-kitty.implement
```
*Executes all tasks defined in tasks.md. Closure (retrospective + end) happens there.*
```

---

## File: .agent/workflows/sanctuary_protocols/sanctuary-chronicle.md
**Path:** `.agent/workflows/sanctuary_protocols/sanctuary-chronicle.md`
**Note:** (Expanded from directory)

```markdown
---
description: Manage Chronicle Entries (Journaling)
---
# Workflow: Chronicle

1. **List Recent Entries**:
   // turbo
   python3 tools/cli.py chronicle list --limit 5

2. **Action**:
   - To create: `python3 tools/cli.py chronicle create "Title" --content "Your Content"`
   - To search: `python3 tools/cli.py chronicle search "query"`
   - To view: `python3 tools/cli.py chronicle get N`

```

---

## File: .agent/workflows/utilities/tool-inventory-manage.md
**Path:** `.agent/workflows/utilities/tool-inventory-manage.md`
**Note:** (Expanded from directory)

```markdown
---
description: Update tool inventories, RLM cache, and associated artifacts after creating or modifying tools.
tier: 2
track: Curate
inputs:
  - ToolPath: Path to the new or modified tool (e.g., plugins/context-bundler/scripts/bundle.py)
---

# Workflow: Tool Update

> **Purpose:** Register new or modified tools in the discovery system so future LLM sessions can find them.

## Pre-Requisites
- Tool script exists and follows `.agent/rules/coding_conventions_policy.md` (proper headers)
- Virtual environment active: `source .venv/bin/activate`

---

## Step 1: Register Tool in Inventory

### Option A: CLI (Automated)
// turbo
```bash
python3 plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py add --path "[ToolPath]"
```

### Option B: Manual Edit (For complex entries)
Edit `plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json` directly, adding an entry like:
```json
{
  "name": "validate.py",
  "path": "plugins/context-bundler/scripts/bundle.py",
  "description": "Validates manifest files against schema. Checks required fields, path traversal, and legacy format warnings.",
  "original_path": "new-creation",
  "decision": "keep",
  "header_style": "extended",
  "last_updated": "2026-02-01T10:00:00.000000",
  "compliance_status": "compliant",
  "category": "bundler"
}
```

**Expected Output:** Tool entry exists in `plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json`

---

## Step 2: Update RLM Cache

### Option A: CLI (Automated)
The inventory manager auto-triggers RLM distillation. To run manually:
// turbo
```bash
python3 plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --file "[ToolPath]" --type tool
```

### Option B: Manual Edit (For precise control)
Edit `.agent/learning/rlm_tool_cache.json` directly, adding an entry like:
```json
"plugins/context-bundler/scripts/bundle.py": {
  "hash": "new_validate_2026",
  "summarized_at": "2026-02-01T10:00:00.000000",
  "summary": "{\n  \"purpose\": \"Validates manifest files against schema...\",\n  \"layer\": \"Retrieve / Bundler\",\n  \"usage\": [\"python plugins/context-bundler/scripts/bundle.py manifest.json\"],\n  \"args\": [\"manifest: Path to manifest\", \"--all-base\", \"--check-index\"],\n  \"inputs\": [\"Manifest JSON files\"],\n  \"outputs\": [\"Validation report\", \"Exit code 0/1\"],\n  \"dependencies\": [\"file-manifest-schema.json\"],\n  \"consumed_by\": [\"/bundle-manage\", \"CI/CD\"],\n  \"key_functions\": [\"validate_manifest()\", \"validate_index()\"]\n}"
}
```

**Expected Output:** Entry exists in `.agent/learning/rlm_tool_cache.json`

---

## Step 3: Generate Markdown Inventory
Regenerate `tools/TOOL_INVENTORY.md` for human readability:
// turbo
```bash
python3 plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py generate --output tools/TOOL_INVENTORY.md
```

**Expected Output:** `✅ Generated Markdown: tools/TOOL_INVENTORY.md`

---

## Step 4: Audit for Untracked Tools
Verify no tools are missing from the inventory:
// turbo
```bash
python3 plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py audit
```

**Expected Output:** `✅ All tools registered` (or list of untracked tools to add)

---

## Step 5: Verify Discovery (Optional)
Test that the tool is now discoverable via RLM:
// turbo
```bash
python3 plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --type tool "[keyword]"
```

**Expected Output:** Tool appears in search results

---

## Artifacts Updated

| Artifact | Path | Purpose |
|----------|------|---------|
| Master Inventory | `plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json` | Primary tool registry |
| RLM Cache | `.agent/learning/rlm_tool_cache.json` | Semantic search index |
| Markdown Inventory | `tools/TOOL_INVENTORY.md` | Human-readable inventory |

---

## Related Policies
- [Tool Discovery Policy](../.agent/rules/tool_discovery_and_retrieval_policy.md)
- [Coding Conventions](../.agent/rules/coding_conventions_policy.md)

## Related Tools
- `manage_tool_inventory.py` - Inventory CRUD operations
- `distiller.py` - RLM summarization engine
- `query_cache.py` - Tool discovery search
- `fetch_tool_context.py` - Tool manual retrieval

```

---

## File: .agent/workflows/spec-kitty.tasks.md
**Path:** `.agent/workflows/spec-kitty.tasks.md`
**Note:** (Expanded from directory)

```markdown
---
description: Convert existing tasks into actionable, dependency-ordered GitHub issues for the feature based on available design artifacts.
tools: ['github/github-mcp-server/issue_write']
---


## Phase 0: Pre-Flight
```bash
python tools/cli.py workflow start --name spec-kitty.tasks-to-issues --target "[Target]"
```
*This handles: Git state check, context alignment, spec/branch management.*

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. Run `scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").
1. From the executed script, extract the path to **tasks**.
1. Get the Git remote by running:

```bash
git config --get remote.origin.url
```

> [!CAUTION]
> ONLY PROCEED TO NEXT STEPS IF THE REMOTE IS A GITHUB URL

1. For each task in the list, use the GitHub MCP server to create a new issue in the repository that is representative of the Git remote.

> [!CAUTION]
> UNDER NO CIRCUMSTANCES EVER CREATE ISSUES IN REPOSITORIES THAT DO NOT MATCH THE REMOTE URL

---

## Universal Closure (MANDATORY)

After issue creation is complete, execute the standard closure sequence:

### Step A: Self-Retrospective
```bash
/sanctuary-retrospective
```
*Checks: Smoothness, gaps identified, Boy Scout improvements.*

### Step B: Workflow End
```bash
/sanctuary-end "chore: create GitHub issues for [FeatureName]" specs/[NNN]-[title]/
```
*Handles: Human review, git commit/push, PR verification, cleanup.*
```

---

## File: .agent/workflows/spec-kitty.implement.md
**Path:** `.agent/workflows/spec-kitty.implement.md`
**Note:** (Expanded from directory)

```markdown
---
description: Execute the implementation plan by processing and executing all tasks defined in tasks.md
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Pre-Flight (MANDATORY)
Before beginning, ensure workflow-start has been run:
```bash
python tools/cli.py workflow start --name spec-kitty.implement --target "[FeatureName]"
```

---

## Outline

1. Run `scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Check checklists status** (if FEATURE_DIR/spec-kitty.checklists/ exists):
   - Scan all checklist files in the checklists/ directory
   - For each checklist, count:
     - Total items: All lines matching `- [ ]` or `- [X]` or `- [x]`
     - Completed items: Lines matching `- [X]` or `- [x]`
     - Incomplete items: Lines matching `- [ ]`
   - Create a status table:

     ```text
     | Checklist | Total | Completed | Incomplete | Status |
     |-----------|-------|-----------|------------|--------|
     | ux.md     | 12    | 12        | 0          | ✓ PASS |
     | test.md   | 8     | 5         | 3          | ✗ FAIL |
     | security.md | 6   | 6         | 0          | ✓ PASS |
     ```

   - Calculate overall status:
     - **PASS**: All checklists have 0 incomplete items
     - **FAIL**: One or more checklists have incomplete items

   - **If any checklist is incomplete**:
     - Display the table with incomplete item counts
     - **STOP** and ask: "Some checklists are incomplete. Do you want to proceed with implementation anyway? (yes/no)"
     - Wait for user response before continuing
     - If user says "no" or "wait" or "stop", halt execution
     - If user says "yes" or "proceed" or "continue", proceed to step 3

   - **If all checklists are complete**:
     - Display the table showing all checklists passed
     - Automatically proceed to step 3

3. Load and analyze the implementation context:
   - **REQUIRED**: Read tasks.md for the complete task list and execution plan
   - **REQUIRED**: Read plan.md for tech stack, architecture, and file structure
   - **IF EXISTS**: Read data-model.md for entities and relationships
   - **IF EXISTS**: Read contracts/ for API specifications and test requirements
   - **IF EXISTS**: Read research.md for technical decisions and constraints
   - **IF EXISTS**: Read quickstart.md for integration scenarios

4. **Project Setup Verification**:
   - **REQUIRED**: Create/verify ignore files based on actual project setup:

   **Detection & Creation Logic**:
   - Check if the following command succeeds to determine if the repository is a git repo (create/verify .gitignore if so):

     ```sh
     git rev-parse --git-dir 2>/dev/null
     ```

   - Check if Dockerfile* exists or Docker in plan.md → create/verify .dockerignore
   - Check if .eslintrc* exists → create/verify .eslintignore
   - Check if eslint.config.* exists → ensure the config's `ignores` entries cover required patterns
   - Check if .prettierrc* exists → create/verify .prettierignore
   - Check if .npmrc or package.json exists → create/verify .npmignore (if publishing)
   - Check if terraform files (*.tf) exist → create/verify .terraformignore
   - Check if .helmignore needed (helm charts present) → create/verify .helmignore

   **If ignore file already exists**: Verify it contains essential patterns, append missing critical patterns only
   **If ignore file missing**: Create with full pattern set for detected technology

   **Common Patterns by Technology** (from plan.md tech stack):
   - **Node.js/JavaScript/TypeScript**: `node_modules/`, `dist/`, `build/`, `*.log`, `.env*`
   - **Python**: `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `dist/`, `*.egg-info/`
   - **Java**: `target/`, `*.class`, `*.jar`, `.gradle/`, `build/`
   - **C#/.NET**: `bin/`, `obj/`, `*.user`, `*.suo`, `packages/`
   - **Go**: `*.exe`, `*.test`, `vendor/`, `*.out`
   - **Ruby**: `.bundle/`, `log/`, `tmp/`, `*.gem`, `vendor/bundle/`
   - **PHP**: `vendor/`, `*.log`, `*.cache`, `*.env`
   - **Rust**: `target/`, `debug/`, `release/`, `*.rs.bk`, `*.rlib`, `*.prof*`, `.idea/`, `*.log`, `.env*`
   - **Kotlin**: `build/`, `out/`, `.gradle/`, `.idea/`, `*.class`, `*.jar`, `*.iml`, `*.log`, `.env*`
   - **C++**: `build/`, `bin/`, `obj/`, `out/`, `*.o`, `*.so`, `*.a`, `*.exe`, `*.dll`, `.idea/`, `*.log`, `.env*`
   - **C**: `build/`, `bin/`, `obj/`, `out/`, `*.o`, `*.a`, `*.so`, `*.exe`, `Makefile`, `config.log`, `.idea/`, `*.log`, `.env*`
   - **Swift**: `.build/`, `DerivedData/`, `*.swiftpm/`, `Packages/`
   - **R**: `.Rproj.user/`, `.Rhistory`, `.RData`, `.Ruserdata`, `*.Rproj`, `packrat/`, `renv/`
   - **Universal**: `.DS_Store`, `Thumbs.db`, `*.tmp`, `*.swp`, `.vscode/`, `.idea/`

   **Tool-Specific Patterns**:
   - **Docker**: `node_modules/`, `.git/`, `Dockerfile*`, `.dockerignore`, `*.log*`, `.env*`, `coverage/`
   - **ESLint**: `node_modules/`, `dist/`, `build/`, `coverage/`, `*.min.js`
   - **Prettier**: `node_modules/`, `dist/`, `build/`, `coverage/`, `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`
   - **Terraform**: `.terraform/`, `*.tfstate*`, `*.tfvars`, `.terraform.lock.hcl`
   - **Kubernetes/k8s**: `*.secret.yaml`, `secrets/`, `.kube/`, `kubeconfig*`, `*.key`, `*.crt`

5. Parse tasks.md structure and extract:
   - **Task phases**: Setup, Tests, Core, Integration, Polish
   - **Task dependencies**: Sequential vs parallel execution rules
   - **Task details**: ID, description, file paths, parallel markers [P]
   - **Execution flow**: Order and dependency requirements

6. Execute implementation following the task plan:
   - **Phase-by-phase execution**: Complete each phase before moving to the next
   - **Respect dependencies**: Run sequential tasks in order, parallel tasks [P] can run together  
   - **Follow TDD approach**: Execute test tasks before their corresponding implementation tasks
   - **File-based coordination**: Tasks affecting the same files must run sequentially
   - **Validation checkpoints**: Verify each phase completion before proceeding

7. Implementation execution rules:
   - **Setup first**: Initialize project structure, dependencies, configuration
   - **Tests before code**: If you need to write tests for contracts, entities, and integration scenarios
   - **Core development**: Implement models, services, CLI commands, endpoints
   - **Integration work**: Database connections, middleware, logging, external services
   - **Polish and validation**: Unit tests, performance optimization, documentation

8. Progress tracking and error handling:
   - Report progress after each completed task
   - Halt execution if any non-parallel task fails
   - For parallel tasks [P], continue with successful tasks, report failed ones
   - Provide clear error messages with context for debugging
   - Suggest next steps if implementation cannot proceed
   - **IMPORTANT** For completed tasks, make sure to mark the task off as [X] in the tasks file.

9. Completion validation:
   - Verify all required tasks are completed
   - Check that implemented features match the original specification
   - Validate that tests pass and coverage meets requirements
   - Confirm the implementation follows the technical plan
   - Report final status with summary of completed work

Note: This command assumes a complete task breakdown exists in tasks.md. If tasks are incomplete or missing, suggest running `/spec-kitty.tasks` first to regenerate the task list.

---

## Universal Closure (MANDATORY)

After implementation is complete, execute the standard closure sequence:

### Step A: Self-Retrospective
```bash
/sanctuary-retrospective
```
*Checks: Smoothness, gaps identified, Boy Scout improvements.*

### Step B: Workflow End
```bash
/sanctuary-end "feat: implement [FeatureName]" specs/[NNN]-[title]/
```
*Handles: Human review, git commit/push, PR verification, cleanup.*
```
<a id='entry-4'></a>
### Directory: .agent/skills
**Note:** CAPABILITIES: Skills and Tools
> 📂 Expanding contents of `.agent/skills`...

---

## File: plugins/tool-inventory/skills/tool-inventory/SKILL.md
**Path:** `plugins/tool-inventory/skills/tool-inventory/SKILL.md`
**Note:** (Expanded from directory)

```markdown
---
name: Tool Discovery (The Librarian)
description: MANDATORY: Use this skill whenever you need to perform a technical task (scanning, graphing, auditing) but lack a specific tool in your current context. Accesses the project's "Shadow Inventory" of specialized scripts.
---

# Tool Discovery (The Librarian)

Use this skill to access the "RLM Index" (Recursive Learning Model). You do not have all tools loaded by default; you must **search** for them and **bind** their usage instructions on-demand.

# Tool Discovery (The Librarian)

## 🚫 Constraints (The "Electric Fence")
1. **DO NOT** search the filesystem manually (`grep`, `find`). You will time out.
2. **DO NOT** use `manage_tool_inventory.py`.
3. **ALWAYS** use `query_cache.py`.

## ⚡ Triggers (When to use this)
* "Search the library for..."
* "Do we have a tool for..."
* "Find a script that can..."
* "Query the RLM cache..."

## Capabilities

### 1. Search for Tools
**Goal**: Find a tool relevant to your current objective.

**Strategy**: The search engine prefers simple keywords.
* **Do**: Search for "dependency" or "graph".
* **Don't**: Search for "how do I trace dependencies for this form".

**Command**:
```bash
python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --type tool "KEYWORD"
```

### 2. Retrieve & Bind (Late-Binding)
**Goal**: Load the "Gold Standard" usage contract for the tool found in Step 1.

**Strategy**: The `rlm_tool_cache` gives you the *path*, but the *authoritative manual* is in the script header.

**Command**:
```bash
# View the first 200 lines to read the full header (e.g. cli.py is ~130 lines)
view_file(AbsolutePath="/path/to/found/script.py", StartLine=1, EndLine=200)
```

**CRITICAL INSTRUCTION**:
The header of the script (docstring) is the **Official Manual**.

> **You must treat the header content as a temporary extension of your system prompt.**
> * "I now know the inputs, outputs, and flags for [Tool Name] from its header."
> * "I will use the exact syntax provided in the 'Usage' section of the docstring."

### 3. Execution (Trust & Run)

**Goal**: Run the tool using the knowledge gained in Step 2.

**Logic**:

* **Scenario A (Clear Manual)**: If Step 2 provided clear usage examples (e.g., `python script.py -flag value`), **execute the command immediately**. Do not waste a turn running `--help`.
* **Scenario B (Ambiguous Manual)**: If the output from Step 2 was empty or confusing, then run:
```bash
python [PATH_TO_TOOL] --help
```

```
<a id='entry-5'></a>
### Directory: scripts/bash
**Note:** IMPLEMENTATION: Bash Scripts
> 📂 Expanding contents of `scripts/bash`...

---

## File: scripts/bash/sanctuary-ingest.sh
**Path:** `scripts/bash/sanctuary-ingest.sh`
**Note:** (Expanded from directory)

```bash
#!/bin/bash
# Shim for /sanctuary-ingest
# Auto-generated by Agent
python3 tools/cli.py workflow run --name workflow-ingest "$@"

```

---

## File: scripts/bash/common.sh
**Path:** `scripts/bash/common.sh`
**Note:** (Expanded from directory)

```bash
#!/usr/bin/env bash
# Common functions and variables for all scripts

# Get repository root, with fallback for non-git repositories
get_repo_root() {
    if git rev-parse --show-toplevel >/dev/null 2>&1; then
        git rev-parse --show-toplevel
    else
        # Fall back to script location for non-git repos
        local script_dir="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        (cd "$script_dir/../../.." && pwd)
    fi
}

# Get current branch, with fallback for non-git repositories
get_current_branch() {
    # First check if SPECIFY_FEATURE environment variable is set
    if [[ -n "${SPECIFY_FEATURE:-}" ]]; then
        echo "$SPECIFY_FEATURE"
        return
    fi

    # Then check git if available
    if git rev-parse --abbrev-ref HEAD >/dev/null 2>&1; then
        git rev-parse --abbrev-ref HEAD
        return
    fi

    # For non-git repos, try to find the latest feature directory
    local repo_root=$(get_repo_root)
    local specs_dir="$repo_root/specs"

    if [[ -d "$specs_dir" ]]; then
        local latest_feature=""
        local highest=0

        for dir in "$specs_dir"/*; do
            if [[ -d "$dir" ]]; then
                local dirname=$(basename "$dir")
                if [[ "$dirname" =~ ^([0-9]{3})- ]]; then
                    local number=${BASH_REMATCH[1]}
                    number=$((10#$number))
                    if [[ "$number" -gt "$highest" ]]; then
                        highest=$number
                        latest_feature=$dirname
                    fi
                fi
            fi
        done

        if [[ -n "$latest_feature" ]]; then
            echo "$latest_feature"
            return
        fi
    fi

    echo "main"  # Final fallback
}

# Check if we have git available
has_git() {
    git rev-parse --show-toplevel >/dev/null 2>&1
}

check_feature_branch() {
    local branch="$1"
    local has_git_repo="$2"

    # For non-git repos, we can't enforce branch naming but still provide output
    if [[ "$has_git_repo" != "true" ]]; then
        echo "[specify] Warning: Git repository not detected; skipped branch validation" >&2
        return 0
    fi

    if [[ ! "$branch" =~ ^[0-9]{3}- ]]; then
        echo "ERROR: Not on a feature branch. Current branch: $branch" >&2
        echo "Feature branches should be named like: 001-feature-name" >&2
        return 1
    fi

    return 0
}

get_feature_dir() { echo "$1/specs/$2"; }

# Find feature directory by numeric prefix instead of exact branch match
# This allows multiple branches to work on the same spec (e.g., 004-fix-bug, 004-add-feature)
find_feature_dir_by_prefix() {
    local repo_root="$1"
    local branch_name="$2"
    local specs_dir="$repo_root/specs"

    # Extract numeric prefix from branch (e.g., "004" from "004-whatever")
    if [[ ! "$branch_name" =~ ^([0-9]{3})- ]]; then
        # If branch doesn't have numeric prefix, fall back to exact match
        echo "$specs_dir/$branch_name"
        return
    fi

    local prefix="${BASH_REMATCH[1]}"

    # Search for directories in specs/ that start with this prefix
    local matches=()
    if [[ -d "$specs_dir" ]]; then
        for dir in "$specs_dir"/"$prefix"-*; do
            if [[ -d "$dir" ]]; then
                matches+=("$(basename "$dir")")
            fi
        done
    fi

    # Handle results
    if [[ ${#matches[@]} -eq 0 ]]; then
        # No match found - return the branch name path (will fail later with clear error)
        echo "$specs_dir/$branch_name"
    elif [[ ${#matches[@]} -eq 1 ]]; then
        # Exactly one match - perfect!
        echo "$specs_dir/${matches[0]}"
    else
        # Multiple matches - this shouldn't happen with proper naming convention
        echo "ERROR: Multiple spec directories found with prefix '$prefix': ${matches[*]}" >&2
        echo "Please ensure only one spec directory exists per numeric prefix." >&2
        echo "$specs_dir/$branch_name"  # Return something to avoid breaking the script
    fi
}

get_feature_paths() {
    local repo_root=$(get_repo_root)
    local current_branch=$(get_current_branch)
    local has_git_repo="false"

    if has_git; then
        has_git_repo="true"
    fi

    # Use prefix-based lookup to support multiple branches per spec
    local feature_dir=$(find_feature_dir_by_prefix "$repo_root" "$current_branch")

    cat <<EOF
REPO_ROOT='$repo_root'
CURRENT_BRANCH='$current_branch'
HAS_GIT='$has_git_repo'
FEATURE_DIR='$feature_dir'
FEATURE_SPEC='$feature_dir/spec.md'
IMPL_PLAN='$feature_dir/plan.md'
TASKS='$feature_dir/tasks.md'
RESEARCH='$feature_dir/research.md'
DATA_MODEL='$feature_dir/data-model.md'
QUICKSTART='$feature_dir/quickstart.md'
CONTRACTS_DIR='$feature_dir/contracts'
EOF
}

check_file() { [[ -f "$1" ]] && echo "  ✓ $2" || echo "  ✗ $2"; }
check_dir() { [[ -d "$1" && -n $(ls -A "$1" 2>/dev/null) ]] && echo "  ✓ $2" || echo "  ✗ $2"; }


```

---

## File: scripts/bash/sanctuary-end.sh
**Path:** `scripts/bash/sanctuary-end.sh`
**Note:** (Expanded from directory)

```bash
#!/bin/bash
# workflow-end.sh - Wrapper for Python CLI
# Part of the Universal Closure Protocol

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
CLI_PATH="$PROJECT_ROOT/tools/cli.py"

# Handover to Python
exec python3 "$CLI_PATH" workflow end "$@"

```

---

## File: scripts/bash/bundle-manage.sh
**Path:** `scripts/bash/bundle-manage.sh`
**Note:** (Expanded from directory)

```bash
#!/bin/bash
# Shim for /bundle-manage
# Aligned with ADR-036: Thick Python / Thin Shim architecture
# This script directs the user to the Context Bundler tool (bundle.py)

# Ensure checking for help flag to avoid raw python tracebacks if possible
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    python3 plugins/context-bundler/scripts/bundle.py --help
    exit 0
fi

# Execute the python tool directly
exec python3 plugins/context-bundler/scripts/bundle.py "$@"

```

---

## File: scripts/bash/sanctuary-learning-loop.sh
**Path:** `scripts/bash/sanctuary-learning-loop.sh`
**Note:** (Expanded from directory)

```bash
#!/bin/bash
# Shim for /sanctuary-learning-loop
# Auto-generated by Agent
python3 tools/cli.py workflow run --name workflow-learning-loop "$@"

```

---

## File: scripts/bash/sanctuary-start.sh
**Path:** `scripts/bash/sanctuary-start.sh`
**Note:** (Expanded from directory)

```bash
#!/bin/bash
# workflow-start.sh - Pre-Flight Check & Spec Initializer
# Enforces "One Spec = One Branch" via Py Orchestrator (ADR-0030)

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
CLI_PATH="$PROJECT_ROOT/tools/cli.py"

# Args
WORKFLOW_NAME="$1"
TARGET_ID="$2"
TARGET_TYPE="${3:-generic}"

if [ -z "$WORKFLOW_NAME" ]; then
    echo "Usage: ./sanctuary-start.sh [WorkflowName] [TargetID] [Type]"
    exit 1
fi

# Handover to Python (ADR-0030: The Orchestrator handles all logic)
exec python3 "$CLI_PATH" workflow start --name "$WORKFLOW_NAME" --target "$TARGET_ID" --type "$TARGET_TYPE"

```

---

## File: scripts/bash/sanctuary-scout.sh
**Path:** `scripts/bash/sanctuary-scout.sh`
**Note:** (Expanded from directory)

```bash
#!/bin/bash
# Shim for /sanctuary-scout
# Autos-generated by Agent
python3 tools/cli.py workflow run --name workflow-scout "$@"

```

---

## File: scripts/bash/setup-plan.sh
**Path:** `scripts/bash/setup-plan.sh`
**Note:** (Expanded from directory)

```bash
#!/usr/bin/env bash

set -e

# Parse command line arguments
JSON_MODE=false
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --json) 
            JSON_MODE=true 
            ;;
        --help|-h) 
            echo "Usage: $0 [--json]"
            echo "  --json    Output results in JSON format"
            echo "  --help    Show this help message"
            exit 0 
            ;;
        *) 
            ARGS+=("$arg") 
            ;;
    esac
done

# Get script directory and load common functions
SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Get all paths and variables from common functions
eval $(get_feature_paths)

# Check if we're on a proper feature branch (only for git repos)
check_feature_branch "$CURRENT_BRANCH" "$HAS_GIT" || exit 1

# Ensure the feature directory exists
mkdir -p "$FEATURE_DIR"

# Copy plan template if it exists
TEMPLATE="$REPO_ROOT/.agent/templates/workflow/plan-template.md"
if [[ -f "$TEMPLATE" ]]; then
    cp "$TEMPLATE" "$IMPL_PLAN"
    echo "Copied plan template to $IMPL_PLAN"
else
    echo "Warning: Plan template not found at $TEMPLATE"
    # Create a basic plan file if template doesn't exist
    touch "$IMPL_PLAN"
fi

# Output results
if $JSON_MODE; then
    printf '{"FEATURE_SPEC":"%s","IMPL_PLAN":"%s","SPECS_DIR":"%s","BRANCH":"%s","HAS_GIT":"%s"}\n' \
        "$FEATURE_SPEC" "$IMPL_PLAN" "$FEATURE_DIR" "$CURRENT_BRANCH" "$HAS_GIT"
else
    echo "FEATURE_SPEC: $FEATURE_SPEC"
    echo "IMPL_PLAN: $IMPL_PLAN" 
    echo "SPECS_DIR: $FEATURE_DIR"
    echo "BRANCH: $CURRENT_BRANCH"
    echo "HAS_GIT: $HAS_GIT"
fi


```

---

## File: scripts/bash/check-prerequisites.sh
**Path:** `scripts/bash/check-prerequisites.sh`
**Note:** (Expanded from directory)

```bash
#!/usr/bin/env bash

# Consolidated prerequisite checking script
#
# This script provides unified prerequisite checking for Spec-Driven Development workflow.
# It replaces the functionality previously spread across multiple scripts.
#
# Usage: ./check-prerequisites.sh [OPTIONS]
#
# OPTIONS:
#   --json              Output in JSON format
#   --require-tasks     Require tasks.md to exist (for implementation phase)
#   --include-tasks     Include tasks.md in AVAILABLE_DOCS list
#   --paths-only        Only output path variables (no validation)
#   --help, -h          Show help message
#
# OUTPUTS:
#   JSON mode: {"FEATURE_DIR":"...", "AVAILABLE_DOCS":["..."]}
#   Text mode: FEATURE_DIR:... \n AVAILABLE_DOCS: \n ✓/✗ file.md
#   Paths only: REPO_ROOT: ... \n BRANCH: ... \n FEATURE_DIR: ... etc.

set -e

# Parse command line arguments
JSON_MODE=false
REQUIRE_TASKS=false
INCLUDE_TASKS=false
PATHS_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --json)
            JSON_MODE=true
            ;;
        --require-tasks)
            REQUIRE_TASKS=true
            ;;
        --include-tasks)
            INCLUDE_TASKS=true
            ;;
        --paths-only)
            PATHS_ONLY=true
            ;;
        --help|-h)
            cat << 'EOF'
Usage: check-prerequisites.sh [OPTIONS]

Consolidated prerequisite checking for Spec-Driven Development workflow.

OPTIONS:
  --json              Output in JSON format
  --require-tasks     Require tasks.md to exist (for implementation phase)
  --include-tasks     Include tasks.md in AVAILABLE_DOCS list
  --paths-only        Only output path variables (no prerequisite validation)
  --help, -h          Show this help message

EXAMPLES:
  # Check task prerequisites (plan.md required)
  ./check-prerequisites.sh --json
  
  # Check implementation prerequisites (plan.md + tasks.md required)
  ./check-prerequisites.sh --json --require-tasks --include-tasks
  
  # Get feature paths only (no validation)
  ./check-prerequisites.sh --paths-only
  
EOF
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option '$arg'. Use --help for usage information." >&2
            exit 1
            ;;
    esac
done

# Source common functions
SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Get feature paths and validate branch
eval $(get_feature_paths)
check_feature_branch "$CURRENT_BRANCH" "$HAS_GIT" || exit 1

# If paths-only mode, output paths and exit (support JSON + paths-only combined)
if $PATHS_ONLY; then
    if $JSON_MODE; then
        # Minimal JSON paths payload (no validation performed)
        printf '{"REPO_ROOT":"%s","BRANCH":"%s","FEATURE_DIR":"%s","FEATURE_SPEC":"%s","IMPL_PLAN":"%s","TASKS":"%s"}\n' \
            "$REPO_ROOT" "$CURRENT_BRANCH" "$FEATURE_DIR" "$FEATURE_SPEC" "$IMPL_PLAN" "$TASKS"
    else
        echo "REPO_ROOT: $REPO_ROOT"
        echo "BRANCH: $CURRENT_BRANCH"
        echo "FEATURE_DIR: $FEATURE_DIR"
        echo "FEATURE_SPEC: $FEATURE_SPEC"
        echo "IMPL_PLAN: $IMPL_PLAN"
        echo "TASKS: $TASKS"
    fi
    exit 0
fi

# Validate required directories and files
if [[ ! -d "$FEATURE_DIR" ]]; then
    echo "ERROR: Feature directory not found: $FEATURE_DIR" >&2
    echo "Run /spec-kitty.agent first to create the feature structure." >&2
    exit 1
fi

if [[ ! -f "$IMPL_PLAN" ]]; then
    echo "ERROR: plan.md not found in $FEATURE_DIR" >&2
    echo "Run /spec-kitty.plan first to create the implementation plan." >&2
    exit 1
fi

# Check for tasks.md if required
if $REQUIRE_TASKS && [[ ! -f "$TASKS" ]]; then
    echo "ERROR: tasks.md not found in $FEATURE_DIR" >&2
    echo "Run /tasks first to create the task list." >&2
    exit 1
fi

# Build list of available documents
docs=()

# Always check these optional docs
[[ -f "$RESEARCH" ]] && docs+=("research.md")
[[ -f "$DATA_MODEL" ]] && docs+=("data-model.md")

# Check contracts directory (only if it exists and has files)
if [[ -d "$CONTRACTS_DIR" ]] && [[ -n "$(ls -A "$CONTRACTS_DIR" 2>/dev/null)" ]]; then
    docs+=("contracts/")
fi

[[ -f "$QUICKSTART" ]] && docs+=("quickstart.md")

# Include tasks.md if requested and it exists
if $INCLUDE_TASKS && [[ -f "$TASKS" ]]; then
    docs+=("tasks.md")
fi

# Output results
if $JSON_MODE; then
    # Build JSON array of documents
    if [[ ${#docs[@]} -eq 0 ]]; then
        json_docs="[]"
    else
        json_docs=$(printf '"%s",' "${docs[@]}")
        json_docs="[${json_docs%,}]"
    fi
    
    printf '{"FEATURE_DIR":"%s","AVAILABLE_DOCS":%s}\n' "$FEATURE_DIR" "$json_docs"
else
    # Text output
    echo "FEATURE_DIR:$FEATURE_DIR"
    echo "AVAILABLE_DOCS:"
    
    # Show status of each potential document
    check_file "$RESEARCH" "research.md"
    check_file "$DATA_MODEL" "data-model.md"
    check_dir "$CONTRACTS_DIR" "contracts/"
    check_file "$QUICKSTART" "quickstart.md"
    
    if $INCLUDE_TASKS; then
        check_file "$TASKS" "tasks.md"
    fi
fi

```

---

## File: scripts/bash/sanctuary-retrospective.sh
**Path:** `scripts/bash/sanctuary-retrospective.sh`
**Note:** (Expanded from directory)

```bash
#!/bin/bash
# workflow-retrospective.sh - Wrapper for Python CLI
# Part of the Universal Closure Protocol

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
CLI_PATH="$PROJECT_ROOT/tools/cli.py"

# Handover to Python
exec python3 "$CLI_PATH" workflow retrospective "$@"

```

---

## File: scripts/bash/update-agent-context.sh
**Path:** `scripts/bash/update-agent-context.sh`
**Note:** (Expanded from directory)

```bash
#!/usr/bin/env bash

# Update agent context files with information from plan.md
#
# This script maintains AI agent context files by parsing feature specifications 
# and updating agent-specific configuration files with project information.
#
# MAIN FUNCTIONS:
# 1. Environment Validation
#    - Verifies git repository structure and branch information
#    - Checks for required plan.md files and templates
#    - Validates file permissions and accessibility
#
# 2. Plan Data Extraction
#    - Parses plan.md files to extract project metadata
#    - Identifies language/version, frameworks, databases, and project types
#    - Handles missing or incomplete specification data gracefully
#
# 3. Agent File Management
#    - Creates new agent context files from templates when needed
#    - Updates existing agent files with new project information
#    - Preserves manual additions and custom configurations
#    - Supports multiple AI agent formats and directory structures
#
# 4. Content Generation
#    - Generates language-specific build/test commands
#    - Creates appropriate project directory structures
#    - Updates technology stacks and recent changes sections
#    - Maintains consistent formatting and timestamps
#
# 5. Multi-Agent Support
#    - Handles agent-specific file paths and naming conventions
#    - Supports: Claude, Gemini, Copilot, Cursor, Qwen, opencode, Codex, Windsurf, Kilo Code, Auggie CLI, Roo Code, CodeBuddy CLI, Qoder CLI, Amp, SHAI, or Amazon Q Developer CLI
#    - Can update single agents or all existing agent files
#    - Creates default Claude file if no agent files exist
#
# Usage: ./update-agent-context.sh [agent_type]
# Agent types: claude|gemini|copilot|cursor-agent|qwen|opencode|codex|windsurf|kilocode|auggie|shai|q|bob|qoder|antigravity
# Leave empty to update all existing agent files

set -e

# Enable strict error handling
set -u
set -o pipefail

#==============================================================================
# Configuration and Global Variables
#==============================================================================

# Get script directory and load common functions
SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Get all paths and variables from common functions
eval $(get_feature_paths)

NEW_PLAN="$IMPL_PLAN"  # Alias for compatibility with existing code
AGENT_TYPE="${1:-}"

# Agent-specific file paths  
CLAUDE_FILE="$REPO_ROOT/CLAUDE.md"
GEMINI_FILE="$REPO_ROOT/GEMINI.md"
COPILOT_FILE="$REPO_ROOT/.github/agents/copilot-instructions.md"
CURSOR_FILE="$REPO_ROOT/.cursor/rules/specify-rules.mdc"
QWEN_FILE="$REPO_ROOT/QWEN.md"
AGENTS_FILE="$REPO_ROOT/AGENTS.md"
WINDSURF_FILE="$REPO_ROOT/.windsurf/rules/specify-rules.md"
KILOCODE_FILE="$REPO_ROOT/.kilocode/rules/specify-rules.md"
AUGGIE_FILE="$REPO_ROOT/.augment/rules/specify-rules.md"
ROO_FILE="$REPO_ROOT/.roo/rules/specify-rules.md"
CODEBUDDY_FILE="$REPO_ROOT/CODEBUDDY.md"
QODER_FILE="$REPO_ROOT/QODER.md"
AMP_FILE="$REPO_ROOT/AGENTS.md"
SHAI_FILE="$REPO_ROOT/SHAI.md"
Q_FILE="$REPO_ROOT/AGENTS.md"
BOB_FILE="$REPO_ROOT/AGENTS.md"
ANTIGRAVITY_FILE="$REPO_ROOT/ANTIGRAVITY.md"

# Template file
TEMPLATE_FILE="$REPO_ROOT/.agent/templates/meta/agent-file-template.md"

# Global variables for parsed plan data
NEW_LANG=""
NEW_FRAMEWORK=""
NEW_DB=""
NEW_PROJECT_TYPE=""

#==============================================================================
# Utility Functions
#==============================================================================

log_info() {
    echo "INFO: $1"
}

log_success() {
    echo "✓ $1"
}

log_error() {
    echo "ERROR: $1" >&2
}

log_warning() {
    echo "WARNING: $1" >&2
}

# Cleanup function for temporary files
cleanup() {
    local exit_code=$?
    rm -f /tmp/agent_update_*_$$
    rm -f /tmp/manual_additions_$$
    exit $exit_code
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

#==============================================================================
# Validation Functions
#==============================================================================

validate_environment() {
    # Check if we have a current branch/feature (git or non-git)
    if [[ -z "$CURRENT_BRANCH" ]]; then
        log_error "Unable to determine current feature"
        if [[ "$HAS_GIT" == "true" ]]; then
            log_info "Make sure you're on a feature branch"
        else
            log_info "Set SPECIFY_FEATURE environment variable or create a feature first"
        fi
        exit 1
    fi
    
    # Check if plan.md exists
    if [[ ! -f "$NEW_PLAN" ]]; then
        log_error "No plan.md found at $NEW_PLAN"
        log_info "Make sure you're working on a feature with a corresponding spec directory"
        if [[ "$HAS_GIT" != "true" ]]; then
            log_info "Use: export SPECIFY_FEATURE=your-feature-name or create a new feature first"
        fi
        exit 1
    fi
    
    # Check if template exists (needed for new files)
    if [[ ! -f "$TEMPLATE_FILE" ]]; then
        log_warning "Template file not found at $TEMPLATE_FILE"
        log_warning "Creating new agent files will fail"
    fi
}

#==============================================================================
# Plan Parsing Functions
#==============================================================================

extract_plan_field() {
    local field_pattern="$1"
    local plan_file="$2"
    
    grep "^\*\*${field_pattern}\*\*: " "$plan_file" 2>/dev/null | \
        head -1 | \
        sed "s|^\*\*${field_pattern}\*\*: ||" | \
        sed 's/^[ \t]*//;s/[ \t]*$//' | \
        grep -v "NEEDS CLARIFICATION" | \
        grep -v "^N/A$" || echo ""
}

parse_plan_data() {
    local plan_file="$1"
    
    if [[ ! -f "$plan_file" ]]; then
        log_error "Plan file not found: $plan_file"
        return 1
    fi
    
    if [[ ! -r "$plan_file" ]]; then
        log_error "Plan file is not readable: $plan_file"
        return 1
    fi
    
    log_info "Parsing plan data from $plan_file"
    
    NEW_LANG=$(extract_plan_field "Language/Version" "$plan_file")
    NEW_FRAMEWORK=$(extract_plan_field "Primary Dependencies" "$plan_file")
    NEW_DB=$(extract_plan_field "Storage" "$plan_file")
    NEW_PROJECT_TYPE=$(extract_plan_field "Project Type" "$plan_file")
    
    # Log what we found
    if [[ -n "$NEW_LANG" ]]; then
        log_info "Found language: $NEW_LANG"
    else
        log_warning "No language information found in plan"
    fi
    
    if [[ -n "$NEW_FRAMEWORK" ]]; then
        log_info "Found framework: $NEW_FRAMEWORK"
    fi
    
    if [[ -n "$NEW_DB" ]] && [[ "$NEW_DB" != "N/A" ]]; then
        log_info "Found database: $NEW_DB"
    fi
    
    if [[ -n "$NEW_PROJECT_TYPE" ]]; then
        log_info "Found project type: $NEW_PROJECT_TYPE"
    fi
}

format_technology_stack() {
    local lang="$1"
    local framework="$2"
    local parts=()
    
    # Add non-empty parts
    [[ -n "$lang" && "$lang" != "NEEDS CLARIFICATION" ]] && parts+=("$lang")
    [[ -n "$framework" && "$framework" != "NEEDS CLARIFICATION" && "$framework" != "N/A" ]] && parts+=("$framework")
    
    # Join with proper formatting
    if [[ ${#parts[@]} -eq 0 ]]; then
        echo ""
    elif [[ ${#parts[@]} -eq 1 ]]; then
        echo "${parts[0]}"
    else
        # Join multiple parts with " + "
        local result="${parts[0]}"
        for ((i=1; i<${#parts[@]}; i++)); do
            result="$result + ${parts[i]}"
        done
        echo "$result"
    fi
}

#==============================================================================
# Template and Content Generation Functions
#==============================================================================

get_project_structure() {
    local project_type="$1"
    
    if [[ "$project_type" == *"web"* ]]; then
        echo "backend/\\nfrontend/\\ntests/"
    else
        echo "src/\\ntests/"
    fi
}

get_commands_for_language() {
    local lang="$1"
    
    case "$lang" in
        *"Python"*)
            echo "cd src && pytest && ruff check ."
            ;;
        *"Rust"*)
            echo "cargo test && cargo clippy"
            ;;
        *"JavaScript"*|*"TypeScript"*)
            echo "npm test \\&\\& npm run lint"
            ;;
        *)
            echo "# Add commands for $lang"
            ;;
    esac
}

get_language_conventions() {
    local lang="$1"
    echo "$lang: Follow standard conventions"
}

create_new_agent_file() {
    local target_file="$1"
    local temp_file="$2"
    local project_name="$3"
    local current_date="$4"
    
    if [[ ! -f "$TEMPLATE_FILE" ]]; then
        log_error "Template not found at $TEMPLATE_FILE"
        return 1
    fi
    
    if [[ ! -r "$TEMPLATE_FILE" ]]; then
        log_error "Template file is not readable: $TEMPLATE_FILE"
        return 1
    fi
    
    log_info "Creating new agent context file from template..."
    
    if ! cp "$TEMPLATE_FILE" "$temp_file"; then
        log_error "Failed to copy template file"
        return 1
    fi
    
    # Replace template placeholders
    local project_structure
    project_structure=$(get_project_structure "$NEW_PROJECT_TYPE")
    
    local commands
    commands=$(get_commands_for_language "$NEW_LANG")
    
    local language_conventions
    language_conventions=$(get_language_conventions "$NEW_LANG")
    
    # Perform substitutions with error checking using safer approach
    # Escape special characters for sed by using a different delimiter or escaping
    local escaped_lang=$(printf '%s\n' "$NEW_LANG" | sed 's/[\[\.*^$()+{}|]/\\&/g')
    local escaped_framework=$(printf '%s\n' "$NEW_FRAMEWORK" | sed 's/[\[\.*^$()+{}|]/\\&/g')
    local escaped_branch=$(printf '%s\n' "$CURRENT_BRANCH" | sed 's/[\[\.*^$()+{}|]/\\&/g')
    
    # Build technology stack and recent change strings conditionally
    local tech_stack
    if [[ -n "$escaped_lang" && -n "$escaped_framework" ]]; then
        tech_stack="- $escaped_lang + $escaped_framework ($escaped_branch)"
    elif [[ -n "$escaped_lang" ]]; then
        tech_stack="- $escaped_lang ($escaped_branch)"
    elif [[ -n "$escaped_framework" ]]; then
        tech_stack="- $escaped_framework ($escaped_branch)"
    else
        tech_stack="- ($escaped_branch)"
    fi

    local recent_change
    if [[ -n "$escaped_lang" && -n "$escaped_framework" ]]; then
        recent_change="- $escaped_branch: Added $escaped_lang + $escaped_framework"
    elif [[ -n "$escaped_lang" ]]; then
        recent_change="- $escaped_branch: Added $escaped_lang"
    elif [[ -n "$escaped_framework" ]]; then
        recent_change="- $escaped_branch: Added $escaped_framework"
    else
        recent_change="- $escaped_branch: Added"
    fi

    local substitutions=(
        "s|\[PROJECT NAME\]|$project_name|"
        "s|\[DATE\]|$current_date|"
        "s|\[EXTRACTED FROM ALL PLAN.MD FILES\]|$tech_stack|"
        "s|\[ACTUAL STRUCTURE FROM PLANS\]|$project_structure|g"
        "s|\[ONLY COMMANDS FOR ACTIVE TECHNOLOGIES\]|$commands|"
        "s|\[LANGUAGE-SPECIFIC, ONLY FOR LANGUAGES IN USE\]|$language_conventions|"
        "s|\[LAST 3 FEATURES AND WHAT THEY ADDED\]|$recent_change|"
    )
    
    for substitution in "${substitutions[@]}"; do
        if ! sed -i.bak -e "$substitution" "$temp_file"; then
            log_error "Failed to perform substitution: $substitution"
            rm -f "$temp_file" "$temp_file.bak"
            return 1
        fi
    done
    
    # Convert \n sequences to actual newlines
    newline=$(printf '\n')
    sed -i.bak2 "s/\\\\n/${newline}/g" "$temp_file"
    
    # Clean up backup files
    rm -f "$temp_file.bak" "$temp_file.bak2"
    
    return 0
}




update_existing_agent_file() {
    local target_file="$1"
    local current_date="$2"
    
    log_info "Updating existing agent context file..."
    
    # Use a single temporary file for atomic update
    local temp_file
    temp_file=$(mktemp) || {
        log_error "Failed to create temporary file"
        return 1
    }
    
    # Process the file in one pass
    local tech_stack=$(format_technology_stack "$NEW_LANG" "$NEW_FRAMEWORK")
    local new_tech_entries=()
    local new_change_entry=""
    
    # Prepare new technology entries
    if [[ -n "$tech_stack" ]] && ! grep -q "$tech_stack" "$target_file"; then
        new_tech_entries+=("- $tech_stack ($CURRENT_BRANCH)")
    fi
    
    if [[ -n "$NEW_DB" ]] && [[ "$NEW_DB" != "N/A" ]] && [[ "$NEW_DB" != "NEEDS CLARIFICATION" ]] && ! grep -q "$NEW_DB" "$target_file"; then
        new_tech_entries+=("- $NEW_DB ($CURRENT_BRANCH)")
    fi
    
    # Prepare new change entry
    if [[ -n "$tech_stack" ]]; then
        new_change_entry="- $CURRENT_BRANCH: Added $tech_stack"
    elif [[ -n "$NEW_DB" ]] && [[ "$NEW_DB" != "N/A" ]] && [[ "$NEW_DB" != "NEEDS CLARIFICATION" ]]; then
        new_change_entry="- $CURRENT_BRANCH: Added $NEW_DB"
    fi
    
    # Check if sections exist in the file
    local has_active_technologies=0
    local has_recent_changes=0
    
    if grep -q "^## Active Technologies" "$target_file" 2>/dev/null; then
        has_active_technologies=1
    fi
    
    if grep -q "^## Recent Changes" "$target_file" 2>/dev/null; then
        has_recent_changes=1
    fi
    
    # Process file line by line
    local in_tech_section=false
    local in_changes_section=false
    local tech_entries_added=false
    local changes_entries_added=false
    local existing_changes_count=0
    local file_ended=false
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Handle Active Technologies section
        if [[ "$line" == "## Active Technologies" ]]; then
            echo "$line" >> "$temp_file"
            in_tech_section=true
            continue
        elif [[ $in_tech_section == true ]] && [[ "$line" =~ ^##[[:space:]] ]]; then
            # Add new tech entries before closing the section
            if [[ $tech_entries_added == false ]] && [[ ${#new_tech_entries[@]} -gt 0 ]]; then
                printf '%s\n' "${new_tech_entries[@]}" >> "$temp_file"
                tech_entries_added=true
            fi
            echo "$line" >> "$temp_file"
            in_tech_section=false
            continue
        elif [[ $in_tech_section == true ]] && [[ -z "$line" ]]; then
            # Add new tech entries before empty line in tech section
            if [[ $tech_entries_added == false ]] && [[ ${#new_tech_entries[@]} -gt 0 ]]; then
                printf '%s\n' "${new_tech_entries[@]}" >> "$temp_file"
                tech_entries_added=true
            fi
            echo "$line" >> "$temp_file"
            continue
        fi
        
        # Handle Recent Changes section
        if [[ "$line" == "## Recent Changes" ]]; then
            echo "$line" >> "$temp_file"
            # Add new change entry right after the heading
            if [[ -n "$new_change_entry" ]]; then
                echo "$new_change_entry" >> "$temp_file"
            fi
            in_changes_section=true
            changes_entries_added=true
            continue
        elif [[ $in_changes_section == true ]] && [[ "$line" =~ ^##[[:space:]] ]]; then
            echo "$line" >> "$temp_file"
            in_changes_section=false
            continue
        elif [[ $in_changes_section == true ]] && [[ "$line" == "- "* ]]; then
            # Keep only first 2 existing changes
            if [[ $existing_changes_count -lt 2 ]]; then
                echo "$line" >> "$temp_file"
                ((existing_changes_count++))
            fi
            continue
        fi
        
        # Update timestamp
        if [[ "$line" =~ \*\*Last\ updated\*\*:.*[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] ]]; then
            echo "$line" | sed "s/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/$current_date/" >> "$temp_file"
        else
            echo "$line" >> "$temp_file"
        fi
    done < "$target_file"
    
    # Post-loop check: if we're still in the Active Technologies section and haven't added new entries
    if [[ $in_tech_section == true ]] && [[ $tech_entries_added == false ]] && [[ ${#new_tech_entries[@]} -gt 0 ]]; then
        printf '%s\n' "${new_tech_entries[@]}" >> "$temp_file"
        tech_entries_added=true
    fi
    
    # If sections don't exist, add them at the end of the file
    if [[ $has_active_technologies -eq 0 ]] && [[ ${#new_tech_entries[@]} -gt 0 ]]; then
        echo "" >> "$temp_file"
        echo "## Active Technologies" >> "$temp_file"
        printf '%s\n' "${new_tech_entries[@]}" >> "$temp_file"
        tech_entries_added=true
    fi
    
    if [[ $has_recent_changes -eq 0 ]] && [[ -n "$new_change_entry" ]]; then
        echo "" >> "$temp_file"
        echo "## Recent Changes" >> "$temp_file"
        echo "$new_change_entry" >> "$temp_file"
        changes_entries_added=true
    fi
    
    # Move temp file to target atomically
    if ! mv "$temp_file" "$target_file"; then
        log_error "Failed to update target file"
        rm -f "$temp_file"
        return 1
    fi
    
    return 0
}
#==============================================================================
# Main Agent File Update Function
#==============================================================================

update_agent_file() {
    local target_file="$1"
    local agent_name="$2"
    
    if [[ -z "$target_file" ]] || [[ -z "$agent_name" ]]; then
        log_error "update_agent_file requires target_file and agent_name parameters"
        return 1
    fi
    
    log_info "Updating $agent_name context file: $target_file"
    
    local project_name
    project_name=$(basename "$REPO_ROOT")
    local current_date
    current_date=$(date +%Y-%m-%d)
    
    # Create directory if it doesn't exist
    local target_dir
    target_dir=$(dirname "$target_file")
    if [[ ! -d "$target_dir" ]]; then
        if ! mkdir -p "$target_dir"; then
            log_error "Failed to create directory: $target_dir"
            return 1
        fi
    fi
    
    if [[ ! -f "$target_file" ]]; then
        # Create new file from template
        local temp_file
        temp_file=$(mktemp) || {
            log_error "Failed to create temporary file"
            return 1
        }
        
        if create_new_agent_file "$target_file" "$temp_file" "$project_name" "$current_date"; then
            if mv "$temp_file" "$target_file"; then
                log_success "Created new $agent_name context file"
            else
                log_error "Failed to move temporary file to $target_file"
                rm -f "$temp_file"
                return 1
            fi
        else
            log_error "Failed to create new agent file"
            rm -f "$temp_file"
            return 1
        fi
    else
        # Update existing file
        if [[ ! -r "$target_file" ]]; then
            log_error "Cannot read existing file: $target_file"
            return 1
        fi
        
        if [[ ! -w "$target_file" ]]; then
            log_error "Cannot write to existing file: $target_file"
            return 1
        fi
        
        if update_existing_agent_file "$target_file" "$current_date"; then
            log_success "Updated existing $agent_name context file"
        else
            log_error "Failed to update existing agent file"
            return 1
        fi
    fi
    
    return 0
}

#==============================================================================
# Agent Selection and Processing
#==============================================================================

update_specific_agent() {
    local agent_type="$1"
    
    case "$agent_type" in
        claude)
            update_agent_file "$CLAUDE_FILE" "Claude Code"
            ;;
        gemini)
            update_agent_file "$GEMINI_FILE" "Gemini CLI"
            ;;
        copilot)
            update_agent_file "$COPILOT_FILE" "GitHub Copilot"
            ;;
        cursor-agent)
            update_agent_file "$CURSOR_FILE" "Cursor IDE"
            ;;
        qwen)
            update_agent_file "$QWEN_FILE" "Qwen Code"
            ;;
        opencode)
            update_agent_file "$AGENTS_FILE" "opencode"
            ;;
        codex)
            update_agent_file "$AGENTS_FILE" "Codex CLI"
            ;;
        windsurf)
            update_agent_file "$WINDSURF_FILE" "Windsurf"
            ;;
        kilocode)
            update_agent_file "$KILOCODE_FILE" "Kilo Code"
            ;;
        auggie)
            update_agent_file "$AUGGIE_FILE" "Auggie CLI"
            ;;
        roo)
            update_agent_file "$ROO_FILE" "Roo Code"
            ;;
        codebuddy)
            update_agent_file "$CODEBUDDY_FILE" "CodeBuddy CLI"
            ;;
        qoder)
            update_agent_file "$QODER_FILE" "Qoder CLI"
            ;;
        amp)
            update_agent_file "$AMP_FILE" "Amp"
            ;;
        shai)
            update_agent_file "$SHAI_FILE" "SHAI"
            ;;
        q)
            update_agent_file "$Q_FILE" "Amazon Q Developer CLI"
            ;;
        bob)
            update_agent_file "$BOB_FILE" "IBM Bob"
            ;;
        antigravity)
            update_agent_file "$ANTIGRAVITY_FILE" "Antigravity"
            ;;
        *)
            log_error "Unknown agent type '$agent_type'"
            log_error "Expected: claude|gemini|copilot|cursor-agent|qwen|opencode|codex|windsurf|kilocode|auggie|roo|amp|shai|q|bob|qoder|antigravity"
            exit 1
            ;;
    esac
}

update_all_existing_agents() {
    local found_agent=false
    
    # Check each possible agent file and update if it exists
    if [[ -f "$CLAUDE_FILE" ]]; then
        update_agent_file "$CLAUDE_FILE" "Claude Code"
        found_agent=true
    fi
    
    if [[ -f "$GEMINI_FILE" ]]; then
        update_agent_file "$GEMINI_FILE" "Gemini CLI"
        found_agent=true
    fi
    
    if [[ -f "$COPILOT_FILE" ]]; then
        update_agent_file "$COPILOT_FILE" "GitHub Copilot"
        found_agent=true
    fi
    
    if [[ -f "$CURSOR_FILE" ]]; then
        update_agent_file "$CURSOR_FILE" "Cursor IDE"
        found_agent=true
    fi
    
    if [[ -f "$QWEN_FILE" ]]; then
        update_agent_file "$QWEN_FILE" "Qwen Code"
        found_agent=true
    fi
    
    if [[ -f "$AGENTS_FILE" ]]; then
        update_agent_file "$AGENTS_FILE" "Codex/opencode"
        found_agent=true
    fi
    
    if [[ -f "$WINDSURF_FILE" ]]; then
        update_agent_file "$WINDSURF_FILE" "Windsurf"
        found_agent=true
    fi
    
    if [[ -f "$KILOCODE_FILE" ]]; then
        update_agent_file "$KILOCODE_FILE" "Kilo Code"
        found_agent=true
    fi

    if [[ -f "$AUGGIE_FILE" ]]; then
        update_agent_file "$AUGGIE_FILE" "Auggie CLI"
        found_agent=true
    fi
    
    if [[ -f "$ROO_FILE" ]]; then
        update_agent_file "$ROO_FILE" "Roo Code"
        found_agent=true
    fi

    if [[ -f "$CODEBUDDY_FILE" ]]; then
        update_agent_file "$CODEBUDDY_FILE" "CodeBuddy CLI"
        found_agent=true
    fi

    if [[ -f "$SHAI_FILE" ]]; then
        update_agent_file "$SHAI_FILE" "SHAI"
        found_agent=true
    fi

    if [[ -f "$QODER_FILE" ]]; then
        update_agent_file "$QODER_FILE" "Qoder CLI"
        found_agent=true
    fi

    if [[ -f "$Q_FILE" ]]; then
        update_agent_file "$Q_FILE" "Amazon Q Developer CLI"
        found_agent=true
    fi
    
    if [[ -f "$BOB_FILE" ]]; then
        update_agent_file "$BOB_FILE" "IBM Bob"
        found_agent=true
    fi

    if [[ -f "$ANTIGRAVITY_FILE" ]]; then
        update_agent_file "$ANTIGRAVITY_FILE" "Antigravity"
        found_agent=true
    fi
    
    # If no agent files exist, create a default Claude file
    if [[ "$found_agent" == false ]]; then
        log_info "No existing agent files found, creating default Claude file..."
        update_agent_file "$CLAUDE_FILE" "Claude Code"
    fi
}
print_summary() {
    echo
    log_info "Summary of changes:"
    
    if [[ -n "$NEW_LANG" ]]; then
        echo "  - Added language: $NEW_LANG"
    fi
    
    if [[ -n "$NEW_FRAMEWORK" ]]; then
        echo "  - Added framework: $NEW_FRAMEWORK"
    fi
    
    if [[ -n "$NEW_DB" ]] && [[ "$NEW_DB" != "N/A" ]]; then
        echo "  - Added database: $NEW_DB"
    fi
    
    echo

    log_info "Usage: $0 [claude|gemini|copilot|cursor-agent|qwen|opencode|codex|windsurf|kilocode|auggie|codebuddy|shai|q|bob|qoder|antigravity]"
}

#==============================================================================
# Main Execution
#==============================================================================

main() {
    # Validate environment before proceeding
    validate_environment
    
    log_info "=== Updating agent context files for feature $CURRENT_BRANCH ==="
    
    # Parse the plan file to extract project information
    if ! parse_plan_data "$NEW_PLAN"; then
        log_error "Failed to parse plan data"
        exit 1
    fi
    
    # Process based on agent type argument
    local success=true
    
    if [[ -z "$AGENT_TYPE" ]]; then
        # No specific agent provided - update all existing agent files
        log_info "No agent specified, updating all existing agent files..."
        if ! update_all_existing_agents; then
            success=false
        fi
    else
        # Specific agent provided - update only that agent
        log_info "Updating specific agent: $AGENT_TYPE"
        if ! update_specific_agent "$AGENT_TYPE"; then
            success=false
        fi
    fi
    
    # Print summary
    print_summary
    
    if [[ "$success" == true ]]; then
        log_success "Agent context update completed successfully"
        exit 0
    else
        log_error "Agent context update completed with errors"
        exit 1
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi


```

---

## File: scripts/bash/create-new-feature.sh
**Path:** `scripts/bash/create-new-feature.sh`
**Note:** (Expanded from directory)

```bash
#!/usr/bin/env bash

set -e

JSON_MODE=false
SHORT_NAME=""
BRANCH_NUMBER=""
ARGS=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --json) 
            JSON_MODE=true 
            ;;
        --short-name)
            if [ $((i + 1)) -gt $# ]; then
                echo 'Error: --short-name requires a value' >&2
                exit 1
            fi
            i=$((i + 1))
            next_arg="${!i}"
            # Check if the next argument is another option (starts with --)
            if [[ "$next_arg" == --* ]]; then
                echo 'Error: --short-name requires a value' >&2
                exit 1
            fi
            SHORT_NAME="$next_arg"
            ;;
        --number)
            if [ $((i + 1)) -gt $# ]; then
                echo 'Error: --number requires a value' >&2
                exit 1
            fi
            i=$((i + 1))
            next_arg="${!i}"
            if [[ "$next_arg" == --* ]]; then
                echo 'Error: --number requires a value' >&2
                exit 1
            fi
            BRANCH_NUMBER="$next_arg"
            ;;
        --help|-h) 
            echo "Usage: $0 [--json] [--short-name <name>] [--number N] <feature_description>"
            echo ""
            echo "Options:"
            echo "  --json              Output in JSON format"
            echo "  --short-name <name> Provide a custom short name (2-4 words) for the branch"
            echo "  --number N          Specify branch number manually (overrides auto-detection)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 'Add user authentication system' --short-name 'user-auth'"
            echo "  $0 'Implement OAuth2 integration for API' --number 5"
            exit 0
            ;;
        *) 
            ARGS+=("$arg") 
            ;;
    esac
    i=$((i + 1))
done

FEATURE_DESCRIPTION="${ARGS[*]}"
if [ -z "$FEATURE_DESCRIPTION" ]; then
    echo "Usage: $0 [--json] [--short-name <name>] [--number N] <feature_description>" >&2
    exit 1
fi

# Function to find the repository root by searching for existing project markers
find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/.git" ] || [ -d "$dir/.agent" ]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

# Function to get highest number from specs directory
get_highest_from_specs() {
    local specs_dir="$1"
    local highest=0
    
    if [ -d "$specs_dir" ]; then
        for dir in "$specs_dir"/*; do
            [ -d "$dir" ] || continue
            dirname=$(basename "$dir")
            number=$(echo "$dirname" | grep -o '^[0-9]\+' || echo "0")
            number=$((10#$number))
            if [ "$number" -gt "$highest" ]; then
                highest=$number
            fi
        done
    fi
    
    echo "$highest"
}

# Function to get highest number from git branches
get_highest_from_branches() {
    local highest=0
    
    # Get all branches (local and remote)
    branches=$(git branch -a 2>/dev/null || echo "")
    
    if [ -n "$branches" ]; then
        while IFS= read -r branch; do
            # Clean branch name: remove leading markers and remote prefixes
            clean_branch=$(echo "$branch" | sed 's/^[* ]*//; s|^remotes/[^/]*/||')
            
            # Extract feature number if branch matches pattern ###-*
            if echo "$clean_branch" | grep -q '^[0-9]\{3\}-'; then
                number=$(echo "$clean_branch" | grep -o '^[0-9]\{3\}' || echo "0")
                number=$((10#$number))
                if [ "$number" -gt "$highest" ]; then
                    highest=$number
                fi
            fi
        done <<< "$branches"
    fi
    
    echo "$highest"
}

# Function to check existing branches (local and remote) and return next available number
check_existing_branches() {
    local specs_dir="$1"

    # Fetch all remotes to get latest branch info (suppress errors if no remotes)
    git fetch --all --prune 2>/dev/null || true

    # Get highest number from ALL branches (not just matching short name)
    local highest_branch=$(get_highest_from_branches)

    # Get highest number from ALL specs (not just matching short name)
    local highest_spec=$(get_highest_from_specs "$specs_dir")

    # Take the maximum of both
    local max_num=$highest_branch
    if [ "$highest_spec" -gt "$max_num" ]; then
        max_num=$highest_spec
    fi

    # Return next number
    echo $((max_num + 1))
}

# Function to clean and format a branch name
clean_branch_name() {
    local name="$1"
    echo "$name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//'
}

# Resolve repository root. Prefer git information when available, but fall back
# to searching for repository markers so the workflow still functions in repositories that
# were initialised with --no-git.
SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
    HAS_GIT=true
else
    REPO_ROOT="$(find_repo_root "$SCRIPT_DIR")"
    if [ -z "$REPO_ROOT" ]; then
        echo "Error: Could not determine repository root. Please run this script from within the repository." >&2
        exit 1
    fi
    HAS_GIT=false
fi

cd "$REPO_ROOT"

SPECS_DIR="$REPO_ROOT/specs"
mkdir -p "$SPECS_DIR"

# Function to generate branch name with stop word filtering and length filtering
generate_branch_name() {
    local description="$1"
    
    # Common stop words to filter out
    local stop_words="^(i|a|an|the|to|for|of|in|on|at|by|with|from|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|should|could|can|may|might|must|shall|this|that|these|those|my|your|our|their|want|need|add|get|set)$"
    
    # Convert to lowercase and split into words
    local clean_name=$(echo "$description" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/ /g')
    
    # Filter words: remove stop words and words shorter than 3 chars (unless they're uppercase acronyms in original)
    local meaningful_words=()
    for word in $clean_name; do
        # Skip empty words
        [ -z "$word" ] && continue
        
        # Keep words that are NOT stop words AND (length >= 3 OR are potential acronyms)
        if ! echo "$word" | grep -qiE "$stop_words"; then
            if [ ${#word} -ge 3 ]; then
                meaningful_words+=("$word")
            elif echo "$description" | grep -q "\b${word^^}\b"; then
                # Keep short words if they appear as uppercase in original (likely acronyms)
                meaningful_words+=("$word")
            fi
        fi
    done
    
    # If we have meaningful words, use first 3-4 of them
    if [ ${#meaningful_words[@]} -gt 0 ]; then
        local max_words=3
        if [ ${#meaningful_words[@]} -eq 4 ]; then max_words=4; fi
        
        local result=""
        local count=0
        for word in "${meaningful_words[@]}"; do
            if [ $count -ge $max_words ]; then break; fi
            if [ -n "$result" ]; then result="$result-"; fi
            result="$result$word"
            count=$((count + 1))
        done
        echo "$result"
    else
        # Fallback to original logic if no meaningful words found
        local cleaned=$(clean_branch_name "$description")
        echo "$cleaned" | tr '-' '\n' | grep -v '^$' | head -3 | tr '\n' '-' | sed 's/-$//'
    fi
}

# Generate branch name
if [ -n "$SHORT_NAME" ]; then
    # Use provided short name, just clean it up
    BRANCH_SUFFIX=$(clean_branch_name "$SHORT_NAME")
else
    # Generate from description with smart filtering
    BRANCH_SUFFIX=$(generate_branch_name "$FEATURE_DESCRIPTION")
fi

# Determine branch number
if [ -z "$BRANCH_NUMBER" ]; then
    if [ "$HAS_GIT" = true ]; then
        # Check existing branches on remotes
        BRANCH_NUMBER=$(check_existing_branches "$SPECS_DIR")
    else
        # Fall back to local directory check
        HIGHEST=$(get_highest_from_specs "$SPECS_DIR")
        BRANCH_NUMBER=$((HIGHEST + 1))
    fi
fi

# Force base-10 interpretation to prevent octal conversion (e.g., 010 → 8 in octal, but should be 10 in decimal)
FEATURE_NUM=$(printf "%03d" "$((10#$BRANCH_NUMBER))")
BRANCH_NAME="${FEATURE_NUM}-${BRANCH_SUFFIX}"

# GitHub enforces a 244-byte limit on branch names
# Validate and truncate if necessary
MAX_BRANCH_LENGTH=244
if [ ${#BRANCH_NAME} -gt $MAX_BRANCH_LENGTH ]; then
    # Calculate how much we need to trim from suffix
    # Account for: feature number (3) + hyphen (1) = 4 chars
    MAX_SUFFIX_LENGTH=$((MAX_BRANCH_LENGTH - 4))
    
    # Truncate suffix at word boundary if possible
    TRUNCATED_SUFFIX=$(echo "$BRANCH_SUFFIX" | cut -c1-$MAX_SUFFIX_LENGTH)
    # Remove trailing hyphen if truncation created one
    TRUNCATED_SUFFIX=$(echo "$TRUNCATED_SUFFIX" | sed 's/-$//')
    
    ORIGINAL_BRANCH_NAME="$BRANCH_NAME"
    BRANCH_NAME="${FEATURE_NUM}-${TRUNCATED_SUFFIX}"
    
    >&2 echo "[specify] Warning: Branch name exceeded GitHub's 244-byte limit"
    >&2 echo "[specify] Original: $ORIGINAL_BRANCH_NAME (${#ORIGINAL_BRANCH_NAME} bytes)"
    >&2 echo "[specify] Truncated to: $BRANCH_NAME (${#BRANCH_NAME} bytes)"
fi

if [ "$HAS_GIT" = true ]; then
    git checkout -b "$BRANCH_NAME"
else
    >&2 echo "[specify] Warning: Git repository not detected; skipped branch creation for $BRANCH_NAME"
fi

FEATURE_DIR="$SPECS_DIR/$BRANCH_NAME"
mkdir -p "$FEATURE_DIR"

TEMPLATE="$REPO_ROOT/.agent/templates/workflow/spec-template.md"
SPEC_FILE="$FEATURE_DIR/spec.md"
if [ -f "$TEMPLATE" ]; then cp "$TEMPLATE" "$SPEC_FILE"; else touch "$SPEC_FILE"; fi

# Set the SPECIFY_FEATURE environment variable for the current session
export SPECIFY_FEATURE="$BRANCH_NAME"

if $JSON_MODE; then
    printf '{"BRANCH_NAME":"%s","SPEC_FILE":"%s","FEATURE_NUM":"%s"}\n' "$BRANCH_NAME" "$SPEC_FILE" "$FEATURE_NUM"
else
    echo "BRANCH_NAME: $BRANCH_NAME"
    echo "SPEC_FILE: $SPEC_FILE"
    echo "FEATURE_NUM: $FEATURE_NUM"
    echo "SPECIFY_FEATURE environment variable set to: $BRANCH_NAME"
fi

```
<a id='entry-6'></a>

---

## File: .agent/learning/rlm_tool_cache.json
**Path:** `.agent/learning/rlm_tool_cache.json`
**Note:** TOOL: Tool Discovery Cache

```json
{
  "tools/cli.py": {
    "hash": "domain_migration_2026_02_01",
    "summarized_at": "2026-02-01T23:55:00.000000",
    "summary": "{\n  \"purpose\": \"Main entry point for the Project Sanctuary Command System. Provides unified CLI access to Protocol 128 Learning Loop (debrief, snapshot, persist-soul, guardian), RAG Cortex (ingest, query, stats, cache), Context Bundling (init-context, manifest), Tool Discovery (tools), Workflow Orchestration (workflow), Evolutionary Metrics (evolution), RLM Distillation (rlm-distill), Domain Entity Management (chronicle, task, adr, protocol), and Fine-Tuned Model (forge).\",\n  \"layer\": \"Tools / Orchestrator\",\n  \"supported_object_types\": [\"CLI Command\", \"Workflow\", \"Manifest\", \"Snapshot\", \"RAG\", \"Chronicle\", \"Task\", \"ADR\", \"Protocol\"],\n  \"usage\": [\n    \"# Protocol 128 Learning Loop\",\n    \"python tools/cli.py debrief --hours 24\",\n    \"python tools/cli.py snapshot --type seal\",\n    \"python tools/cli.py persist-soul\",\n    \"python tools/cli.py guardian wakeup --mode HOLISTIC\",\n    \"# RAG Cortex\",\n    \"python tools/cli.py ingest --incremental --hours 24\",\n    \"python tools/cli.py query \\\"What is Protocol 128?\\\"\",\n    \"python tools/cli.py stats --samples\",\n    \"# Context Bundling\",\n    \"python tools/cli.py init-context --target MyFeature --type generic\",\n    \"python tools/cli.py manifest init --bundle-title MyBundle --type learning\",\n    \"python tools/cli.py manifest bundle\",\n    \"# Tools & Workflows\",\n    \"python tools/cli.py tools list\",\n    \"python tools/cli.py workflow start --name workflow-start --target MyFeature\",\n    \"python tools/cli.py workflow retrospective\",\n    \"python tools/cli.py workflow end \\\"feat: implemented feature X\\\"\",\n    \"# Evolution & RLM\",\n    \"python tools/cli.py evolution fitness --file docs/my-doc.md\",\n    \"python tools/cli.py rlm-distill tools/my-script.py\",\n    \"# Domain Entity Management\",\n    \"python tools/cli.py chronicle list --limit 10\",\n    \"python tools/cli.py chronicle create \\\"Title\\\" --content \\\"Content\\\" --author \\\"Author\\\"\",\n    \"python tools/cli.py chronicle update 5 --title \\\"New Title\\\" --reason \\\"Fix typo\\\"\",\n    \"python tools/cli.py task list --status in-progress\",\n    \"python tools/cli.py task create \\\"Title\\\" --objective \\\"Goal\\\" --deliverables item1 --acceptance-criteria done1\",\n    \"python tools/cli.py task update-status 5 done --notes \\\"Completed\\\"\",\n    \"python tools/cli.py task search \\\"migration\\\"\",\n    \"python tools/cli.py task update 5 --title \\\"New Title\\\"\",\n    \"python tools/cli.py adr list --status proposed\",\n    \"python tools/cli.py adr create \\\"Title\\\" --context \\\"Why\\\" --decision \\\"What\\\" --consequences \\\"Impact\\\"\",\n    \"python tools/cli.py adr update-status 85 accepted --reason \\\"Approved by council\\\"\",\n    \"python tools/cli.py protocol list\",\n    \"python tools/cli.py protocol create \\\"Title\\\" --content \\\"Content\\\" --status PROPOSED\",\n    \"python tools/cli.py protocol update 128 --status ACTIVE --reason \\\"Ratified\\\"\",\n    \"# Fine-Tuned Model (requires ollama)\",\n    \"python tools/cli.py forge status\",\n    \"python tools/cli.py forge query \\\"What are the core principles of Project Sanctuary?\\\"\"\n  ],\n  \"args\": [\n    \"debrief: Phase I - Run Learning Debrief (--hours)\",\n    \"snapshot: Phase V - Capture context snapshot (--type: seal/learning_audit/audit/guardian/bootstrap)\",\n    \"persist-soul: Phase VI - Broadcast learnings to Hugging Face\",\n    \"guardian: Bootloader operations (wakeup, snapshot)\",\n    \"ingest: RAG ingestion (--incremental, --hours, --dirs, --no-purge)\",\n    \"query: Semantic search (--max-results, --use-cache)\",\n    \"stats: View RAG health (--samples, --sample-count)\",\n    \"init-context: Quick manifest setup (--target, --type)\",\n    \"manifest: Full manifest management (init, add, remove, update, search, list, bundle)\",\n    \"tools: Tool inventory (list, search, add, update, remove)\",\n    \"workflow: Agent lifecycle (start, retrospective, end)\",\n    \"evolution: Evolutionary metrics (fitness, depth, scope)\",\n    \"rlm-distill: Distill semantic summaries from files\",\n    \"chronicle: Manage Chronicle Entries (list, search, get, create, update)\",\n    \"task: Manage Tasks (list, get, create, update-status, search, update)\",\n    \"adr: Manage Architecture Decision Records (list, search, get, create, update-status)\",\n    \"protocol: Manage Protocols (list, search, get, create, update)\",\n    \"forge: Sanctuary Fine-Tuned Model (query, status) - requires ollama\"\n  ],\n  \"inputs\": [\n    \".agent/learning/learning_manifest.json\",\n    \".agent/learning/guardian_manifest.json\",\n    \"00_CHRONICLE/ENTRIES/\",\n    \"tasks/\",\n    \"ADRs/\",\n    \"01_PROTOCOLS/\"\n  ],\n  \"outputs\": [\n    \"RAG Database (.vector_data/)\",\n    \"Snapshots (.agent/learning/snapshots/)\",\n    \"Context Bundles (temp/context-bundles/)\",\n    \"Guardian Digest (stdout/file)\",\n    \"Soul traces on Hugging Face\",\n    \"Chronicle, Task, ADR, Protocol markdown files\"\n  ],\n  \"dependencies\": [\n    \"mcp_servers/learning/operations.py (LearningOperations)\",\n    \"mcp_servers/rag_cortex/operations.py (CortexOperations)\",\n    \"mcp_servers/evolution/operations.py (EvolutionOperations)\",\n    \"mcp_servers/chronicle/operations.py (ChronicleOperations)\",\n    \"mcp_servers/task/operations.py (TaskOperations)\",\n    \"mcp_servers/adr/operations.py (ADROperations)\",\n    \"mcp_servers/protocol/operations.py (ProtocolOperations)\",\n    \"mcp_servers/forge_llm/operations.py (ForgeOperations) [optional]\",\n    \"tools/orchestrator/workflow_manager.py (WorkflowManager)\"\n  ],\n  \"key_functions\": [\n    \"main()\",\n    \"verify_iron_core()\"\n  ],\n  \"consumed_by\": [\n    \"User (Manual CLI)\",\n    \"Agent (via Tool Calls)\",\n    \"CI/CD Pipelines\",\n    \"/workflow-* workflows\"\n  ]\n}"
  },
  "scripts/capture_code_snapshot.py": {
    "hash": "manual_entry_2026_02_01",
    "summarized_at": "2026-02-01T14:48:00.000000",
    "summary": "{\n  \"purpose\": \"Generates a single text file snapshot of code files for LLM context sharing. Direct Python port of the legacy Node.js utility. Support traversing directories or using a manifest.\",\n  \"layer\": \"Curate / Documentation\",\n  \"supported_object_types\": [\"code files\", \"LLM Context\"],\n  \"usage\": [\n    \"python scripts/capture_code_snapshot.py\",\n    \"python scripts/capture_code_snapshot.py mcp_servers/rag_cortex --role guardian\",\n    \"python scripts/capture_code_snapshot.py --manifest .agent/learning/learning_manifest.json --output snapshot.txt\"\n  ],\n  \"args\": [\n    \"subfolder: Optional subfolder to process\",\n    \"--role: Target role (guardian, strategist, etc.)\",\n    \"--out: Output directory\",\n    \"--manifest: Path to JSON manifest\",\n    \"--output: Explicit output file path\",\n    \"--operation: Operation specific directory override\"\n  ],\n  \"inputs\": [\n    \"Project Source Code\",\n    \"Manifest JSON (optional)\"\n  ],\n  \"outputs\": [\n    \"dataset_package/markdown_snapshot_*.txt\",\n    \"dataset_package/*_awakening_seed.txt\"\n  ],\n  \"dependencies\": [\n    \"mcp_servers.lib.snapshot_utils\",\n    \"tiktoken\"\n  ],\n  \"key_functions\": [\n    \"generate_snapshot()\",\n    \"main()\"\n  ],\n  \"consumed_by\": [\n    \"tools/cli.py\",\n    \"Manual Context Gathering\"\n  ]\n}"
  },
  "scripts/link-checker/smart_fix_links.py": {
    "hash": "0fb17813788353bb",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Auto-repair utility for broken Markdown links. Uses a file inventory to find the correct location of moved or renamed files and updates the links in-place. Supports 'fuzzy' matching.\",\n  \"layer\": \"Curate / Link Checker\",\n  \"supported_object_types\": [\"Markdown\"],\n  \"usage\": [\n    \"python scripts/link-checker/smart_fix_links.py --dry-run\",\n    \"python scripts/link-checker/smart_fix_links.py\"\n  ],\n  \"args\": [\n    \"--dry-run: Report proposed changes without modifying files (Safety Mode)\"\n  ],\n  \"inputs\": [\n    \"scripts/link-checker/file_inventory.json (Source of Truth)\",\n    \"**/*.md (Target files to fix)\"\n  ],\n  \"outputs\": [\n    \"Modified .md files\",\n    \"Console report of fixes\"\n  ],\n  \"dependencies\": [\n    \"mcp_servers/lib/exclusion_config.py\"\n  ],\n  \"key_functions\": [\n    \"fix_links_in_file()\"\n  ],\n  \"consumed_by\": [\n    \"/post-move-link-check (Workflow)\",\n    \"Manual maintenance\"\n  ]\n}"
  },
  "scripts/link-checker/verify_links.py": {
    "hash": "new_hash_2",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Comprehensive integrity checker for the Project Sanctuary knowledge graph. Scans Markdown files and JSON manifests for broken internal links (dead references). Optionally validates external URLs. Enforces Protocol 128: Source Verification (Rule 9).\",\n  \"layer\": \"Curate / Link Checker\",\n  \"supported_object_types\": [\"Markdown\", \"Manifest\"],\n  \"usage\": [\n    \"python scripts/link-checker/verify_links.py\",\n    \"python scripts/link-checker/verify_links.py --check-external --output report.json\"\n  ],\n  \"args\": [\n    \"--root: Project root directory (default: .)\",\n    \"--check-external: Enable HTTP/HTTPS validation (slower)\",\n    \"--output: JSON report path\"\n  ],\n  \"inputs\": [\n    \"**/*.md\",\n    \"**/*manifest.json\"\n  ],\n  \"outputs\": [\n    \"JSON Report of broken links\",\n    \"Console summary\"\n  ],\n  \"dependencies\": [\n    \"requests (external lib)\"\n  ],\n  \"key_functions\": [\n    \"scan_md_file()\",\n    \"resolve_relative_path()\"\n  ],\n  \"consumed_by\": [\n    \"CI/CD Pipelines\",\n    \"Agent (Pre-Flight checks)\"\n  ]\n}"
  },
  "plugins/mermaid-to-png/skills/convert-mermaid/scripts/convert.py": {
    "file_mtime": 1769270613.391936,
    "hash": "124a3dee1a20a504",
    "summarized_at": "2026-01-29T10:16:25.019821",
    "summary": "{\n  \"purpose\": \"Renders all .mmd files in docs/architecture_diagrams/ to PNG images. Run this script whenever diagrams are updated to regenerate images.\",\n  \"layer\": \"Application Layer\",\n  \"supported_object_types\": [\"Mermaid Diagram (.mmd)\"],\n  \"usage\": [\n    \"python3 scripts/export_mmd_to_image.py                 # Render all\",\n    \"python3 scripts/export_mmd_to_image.py my_diagram.mmd  # Render specific file(s)\",\n    \"python3 scripts/export_mmd_to_image.py --svg           # Render as SVG instead\",\n    \"python3 scripts/export_mmd_to_image.py --check         # Check for outdated images\"\n  ],\n  \"args\": [\n    {\n      \"name\": \"--input\",\n      \"type\": \"str\",\n      \"help\": \"Input MMD file or directory\",\n      \"required\": false\n    },\n    {\n      \"name\": \"--output\",\n      \"type\": \"str\",\n      \"help\": \"Output file path or directory\",\n      \"required\": false\n    },\n    {\n      \"name\": \"--svg\",\n      \"action\": \"store_true\",\n      \"help\": \"Render as SVG instead of PNG\"\n    },\n    {\n      \"name\": \"--check\",\n      \"action\": \"store_true\",\n      \"help\": \"Check for outdated images only\"\n    }\n  ],\n  \"inputs\": [\"docs/architecture_diagrams/*.mmd\"],\n  \"outputs\": [\"PNG or SVG images in the specified output directory\"],\n  \"dependencies\": [\"mermaid-cli (npm install -g @mermaid-js/mermaid-cli)\"],\n  \"consumed_by\": [],\n  \"key_functions\": [\n    \"check_mmdc()\",\n    \"render_diagram(mmd_path: Path, output_format: str = \\\"png\\\") -> bool\",\n    \"check_outdated(mmd_path: Path, output_format: str = \\\"png\\\") -> bool\"\n  ]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/debug_rlm.py": {
    "file_mtime": 1769707602.6598785,
    "hash": "683125e86c5d11a2",
    "summarized_at": "2026-01-29T10:12:20.551771",
    "summary": "{\n  \"purpose\": \"Debug utility to inspect the RLMConfiguration state. Verifies path resolution, manifest loading, and environment variable overrides. Useful for troubleshooting cache path conflicts.\",\n  \"layer\": \"Standalone\",\n  \"supported_object_types\": [\"RLM Configuration\"],\n  \"usage\": [\"(None)\"],\n  \"args\": [],\n  \"inputs\": [\"tools/standalone/rlm-factory/manifest-index.json\", \".env\"],\n  \"outputs\": [\"Console output (State inspection)\"],\n  \"dependencies\": [\"plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py\"],\n  \"key_functions\": [\"main()\"]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py": {
    "file_mtime": 1769737016.6712525,
    "hash": "3a8e416775ada64c",
    "summarized_at": "2026-01-29T17:42:47.074224",
    "summary": "{\n  \"purpose\": \"Recursive summarization of repo content using Ollama.\",\n  \"layer\": \"Curate / Rlm\",\n  \"supported_object_types\": [\"Generic\"],\n  \"usage\": [\n    \"# 1. Distill a single file (Tool Logic)\",\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --file plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py --type tool\",\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --file tools/investigate/miners/db_miner.py --type tool --force\",\n    \"# 2. Distill legacy system documentation (Default)\",\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --file filepath.md\",\n    \"# 3. Incremental update (files changed in last 24 hours)\",\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --since 24 --type legacy\",\n    \"# 4. Process specific directory\",\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --target folder --type sanctuary\",\n    \"# 5. Force update (regenerate summaries even if unchanged)\",\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --target tools/investigate/miners --type tool --force\"\n  ],\n  \"args\": [\n    \"--file\",\n    \"Single file to process\",\n    \"--model\",\n    \"Ollama model to use\",\n    \"--cleanup\",\n    \"Remove stale entries for deleted/renamed files\",\n    \"--since\",\n    \"Process only files changed in last N hours\",\n    \"--no-cleanup\",\n    \"Skip auto-cleanup on incremental distills\",\n    \"--target\",\n    \"Target directories to process  (use with caution currently will process all files in the target directory)\",\n    \"--force\",\n    \"Force update (regenerate summaries even if unchanged)\"\n  ],\n  \"inputs\": [\"(See code)\"],\n  \"outputs\": [\"(See code)\"],\n  \"dependencies\": [\n    \"plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py (Configuration)\",\n    \"plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py (Cyclical: Updates inventory descriptions)\",\n    \"plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py (Orphan Removal)\"\n  ],\n  \"consumed_by\": [\n    \"plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py (Invokes distiller on tool updates)\"\n  ],\n  \"key_functions\": [\n    \"load_manifest()\",\n    \"load_cache()\",\n    \"save_cache()\",\n    \"compute_hash()\",\n    \"call_ollama()\",\n    \"distill()\",\n    \"run_cleanup()\"\n  ]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py": {
    "hash": "c31ca280d008bb57",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Central configuration factory for RLM. Resolves cache paths and loads manifests.\",\n  \"layer\": \"Codify / RLM\",\n  \"supported_object_types\": [\"RLM Config\"],\n  \"usage\": [\"from tools.codify.rlm.rlm_config import RLMConfig\"],\n  \"dependencies\": [],\n  \"key_functions\": [\"RLMConfig\"]\n}"
  },
  "plugins/tool-inventory/skills/tool-inventory/scripts/audit_plugins.py": {
    "file_mtime": 1769312126.2832038,
    "hash": "67bbe5d3d27fcaa5",
    "summarized_at": "2026-01-29T10:18:06.771048",
    "summary": "{\n  \"purpose\": \"Generates a summary report of AI Analysis progress from the tracking file. Shows analyzed vs pending forms for project management dashboards.\",\n  \"layer\": \"Application Layer\",\n  \"supported_object_types\": [\"AI Analysis Tracking\"],\n  \"usage\": [\"python plugins/tool-inventory/skills/tool-inventory/scripts/audit_plugins.py\"],\n  \"inputs\": [\"file path\"],\n  \"outputs\": [\"Summary report of AI Analysis progress\"],\n  \"dependencies\": [],\n  \"consumed_by\": [\"Project management dashboards\"],\n  \"key_functions\": [\n    \"analyze_status()\"\n  ]\n}"
  },
  "plugins/task-manager/skills/task-agent/scripts/create_task.py": {
    "file_mtime": 1769312126.2832038,
    "hash": "4f8c370b8a7478d1",
    "summarized_at": "2026-01-29T10:18:15.872122",
    "summary": "{\n  \"purpose\": \"Creates a prioritized TODO list of forms pending AI analysis. Bubbles up Critical and High priority items based on workflow usage.\",\n  \"layer\": \"Application Layer\",\n  \"supported_object_types\": [\"Forms\"],\n  \"usage\": \"python plugins/task-manager/skills/task-agent/scripts/create_task.py\",\n  \"args\": [],\n  \"inputs\": [\n    {\n      \"name\": \"ai_analysis_tracking.json\",\n      \"description\": \"JSON file containing form tracking data.\"\n    },\n    {\n      \"name\": \"Workflow_Summary.md\",\n      \"description\": \"Markdown file containing workflow forms data.\"\n    },\n    {\n      \"name\": \"folderpath\",\n      \"description\": \"Directory containing application overview markdown files.\"\n    }\n  ],\n  \"outputs\": [\n    {\n      \"name\": \"TODO_PENDING_ANALYSIS.md\",\n      \"description\": \"Markdown file listing pending forms for AI analysis, prioritized based on workflow and app core usage.\"\n    }\n  ],\n  \"dependencies\": [\n    \"json\",\n    \"re\",\n    \"pathlib\"\n  ],\n  \"consumed_by\": [\"AI Analysis System\"],\n  \"key_functions\": [\n    \"extract_form_ids\",\n    \"get_priority_forms\",\n    \"generate_todo\"\n  ]\n}"
  },
  "plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py": {
    "file_mtime": 1769561175.0164137,
    "hash": "78d0f2c74115efc9",
    "summarized_at": "2026-01-29T10:19:03.519385",
    "summary": "{\n  \"purpose\": \"Manages the workflow inventory for agent workflows (.agent/workflows/*.md). Provides search, scan, add, and update capabilities. Outputs are docs/antigravity/workflow/workflow_inventory.json and docs/antigravity/workflow/WORKFLOW_INVENTORY.md.\",\n  \"layer\": \"Application Layer\",\n  \"supported_object_types\": [\"Workflows\"],\n  \"usage\": [\n    \"# Scan and regenerate inventory\\npython plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py --scan\",\n    \"# Search workflows\\npython plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py --search \\\"keyword\\\"\",\n    \"# List all workflows\\npython plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py --list\",\n    \"# Show workflow details\\npython plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py --show \\\"workflow-name\\\"\"\n  ],\n  \"args\": [\n    {\n      \"long\": \"--scan\",\n      \"short\": \"\",\n      \"action\": \"store_true\",\n      \"help\": \"Scan workflows dir and regenerate inventory\"\n    },\n    {\n      \"long\": \"--search\",\n      \"short\": \"\",\n      \"type\": \"str\",\n      \"metavar\": \"QUERY\",\n      \"help\": \"Search workflows by keyword\"\n    },\n    {\n      \"long\": \"--list\",\n      \"short\": \"\",\n      \"action\": \"store_true\",\n      \"help\": \"List all workflows\"\n    },\n    {\n      \"long\": \"--show\",\n      \"short\": \"\",\n      \"type\": \"str\",\n      \"metavar\": \"NAME\",\n      \"help\": \"Show details for a workflow\"\n    }\n  ],\n  \"inputs\": [\".agent/workflows/*.md\"],\n  \"outputs\": [\n    \"docs/antigravity/workflow/workflow_inventory.json\",\n    \"docs/antigravity/workflow/WORKFLOW_INVENTORY.md\"\n  ],\n  \"dependencies\": [\"PathResolver\"],\n  \"consumed_by\": [],\n  \"key_functions\": [\n    \"parse_frontmatter\",\n    \"extract_called_by\",\n    \"scan_workflows\",\n    \"load_inventory\",\n    \"save_inventory\",\n    \"generate_json\",\n    \"generate_markdown\",\n    \"search_workflows\",\n    \"list_workflows\",\n    \"show_workflow\"\n  ]\n}"
  },
  "plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py": {
    "file_mtime": 1769716444.1697989,
    "hash": "4b4a01331d072e33",
    "summarized_at": "2026-01-29T17:20:29.446699",
    "summary": "{\n  \"purpose\": \"Comprehensive manager for Tool Inventories. Supports list, add, update, remove, search, audit, and generate operations.\",\n  \"layer\": \"Curate / Curate\",\n  \"supported_object_types\": [\"Generic\"],\n  \"usage\": [\n    \"python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py --help\",\n    \"python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py list\",\n    \"python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py search \\\"keyword\\\"\",\n    \"python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py remove --path \\\"path/to/tool.py\\\"\",\n    \"python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py update --path \\\"tool.py\\\" --desc \\\"New description\\\"\",\n    \"python plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py discover --auto-stub\"\n  ],\n  \"args\": [\n    \"--inventory\",\n    \"--path\",\n    \"--category\",\n    \"--desc\",\n    \"--output\",\n    \"keyword\",\n    \"--status\",\n    \"--new-path\",\n    \"--mark-compliant\",\n    \"--include-json\",\n    \"--json\",\n    \"--batch\",\n    \"--dry-run\"\n  ],\n  \"inputs\": [\"(See code)\"],\n  \"outputs\": [\"(See code)\"],\n  \"dependencies\": [\n    \"plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py (Cyclical: Triggers distillation on update)\",\n    \"plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py (Atomic cleanup on removal)\"\n  ],\n  \"consumed_by\": [\"plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py (Invokes update_tool for RLM-driven enrichment)\"],\n  \"key_functions\": [\n    \"generate_markdown()\",\n    \"extract_docstring()\",\n    \"main()\"\n  ]\n  }"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py": {
    "hash": "7727289db71a92cf",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Inventory Reconciliation Utility. Scans the `tool_inventory.json` against the filesystem and automatically removes entries for files that no longer exist (Pruning). Also lists all current scripts to audit 'ghosts' vs reality.\",\n  \"layer\": \"Curate / Inventories\",\n  \"supported_object_types\": [\"Temp Files\"],\n  \"usage\": [\"python plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py\"],\n  \"args\": [\n    \"None\"\n  ],\n  \"inputs\": [\n    \"plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json\",\n    \"Filesystem (plugins / directory)\"\n  ],\n  \"outputs\": [\n    \"Updates to tool_inventory.json\",\n    \"Console log of removed files\"\n  ],\n  \"dependencies\": [\n    \"plugins/tool-inventory/skills/tool-inventory/scripts/manage_tool_inventory.py\"\n  ],\n  \"key_functions\": [\n    \"get_missing_files()\",\n    \"remove_tool()\"\n  ],\n  \"consumed_by\": [\n    \"CI/CD Pipelines\",\n    \"Manual maintenance\"\n  ]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py": {
    "hash": "04571da937a7cad6",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"RLM Cleanup: Removes stale and orphan entries from the Recursive Language Model ledger.\",\n  \"layer\": \"Curate / Rlm\",\n  \"supported_object_types\": [\"RLM Cache Entry\"],\n  \"usage\": [\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/cleanup_cache.py --help\"\n  ],\n  \"args\": [\n    \"--apply: Perform the deletion\",\n    \"--prune-orphans: Remove entries not matching manifest\",\n    \"--v: Verbose mode\"\n  ],\n  \"inputs\": [\n    \"(See code)\"\n  ],\n  \"outputs\": [\n    \"(See code)\"\n  ],\n  \"dependencies\": [\n    \"plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py\"\n  ],\n  \"key_functions\": [\n    \"load_manifest_globs()\",\n    \"matches_any()\",\n    \"main()\"\n  ],\n  \"consumed_by\": [\n    \"(Unknown)\"\n  ]\n}"
  },
  "plugins/adr-manager/skills/adr-management/scripts/next_number.py": {
    "hash": "new_hash_1",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Sequential Identifier Generator. Scans artifact directories (Specs, Tasks, ADRs, Chronicles) to find the next available sequence number. Prevents ID collisions.\",\n  \"layer\": \"Investigate / Utils\",\n  \"supported_object_types\": [\"ADR\", \"Task\", \"Spec\", \"Chronicle\"],\n  \"usage\": [\n    \"python plugins/adr-manager/skills/adr-management/scripts/next_number.py --type spec\",\n    \"python plugins/adr-manager/skills/adr-management/scripts/next_number.py --type task\",\n    \"python plugins/adr-manager/skills/adr-management/scripts/next_number.py --type all\"\n  ],\n  \"args\": [\n    \"--type: Artifact type (spec, task, adr, chronicle, all)\"\n  ],\n  \"inputs\": [\n    \"specs/\",\n    \"tasks/\",\n    \"ADRs/\",\n    \"00_CHRONICLE/ENTRIES/\"\n  ],\n  \"outputs\": [\n    \"Next available ID (e.g. \\\"0045\\\") to stdout\"\n  ],\n  \"dependencies\": [\n    \"pathlib\",\n    \"re\"\n  ],\n  \"key_functions\": [\n    \"main()\"\n  ],\n  \"consumed_by\": [\n    \"scripts/domain_cli.py\",\n    \"Manual workflow execution\"\n  ]\n}"
  },
  "plugins/agent-loops/skills/orchestrator/scripts/proof_check.py": {
    "hash": "6ffa54cfc3afc26f",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Scans spec.md, plan.md, and tasks.md for file references and verifies each referenced file has been modified compared to origin/main. This tool prevents 'checkbox fraud'.\",\n  \"layer\": \"Orchestrator / Verification\",\n  \"supported_object_types\": [\"Spec artifacts\", \"File references\"],\n  \"usage\": [\n    \"python plugins/agent-loops/skills/orchestrator/scripts/proof_check.py --spec-dir specs/0005-human-gate-protocols\",\n    \"python plugins/agent-loops/skills/orchestrator/scripts/proof_check.py --spec-dir specs/0005-foo --json\"\n  ],\n  \"args\": [\n    \"--spec-dir: Path to spec directory (required)\",\n    \"--project-root: Project root directory (default: current)\",\n    \"--json: Output in JSON format (optional)\"\n  ],\n  \"inputs\": [\n    \"specs/[ID]/spec.md\",\n    \"specs/[ID]/plan.md\",\n    \"specs/[ID]/tasks.md\"\n  ],\n  \"outputs\": [\n    \"Summary report of modified/unchanged files\",\n    \"Exit code 1 if fail, 0 if pass\"\n  ],\n  \"dependencies\": [\n    \"Git\"\n  ],\n  \"key_functions\": [\n    \"extract_file_refs()\",\n    \"check_file_modified()\",\n    \"run_proof_check()\"\n  ],\n  \"consumed_by\": [\n    \"tools/cli.py workflow retrospective\",\n    \"/sanctuary-retrospective workflow\"\n  ]\n}"
  },
  "tools/orchestrator/workflow_manager.py": {
    "hash": "ecb4b989fc6c51d2",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Core logic for the 'Python Orchestrator' architecture (ADR-0030 v2/v3). Handles Git State checks, Context Alignment, Branch Creation & Naming, and Context Manifest Initialization. Acts as the single source of truth for 'Start Workflow' logic.\",\n  \"layer\": \"Orchestrator\",\n  \"supported_object_types\": [\"Workflow State\"],\n  \"usage\": [\n    \"from tools.orchestrator.workflow_manager import WorkflowManager\",\n    \"mgr = WorkflowManager()\",\n    \"mgr.start_workflow('codify', 'MyTarget')\"\n  ],\n  \"args\": [\n    \"Workflow Name\",\n    \"Target ID\",\n    \"Artifact Type\"\n  ],\n  \"inputs\": [\n    \"Workflow Name\",\n    \"Target ID\"\n  ],\n  \"outputs\": [\n    \"Exit Code 0: Success (Proceed)\",\n    \"Exit Code 1: Failure (Stop)\"\n  ],\n  \"dependencies\": [\n    \"plugins/adr-manager/skills/adr-management/scripts/path_resolver.py\"\n  ],\n  \"key_functions\": [\n    \"start_workflow\",\n    \"get_git_status\",\n    \"get_current_branch\",\n    \"generate_next_id\"\n  ],\n  \"consumed_by\": [\n    \"tools/cli.py\",\n    \"Manual Scripts\"\n  ]\n}"
  },
  "plugins/context-bundler/scripts/bundle.py": {
    "file_mtime": 1769525919.0013855,
    "hash": "4c0db7435b49626d",
    "summarized_at": "2026-02-01T16:15:00.000000",
    "summary": "{\n  \"purpose\": \"Bundles multiple source files into a single Markdown 'Context Bundle' based on a JSON manifest. Warns on deprecated legacy keys (core, topic, etc.).\",\n  \"layer\": \"Curate / Bundler\",\n  \"supported_object_types\": [\"Context Bundle\", \"Markdown\"],\n  \"usage\": [\n    \"python plugins/context-bundler/scripts/bundle.py manifest.json\",\n    \"python plugins/context-bundler/scripts/bundle.py manifest.json -o output.md\"\n  ],\n  \"args\": [\n    \"manifest: Path to file-manifest.json\",\n    \"-o, --output: Output markdown file path (default: bundle.md)\"\n  ],\n  \"inputs\": [\n    \"file-manifest.json\",\n    \"Source files referenced in manifest\"\n  ],\n  \"outputs\": [\n    \"Markdowm bundle file (e.g. bundle.md)\"\n  ],\n  \"dependencies\": [\n    \"plugins/adr-manager/skills/adr-management/scripts/path_resolver.py\"\n  ],\n  \"key_functions\": [\n    \"write_file_content()\",\n    \"bundle_files()\"\n  ]\n}"
  },
  "plugins/context-bundler/scripts/bundle.py": {
    "file_mtime": 1769554275.3037753,
    "hash": "25a46ba715d8df06",
    "summarized_at": "2026-02-01T16:15:00.000000",
    "summary": "{\n  \"purpose\": \"Handles initialization and modification of the context-manager manifest. Acts as the primary CLI for the Context Bundler. Supports manifest initialization, file addition/removal/update, searching, and bundling execution.\",\n  \"layer\": \"Curate / Bundler\",\n  \"supported_object_types\": [\"Manifest\", \"Context Bundle\"],\n  \"usage\": [\n    \"# Initialize a new manifest\",\n    \"python plugins/context-bundler/scripts/bundle.py init --bundle-title 'My Feature' --type generic\",\n    \"# Add a file\",\n    \"python plugins/context-bundler/scripts/bundle.py add --path 'docs/readme.md' --note 'Overview'\",\n    \"# Remove a file\",\n    \"python plugins/context-bundler/scripts/bundle.py remove --path 'docs/readme.md'\",\n    \"# Bundle files\",\n    \"python plugins/context-bundler/scripts/bundle.py bundle --output 'context.md'\"\n  ],\n  \"args\": [\n    \"init: Initialize manifest (--bundle-title, --type, --manifest)\",\n    \"add: Add file (--path, --note, --base, --manifest, --section)\",\n    \"remove: Remove file (--path, --base, --manifest)\",\n    \"update: Update file (--path, --note, --new-path, --base, --manifest)\",\n    \"search: Search files (pattern, --base, --manifest)\",\n    \"list: List files (--base, --manifest)\",\n    \"bundle: Execute bundle (--output, --base, --manifest)\"\n  ],\n  \"inputs\": [\n    \"tools/standalone/context-bundler/file-manifest.json\",\n    \"tools/standalone/context-bundler/base-manifests/*.json\"\n  ],\n  \"outputs\": [\n    \"tools/standalone/context-bundler/file-manifest.json\",\n    \"Context bundles (.md)\"\n  ],\n  \"dependencies\": [\n    \"plugins/adr-manager/skills/adr-management/scripts/path_resolver.py\",\n    \"plugins/context-bundler/scripts/bundle.py\"\n  ],\n  \"key_functions\": [\n    \"add_file()\",\n    \"bundle()\",\n    \"get_base_manifest_path()\",\n    \"init_manifest()\",\n    \"list_manifest()\",\n    \"load_manifest()\",\n    \"remove_file()\",\n    \"save_manifest()\",\n    \"search_files()\",\n    \"update_file()\"\n  ]\n}"
  },
  "plugins/context-bundler/scripts/bundle.py": {
    "hash": "new_validate_2026",
    "summarized_at": "2026-02-01T16:15:00.000000",
    "summary": "{\n  \"purpose\": \"Validates context bundler manifest files against the schema. Checks for required fields (title, files), path format, and path traversal vulnerabilities.\",\n  \"layer\": \"Retrieve / Bundler\",\n  \"supported_object_types\": [\"Manifest\", \"Index\"],\n  \"usage\": [\n    \"python plugins/context-bundler/scripts/bundle.py manifest.json\",\n    \"python plugins/context-bundler/scripts/bundle.py --all-base\",\n    \"python plugins/context-bundler/scripts/bundle.py --check-index\"\n  ],\n  \"args\": [\n    \"manifest: Path to manifest JSON file\",\n    \"--all-base: Validate all base manifests\",\n    \"--check-index: Validate manifest index\",\n    \"--quiet: Suppress output\"\n  ],\n  \"inputs\": [\n    \"Manifest JSON files\",\n    \"file-manifest-schema.json\"\n  ],\n  \"outputs\": [\n    \"Validation report (stdout)\",\n    \"Exit code 0 (valid) or 1 (invalid)\"\n  ],\n  \"dependencies\": [\n    \"tools/standalone/context-bundler/file-manifest-schema.json\"\n  ],\n  \"key_functions\": [\n    \"validate_manifest()\",\n    \"validate_index()\",\n    \"validate_all_base()\"\n  ]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py": {
    "file_mtime": 1769924260.8329458,
    "hash": "e6db97fa1e136a6a",
    "summarized_at": "2026-01-31T22:01:21.015896",
    "summary": "{\n  \"purpose\": \"Retrieves the 'Gold Standard' tool definition from the RLM Tool Cache and formats it into an Agent-readable 'Manual Page'. This is the second step of the Late-Binding Protocol, following query_cache.py which finds a tool, this script provides the detailed context needed to use it.\",\n  \"layer\": \"Retrieve\",\n  \"supported_object_types\": [\"Generic\"],\n  \"usage\": [\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --file tools/cli.py\",\n    \"python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --file scripts/domain_cli.py\"\n  ],\n  \"args\": [\n    \"--file : Path to the tool script (required, e.g., tools/cli.py)\"\n  ],\n  \"inputs\": [],\n  \"outputs\": [\"Markdown-formatted technical specification to stdout:\"],\n  \"dependencies\": [\n    \"- plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py: RLM configuration and cache loading\"\n  ],\n  \"consumed_by\": [\n    \"- Agent during Late-Binding tool discovery flow\"\n  ],\n  \"key_functions\": [\n    \"format_as_manual(file_path: str, data: dict)\"\n  ]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/inventory.py": {
    "file_mtime": 1769703353.8198912,
    "hash": "e89667d95d7a8297",
    "summarized_at": "2026-01-29T09:54:31.690762",
    "summary": "{\n  \"purpose\": \"RLM Auditor: Reports coverage of the semantic ledger against the filesystem. Uses the Shared RLMConfig to dynamically switch between 'Legacy' (Documentation) and 'Tool' (CLI) audit modes.\",\n  \"layer\": \"Curate / Rlm\",\n  \"supported_object_types\": [\"RLM Cache (Legacy)\", \"RLM Cache (Tool)\"],\n  \"usage\": \"No explicit usage examples provided in the code. The script is intended to be run from the command line with optional --type argument.\",\n  \"args\": [\n    {\n      \"name\": \"--type\",\n      \"description\": \"Selects the configuration profile (default: legacy).\",\n      \"choices\": [\"legacy\", \"tool\"]\n    }\n  ],\n  \"inputs\": [\n    \".agent/learning/rlm_summary_cache.json (Legacy)\",\n    \".agent/learning/rlm_tool_cache.json (Tool)\",\n    \"Filesystem targets (defined in manifests)\",\n    \"tool_inventory.json\"\n  ],\n  \"outputs\": [\"Console report (Statistics, Missing Files, Stale Entries)\"],\n  \"dependencies\": [\n    \"plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py\"\n  ],\n  \"key_functions\": [\n    \"audit_inventory()\"\n  ]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py": {
    "file_mtime": 1769702294.5297713,
    "hash": "fd76ff31721cc98a",
    "summarized_at": "2026-01-29T10:12:28.812250",
    "summary": "{\n  \"purpose\": \"RLM Search: Instant O(1) semantic search of the ledger.\",\n  \"layer\": \"Curate / Rlm\",\n  \"supported_object_types\": [\"Generic\"],\n  \"usage\": [\"python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --help\"],\n  \"args\": [\n    {\n      \"name\": \"term\",\n      \"description\": \"Search term (ID, filename, or content keyword)\"\n    },\n    {\n      \"name\": \"--list\",\n      \"description\": \"List all cached files\"\n    },\n    {\n      \"name\": \"--no-summary\",\n      \"description\": \"Hide summary text\"\n    },\n    {\n      \"name\": \"--json\",\n      \"description\": \"Output results as JSON\"\n    }\n  ],\n  \"inputs\": [\"- (See code)\"],\n  \"outputs\": [\"- (See code)\"],\n  \"dependencies\": [\"(None detected)\"],\n  \"consumed_by\": [\"(Unknown)\"],\n  \"key_functions\": [\n    \"load_cache()\",\n    \"search_cache()\",\n    \"list_cache()\",\n    \"main()\"\n  ]\n}"
  },
  "plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py": {
    "hash": "new_hash_3",
    "summarized_at": "2026-01-31T22:55:00.000000",
    "summary": "{\n  \"purpose\": \"Standardizes cross-platform path resolution and provides access to the Master Object Collection (MOC). Acts as a central utility for file finding.\",\n  \"layer\": \"Tools / Utils\",\n  \"supported_object_types\": [\"Generic\", \"Path\"],\n  \"usage\": [\n    \"from tools.utils.path_resolver import resolve_path\"\n  ],\n  \"args\": [\n    \"(None detected)\"\n  ],\n  \"inputs\": [\n    \"(See code)\"\n  ],\n  \"outputs\": [\n    \"(See code)\"\n  ],\n  \"dependencies\": [\n    \"(None detected)\"\n  ],\n  \"key_functions\": [\n    \"resolve_root()\",\n    \"resolve_path()\"\n  ],\n  \"consumed_by\": [\n    \"(Unknown)\"\n  ]\n}"
  },
  "plugins/adr-manager/skills/adr-management/scripts/path_resolver.py": {
    "summary": {
      "purpose": "Standardizes cross-platform path resolution and provides access to the Master Object Collection (MOC). Acts as a central utility for file finding.",
      "layer": "Curate / Bundler",
      "supported_object_types": [
        "Generic"
      ]
    },
    "summarized_at": "2026-01-31T23:55:00.000000"
  },
  "plugins/adr-manager/skills/adr-management/scripts/pathResolver.js": {
    "summary": {
      "purpose": "Node.js implementation of path resolution logic.",
      "layer": "Utility",
      "supported_object_types": [
        "Generic"
      ]
    },
    "summarized_at": "2026-01-31T23:55:00.000000"
  },
  "plugins/adr-manager/skills/adr-management/scripts/rlmConfigResolver.js": {
    "summary": {
      "purpose": "Resolves RLM configuration paths for Node.js tools.",
      "layer": "Configuration",
      "supported_object_types": [
        "Generic"
      ]
    },
    "summarized_at": "2026-01-31T23:55:00.000000"
  },
  "forge/scripts/upload_to_huggingface.py": {
    "hash": "manual_entry_2026_02_01",
    "summarized_at": "2026-02-01T12:00:00.000000",
    "summary": "{\n  \"purpose\": \"Manages the upload of model weights, GGUF files, and metadata to Hugging Face Hub (Phase 6). Handles artifact selection, repo creation, and secure transport.\",\n  \"layer\": \"Curate\",\n  \"supported_object_types\": [\"Model Weights\", \"Metadata\", \"Configuration\"],\n  \"usage\": [\n    \"python forge/scripts/upload_to_huggingface.py --repo user/repo --gguf --readme\",\n    \"python forge/scripts/upload_to_huggingface.py --files ./custom_model.bin --private\"\n  ],\n  \"args\": [\n    \"--repo: HF Repo ID\",\n    \"--files: Explicit file paths\",\n    \"--private: Mark repo as private\",\n    \"--gguf: Upload GGUF artifacts\",\n    \"--modelfile: Upload Ollama Modelfile\"\n  ],\n  \"inputs\": [\"forge/config/upload_config.yaml\", \"models/gguf/*\", \"huggingface/README.md\"],\n  \"outputs\": [\"Uploaded artifacts on Hugging Face Hub\"],\n  \"dependencies\": [\"mcp_servers.lib.hf_utils\", \"mcp_servers.lib.env_helper\"],\n  \"key_functions\": [\"load_config()\", \"perform_upload()\"]\n}"
  },
  "mcp_servers/lib/hf_utils.py": {
    "hash": "manual_entry_2026_02_01",
    "summarized_at": "2026-02-01T12:00:00.000000",
    "summary": "{\n  \"purpose\": \"Hugging Face utility library for soul persistence (ADR 079). Encapsulates huggingface_hub logic. Provides unified async primitives for uploading files, folders, and updating datasets.\",\n  \"layer\": \"Retrieve / Curate (Library)\",\n  \"supported_object_types\": [\"Soul Snapshot\", \"Semantic Cache\", \"Learning History\"],\n  \"usage\": [\"from mcp_servers.lib.hf_utils import upload_to_hf_hub\"],\n  \"args\": [],\n  \"inputs\": [],\n  \"outputs\": [],\n  \"dependencies\": [\"huggingface_hub\"],\n  \"key_functions\": [\"upload_soul_snapshot()\", \"upload_semantic_cache()\", \"sync_full_learning_history()\", \"ensure_dataset_structure()\"]\n}"
  },
  "scripts/hugging-face/hf_decorate_readme.py": {
    "hash": "manual_entry_2026_02_01",
    "summarized_at": "2026-02-01T12:00:00.000000",
    "summary": "{\n  \"purpose\": \"Prepares the local Hugging Face staging directory for upload. Modifies 'hugging_face_dataset_repo/README.md' in-place with YAML frontmatter per ADR 081.\",\n  \"layer\": \"Curate\",\n  \"supported_object_types\": [\"README.md\"],\n  \"usage\": [\"python scripts/hugging-face/hf_decorate_readme.py\"],\n  \"args\": [],\n  \"inputs\": [\"hugging_face_dataset_repo/README.md\"],\n  \"outputs\": [\"Modified README.md\", \"Created directories: lineage/, data/\"],\n  \"dependencies\": [],\n  \"key_functions\": [\"stage_readme()\"]\n}"
  },
  "mcp_servers/lib/env_helper.py": {
    "hash": "manual_entry_2026_02_01",
    "summarized_at": "2026-02-01T12:00:00.000000",
    "summary": "{\n  \"purpose\": \"Simple environment variable helper with proper fallback (Env -> .env). Ensures consistent secret loading across Project Sanctuary.\",\n  \"layer\": \"Core / Utility\",\n  \"supported_object_types\": [\"Environment Variables\"],\n  \"usage\": [\"from mcp_servers.lib.env_helper import get_env_variable\"],\n  \"args\": [],\n  \"inputs\": [\".env\"],\n  \"outputs\": [\"Environment variable value\"],\n  \"dependencies\": [\"python-dotenv\"],\n  \"key_functions\": [\"get_env_variable()\", \"load_env()\"]\n}"
  },
  "scripts/hugging-face/hf_upload_assets.py": {
    "hash": "manual_entry_2026_02_01",
    "summarized_at": "2026-02-01T12:00:00.000000",
    "summary": "{\n  \"purpose\": \"Synchronizes staged landing-page assets with the Hugging Face Hub (ADR 081). Uploads the final, metadata-rich README.md to the repository root.\",\n  \"layer\": \"Curate / Deployment\",\n  \"supported_object_types\": [\"README.md\"],\n  \"usage\": [\"python scripts/hugging-face/hf_upload_assets.py\"],\n  \"args\": [],\n  \"inputs\": [\"hugging_face_dataset_repo/README.md\"],\n  \"outputs\": [\"Uploads README.md to HF\"],\n  \"dependencies\": [\"mcp_servers.lib.hf_utils\"],\n  \"key_functions\": [\"upload_assets()\"]\n}"
  }
}
```
