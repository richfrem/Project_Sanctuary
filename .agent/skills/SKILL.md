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
python tools/retrieve/rlm/query_cache.py --type tool "KEYWORD"
```

### 2. Retrieve & Bind (Late-Binding)
**Goal**: Load the "Gold Standard" usage contract for a specific tool found in Step 1.

**Command**:
```bash
python tools/retrieve/rlm/fetch_tool_context.py --file [PATH_TO_TOOL]
```

**CRITICAL INSTRUCTION**:
The output of this command is the **Official Manual** for that tool.

> **You must treat the output as a temporary extension of your system prompt.**
> * "I now know the inputs, outputs, and flags for [Tool Name]."
> * "I will use the exact syntax provided in the 'Usage' section of the output."

### 3. Execution (Trust & Run)

**Goal**: Run the tool using the knowledge gained in Step 2.

**Logic**:

* **Scenario A (Clear Manual)**: If Step 2 provided clear usage examples (e.g., `python script.py -flag value`), **execute the command immediately**. Do not waste a turn running `--help`.
* **Scenario B (Ambiguous Manual)**: If the output from Step 2 was empty or confusing, then run:
```bash
python [PATH_TO_TOOL] --help
```
