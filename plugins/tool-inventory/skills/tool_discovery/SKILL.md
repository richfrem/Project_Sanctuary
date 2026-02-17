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
python plugins/tool-inventory/scripts/query_cache.py --type tool "KEYWORD"
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
