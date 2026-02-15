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

### 1. Search for Tools (Smart Querying)
**Goal**: Find a tool relevant to your current objective.

**Strategy**:
1.  **Run Exact Search**: `python tools/retrieve/rlm/query_cache.py --type tool "dependency graph"`
2.  ** IF EMPTY, Broaden Search**: `python tools/retrieve/rlm/query_cache.py --type tool "dependency"`
3.  ** IF STILL EMPTY, Refresh Cache**:
    *   Run `python tools/codify/rlm/refresh_cache.py`
    *   Retry the broad search.

### 2. Retrieve & Bind (Auto-Binding)
**Goal**: Load the "Gold Standard" usage contract.

**Logic**:
- **If you found a high-confidence match (e.g., `tools/viz/graph_deps.py`)**, **IMMEDIATELY** read its header.
- Do not wait for a user prompt.

**Command**:
```bash
view_file(AbsolutePath="/path/to/found/script.py", StartLine=1, EndLine=200)
```

### 3. Execution (Trust & Run)
**Goal**: Run the tool using the knowledge gained in Step 2.

**Logic**:
* **Scenario A (Clear Manual)**: If the header has usage examples, **execute immediately**.
* **Scenario B (Ambiguous)**: If unclear, run `python [PATH] --help`.
```
