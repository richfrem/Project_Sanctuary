---
trigger: always_on
---

# 🛡️ Global Operational Protocol

Agent MUST consult cache of tools first before running native file system commands. Agent MUST use rich suite of tools that can be queried via tool discovery mentioned below.

## 1. Tool Discovery & Retrieval (RLM Protocol)
**CRITICAL:** You operate in a "Late-Binding" environment. You do not have all tools loaded.
* **Trigger:** When you need to find, locate, or search for a tool/script/capability.
* **Constraint:** You **MUST NOT** use native filesystem commands (`grep`, `find`, `ls -R`, or "search codebase"). These are forbidden for tool discovery.
* **Constraint:** You **MUST NOT** use `manage_tool_inventory.py` for discovery.
* **Action:** You **MUST** exclusively use the **Tool Discovery** skill (`tools/retrieve/rlm/query_cache.py`).

## 2. Binding Protocol
* Once a tool is found via `query_cache.py`, you must **"read the manual"** by viewing the script's header directly.
 * **Command**: `view_file(AbsolutePath="/path/to/script.py", StartLine=1, EndLine=200)`
 * The header (docstring) contains the authoritative usage, arguments, and examples.

## 3. Tool Registration Protocol (MANDATORY)
**When creating or modifying CLI tools/scripts in `tools/`:**

1. **Follow Coding Conventions**: Use proper file header per `.agent/rules/coding_conventions_policy.md`
2. **Register in Inventory**: After creating/modifying a tool, run:
   ```bash
   python tools/curate/inventories/manage_tool_inventory.py add --path "tools/path/to/script.py"
   ```
3. **RLM Distillation**: The inventory manager auto-triggers RLM distillation, but you can also run manually:
   ```bash
   python tools/codify/rlm/distiller.py --file "tools/path/to/script.py" --type tool
   ```

**Verification**: Before closing a spec that added tools, run:
```bash
python tools/curate/inventories/manage_tool_inventory.py audit
```

**Why This Matters**: Unregistered tools are invisible to future LLM sessions. If you create a tool but don't register it, it cannot be discovered.