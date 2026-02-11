# Workflow Creation & Modification Policy

## 1. Overview
This policy governs the lifecycle of Agent Workflows in Project Sanctuary. All workflows must be discoverable, documented, and architecturally aligned (ADR-036).

## 2. File Standards
- **Location**: `.agent/workflows/[name].md`
- **Naming**: `kebab-case` (e.g. `bundle-manage.md`)
- **Frontmatter**: Must include `description`, `inputs` (optional), `tier` (optional), and `track` (optional).
  ```yaml
  ---
  description: Brief summary of the workflow.
  tier: 1
  track: Factory
  ---
  ```

## 3. Architecture Alignment (ADR-036)
- **Thin Shim**: If a CLI convenience wrapper is needed, create `scripts/bash/[name].sh`.
- **Thick Logic**: The shim MUST NOT contain logic. It must `exec` a Python script (`tools/cli.py` or specific tool).
- **No Shim Proliferation**: Prefer using `workflow-start` for complex, branch-managed workflows. Use specific shims only for atomic tools (e.g. `workflow-bundle`).

## 4. Registration Process
After creating or modifying a workflow file (`.md`):
1. **Inventory Scan**: Run `python tools/curate/documentation/workflow_inventory_manager.py --scan` to update `WORKFLOW_INVENTORY.md`.
2. **Tool Integration**: If the workflow uses a new tool, ensure the tool is registered in `tools/tool_inventory.json`.

## 5. Documentation
- **Self-Documenting**: The `.md` file is the source of truth.
- **Inventory**: Do not manually edit `WORKFLOW_INVENTORY.md`. Use the manager.
- **Readme**: Update relevant tool READMEs if the workflow exposes new capabilities.

## 6. Verification
- Validate the workflow using a test run (if possible) or by verifying the inventory entry.
