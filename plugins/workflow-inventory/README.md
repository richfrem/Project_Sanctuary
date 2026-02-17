# Workflow Inventory Plugin 📜

Manages the registry of Agent Workflows (`.agent/workflows/*.md`) and generates documentation.

## Features
- **Auto-Discovery**: Scans `.agent/workflows/` to find valid slash commands.
- **Documentation**: Generates `workflow_inventory.json` and `WORKFLOW_INVENTORY.md` in `docs/antigravity/workflow/`.
- **Search**: CLI tool to find workflows by keyword.

## Installation

### Local Development
```bash
claude --plugin-dir ./plugins/workflow-inventory
```

## Usage

### CLI
```bash
# Update inventory
python3 plugins/workflow-inventory/scripts/workflow_inventory_manager.py --scan

# Search
python3 plugins/workflow-inventory/scripts/workflow_inventory_manager.py --search "test"
```

### Agent Integration
Invoke **workflow-inventory-agent** to find or list workflows.

## License
MIT
