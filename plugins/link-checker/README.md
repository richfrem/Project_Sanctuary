# Link Checker Plugin 🔗

Validate and auto-repair broken documentation links across your repository using
file inventory mapping and fuzzy matching.

## Installation

### Local Development
```bash
claude --plugin-dir ./plugins/link-checker
```

### From Marketplace (when published)
```
/plugin install link-checker
```

### Prerequisites
- **Claude Code** ≥ 1.0.33
- **Python** ≥ 3.8 (stdlib only — no pip dependencies)

### Verify Installation
After loading, `/help` should show:
```
/link-checker:map     Index repository files
/link-checker:check   Scan for broken links
/link-checker:fix     Auto-repair broken links
```

---

## Usage Guide

### Quick Start (3-Step Protocol)

**Order matters: Map → Fix → Verify**

```bash
# 1. Index the repository (creates file inventory)
/link-checker:map

# 2. Auto-repair broken links using the inventory
/link-checker:fix

# 3. Final audit — verify remaining issues
/link-checker:check
```

### Direct CLI Usage (without Claude)
```bash
cd /path/to/your/repo

# Step 1: Map
python3 plugins/link-checker/scripts/map_repository_files.py

# Step 2: Fix
python3 plugins/link-checker/scripts/smart_fix_links.py

# Step 3: Check
python3 plugins/link-checker/scripts/check_broken_paths.py
python3 plugins/link-checker/scripts/check_broken_paths.py --file docs/specific.md
```

### Commands Reference

| Command | Script | Description |
|:---|:---|:---|
| `/link-checker:map` | `map_repository_files.py` | Index all files → `file_inventory.json` |
| `/link-checker:fix` | `smart_fix_links.py` | Auto-repair broken links via fuzzy match |
| `/link-checker:check` | `check_broken_paths.py` | Audit & report → `broken_links.log` |

### How the Fixer Works

1. Scans `.md` files for `[text](broken/path)` patterns
2. Extracts the basename from broken paths
3. Looks up the basename in `file_inventory.json`
4. **Unique match** → rewrites with correct relative path
5. **Ambiguous** (multiple files with same name) → skips with warning
6. **Not found** → marks as `(Reference Missing: filename)`

### Safety Features
- Only modifies files with actual broken links
- Skips `README.md` basename matches (too ambiguous across repos)
- Preserves anchor fragments (`#section`)
- Excludes `.git`, `node_modules`, `.venv`, `bin`, `obj` from scanning

---

## Architecture

See [docs/link-checker-workflow.mmd](docs/link-checker-workflow.mmd) for the full
sequence diagram.

```mermaid
graph LR
    A["Map 🗺️"] -->|file_inventory.json| B["Fix 🔧"]
    B -->|Modified .md files| C["Check ✅"]
    C -->|broken_links.log| D[Review]
```

Additional diagrams (from original tool):
- [logic.mmd](docs/logic.mmd) — Internal decision logic
- [workflow.mmd](docs/workflow.mmd) — User workflow
- [unpacking.mmd](docs/unpacking.mmd) — Legacy unpacking flow

### Plugin Directory Structure
```
link-checker/
├── .claude-plugin/
│   └── plugin.json              # Plugin identity
├── commands/
│   ├── map.md                   # /link-checker:map
│   ├── fix.md                   # /link-checker:fix
│   └── check.md                 # /link-checker:check
├── skills/
│   └── link-checker-agent/
│       └── SKILL.md             # Auto-invoked QA skill
├── scripts/
│   ├── map_repository_files.py  # The Mapper
│   ├── smart_fix_links.py       # The Fixer
│   └── check_broken_paths.py    # The Inspector
├── docs/
│   ├── link-checker-workflow.mmd  # Sequence diagram
│   ├── logic.mmd                  # Internal logic
│   ├── workflow.mmd               # User workflow
│   └── unpacking.mmd             # Legacy flow
└── README.md
```

---

## License

MIT
