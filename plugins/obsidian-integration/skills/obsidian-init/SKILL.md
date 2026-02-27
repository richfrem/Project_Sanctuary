---
name: obsidian-init
description: "Initialize and onboard a new project repository as an Obsidian Vault. Configures exclusion filters, sets SANCTUARY_VAULT_PATH, and validates the vault structure."
---

# Obsidian Init (Vault Onboarding)

**Status:** Active
**Author:** Sanctuary Guardian
**Domain:** Obsidian Integration

## Purpose

This skill initializes any project directory as a functioning Obsidian Vault. It is the **first skill to run** when onboarding a new project into the Obsidian ecosystem.

## What It Does

1. **Validates** the target directory exists and contains `.md` files
2. **Creates** the `.obsidian/` configuration directory (if not present)
3. **Writes** the `app.json` with sensible exclusion filters for developer repos
4. **Sets** the `SANCTUARY_VAULT_PATH` environment variable hint
5. **Verifies** that the vault can be opened by Obsidian

## Usage

### Interactive Init
```bash
python plugins/obsidian-integration/skills/obsidian-init/scripts/init_vault.py --vault-root <path>
```

### With Custom Exclusions
```bash
python plugins/obsidian-integration/skills/obsidian-init/scripts/init_vault.py \
  --vault-root <path> \
  --exclude "node_modules/" ".worktrees/" "venv/"
```

### Validate Only (No Changes)
```bash
python plugins/obsidian-integration/skills/obsidian-init/scripts/init_vault.py --vault-root <path> --validate-only
```

## Default Exclusion Filters

The init script applies these exclusions by default (see `VAULT_CONFIG.md` for rationale):

| Pattern | Reason |
|:--------|:-------|
| `node_modules/` | NPM dependencies |
| `.worktrees/` | Git worktree isolation |
| `.vector_data/` | ChromaDB binary data |
| `.git/` | Git internals |
| `venv/` | Python virtual environments |
| `__pycache__/` | Python bytecode cache |
| `*.json` | Data/config files (not knowledge) |
| `*.jsonl` | Export payloads |
| `learning_package_snapshot.md` | Machine-generated bundle |
| `bootstrap_packet.md` | Machine-generated bundle |
| `learning_debrief.md` | Machine-generated bundle |
| `*_packet.md` | Audit/review bundles |
| `*_digest.md` | Context digests |
| `dataset_package/` | Export artifacts |

## Post-Init Steps

After running the init:
1. Open the Obsidian desktop app
2. Click "Open Folder as Vault"
3. Select the vault root directory
4. Obsidian will immediately index all non-excluded `.md` files
5. The graph view will show all `[[wikilink]]` connections

## Portability Note

This skill is **project-agnostic**. It works on any Git repository with markdown files. The exclusion filters are sensible defaults for developer projects.
