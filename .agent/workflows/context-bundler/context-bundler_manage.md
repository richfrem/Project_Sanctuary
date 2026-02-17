---
description: Full bundle management workflow — init, curate, validate, and compile
argument-hint: "--type <generic|tool|learning|red-team|...> [--output path]"
---

# Bundle Management Workflow

Complete workflow for creating and managing context bundles.

## Step 1: Determine Bundle Type
Available types (registered in base manifests):

| Type | Description |
|:-----|:------------|
| `generic` | One-off bundles, no core context |
| `context-bundler` | Context bundler tool export |
| `learning` | Protocol 128 learning seals |
| `learning-audit-core` | Learning audit packets |
| `red-team` | Technical audit snapshots |
| `guardian` | Session bootloader context |
| `bootstrap` | Fresh repo onboarding |

## Step 2: Initialize Manifest
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manifest_manager.py init --type [TYPE] --bundle-title "[Title]"
```

## Step 3: Add/Remove Files
```bash
# Add
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manifest_manager.py add --path "[file.md]" --note "Description"

# Remove
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manifest_manager.py remove --path "[file.md]"
```

## Step 4: Execute Bundle
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/manifest_manager.py bundle -o [OutputPath]
```

Or directly with bundle.py:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/bundle.py [ManifestPath] -o [OutputPath]
```

## Step 5: Verify
```bash
ls -lh [OutputPath]
```

## Recursive Loop (Protocol 128)
For learning workflows, iterate:
1. **Research/Analysis**: Agent performs work
2. **Modify Manifest**: Add new findings via `add`
3. **Rebundle**: Generate updated context
4. **Repeat** until complete
5. **Seal**: When finished
