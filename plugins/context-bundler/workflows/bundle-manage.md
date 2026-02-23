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
python3 plugins/context-bundler/scripts/manifest_manager.py init --type [TYPE] --bundle-title "[Title]"
```

## Step 3: Add Files to Manifest (optional)
To add files to the manifest (uses `files` array by default):
// turbo
```bash
python3 plugins/context-bundler/scripts/manifest_manager.py add --path "[file.md]" --note "Description of file"
```

To remove files:
// turbo
```bash
python3 plugins/context-bundler/scripts/manifest_manager.py remove --path "[file.md]"
```

## Step 4: Validate Manifest (recommended)
// turbo
```bash
python3 tools/retrieve/bundler/validate.py [ManifestPath]
```

## Step 5: Execute Bundle
// turbo
```bash
python3 plugins/context-bundler/scripts/manifest_manager.py bundle -o [OutputPath]
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
- `tools/retrieve/bundler/validate.py`: Manifest validation tool

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
