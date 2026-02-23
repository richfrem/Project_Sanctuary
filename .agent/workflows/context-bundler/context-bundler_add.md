---
description: Add a file to the current context bundle manifest
argument-hint: "--path <file_path> --note \"description of this file\""
---

# Add File to Manifest

Add a file entry to the active bundle manifest.

## Usage
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/manifest_manager.py add --path $0 --note "$1"
```

## Behavior
1. Checks if the file already exists in the manifest (deduplication)
2. Adds the file path and description note to `file-manifest.json`
3. The file will be included in the next `bundle` operation

## Related Commands
- `/context-bundler:init` — Create a new manifest first
- `/context-bundler:bundle` — Generate the bundle after adding files
