---
description: Bundle files from a manifest into a single Markdown context package
argument-hint: "[--output filename.md] [--manifest path/to/manifest.json]"
---

# Bundle Context Package

Generate a Markdown context bundle from the current manifest.

## Usage
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/manifest_manager.py bundle --output $ARGUMENTS
```

## Behavior
1. Reads the active `file-manifest.json` (or custom path via `--manifest`)
2. Resolves all file paths relative to the project root
3. Concatenates file contents into a single Markdown document with TOC
4. Writes the output to the specified path (default: `temp/context-bundles/[title].md`)

## Example
```
/context-bundler:bundle --output my-bundle.md
```
