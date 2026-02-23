---
description: Initialize a new context bundle manifest from a base template
argument-hint: "--target <NAME> --type <generic|tool|context-bundler>"
---

# Initialize Bundle Manifest

Create a new `file-manifest.json` from a base template.

## Usage
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/manifest_manager.py init --target $0 --type $1
```

## Available Types
- `generic` — Empty manifest for ad-hoc bundling
- `tool` — Pre-populated manifest for tool distribution
- `context-bundler` — Self-bundling manifest (for distributing this tool)

## Behavior
1. Loads the base manifest template from `resources/base-manifests/`
2. Creates a new `file-manifest.json` with the given title
3. Populates default file entries based on the template type

## Example
```
/context-bundler:init --target MyAnalysis --type generic
```
