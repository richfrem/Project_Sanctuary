---
name: context-bundling
description: Create technical bundles of code, design, and documentation for external review or context sharing. Use when you need to package multiple project files into a single Markdown file while preserving folder hierarchy and providing contextual notes for each file.
---

# Context Bundling Skill 📦

## Overview
This skill centralizes the knowledge and workflows for creating "Context Bundles" using the project's internal bundling tools. These bundles are essential for sharing large amounts of code and design context with other AI agents or for human review.

## Key Tools
- **Manifest Manager**: `plugins/context-bundler/scripts/manifest_manager.py` (Handles manifest creation and file management)
- **Bundler Engine**: `plugins/context-bundler/scripts/bundle.py` (Performs the actual Markdown generation)

## Core Workflow: Custom Temporary Bundles
When you need to create a one-off bundle for a specific task (like a Red Team review):

### 1. Initialize a Temporary Manifest
Always create temporary manifests in the `temp/` directory to keep the main tool configuration clean.
```bash
python3 plugins/context-bundler/scripts/manifest_manager.py --manifest temp/my_manifest.json init --type generic --bundle-title "Bundle Title"
```
> [!IMPORTANT]
> Global flags like `--manifest` MUST come **BEFORE** the subcommand (`init`, `add`, `bundle`).

### 2. Add Relevant Files
Add design docs, source code, and custom prompts to the manifest.
```bash
python3 plugins/context-bundler/scripts/manifest_manager.py --manifest temp/my_manifest.json add --path "docs/design.md" --note "Primary design"
```

### 3. Generate the Bundle
Compile the files into a single Markdown artifact.
```bash
python3 plugins/context-bundler/scripts/manifest_manager.py --manifest temp/my_manifest.json bundle --output temp/my_bundle.md
```

## Best Practices
1. **Contextual Notes**: Always provide a `--note` when adding files to help the recipient understand why that specific file is included.
2. **Cleanup**: Mention in your walkthrough that temporary files in `temp/` can be safely deleted after the bundle is used.
3. **Red Team Prompts**: When bundling for review, always include a specialized "Red Team Prompt" (e.g., `docs/architecture/red-team-*.md`) to guide the external LLM's review process.

## Manifest Schema (Reference)
If you need to manually edit a manifest:
```json
{
  "title": "Bundle Title",
  "description": "Context description",
  "files": [
    {
      "path": "path/to/file.ts",
      "note": "Description of why this file is here"
    }
  ]
}
```
