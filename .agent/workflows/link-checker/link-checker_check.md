---
description: Scan documentation for broken relative links and generate a report
argument-hint: "[target_directory] [--file specific_file.md]"
---

# Check Broken Links

Scan Markdown and text files for broken relative links. Generates a `broken_links.log` report.

## Usage
```bash
# Scan entire repository from current directory
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/check_broken_paths.py

# Scan a specific directory
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/check_broken_paths.py /path/to/docs

# Check a single file
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/check_broken_paths.py --file docs/README.md
```

## Behavior
1. Recursively scans `.md`, `.txt`, `.json`, `.markdown`, `.mmd` files
2. Detects broken Markdown links `[text](path)`, HTML `src`/`href`, and legacy paths
3. Ignores web links, anchors, and absolute paths
4. Outputs report to `broken_links.log` and console
