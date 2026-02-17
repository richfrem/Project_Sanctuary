---
description: Check which .mmd diagrams have outdated or missing images
argument-hint: "--input <file.mmd|dir/> [--svg]"
---

# Check Outdated Diagrams

Report which `.mmd` files need re-rendering (image missing or older than source).

## Usage
```bash
# Check default directory
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/export_mmd_to_image.py --check

# Check specific directory
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/export_mmd_to_image.py -i docs/diagrams/ --check
```

## Output
Lists all diagrams that need rendering — does not modify any files.
