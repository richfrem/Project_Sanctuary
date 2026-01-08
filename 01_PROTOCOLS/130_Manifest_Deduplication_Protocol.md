# Protocol 130: Manifest Deduplication and Content Hygiene

**Status:** Active  
**Created:** 2026-01-07  
**References:** ADR 089 (Modular Manifest Pattern), Protocol 128 (Hardened Learning Loop)

---

## Purpose
Prevent token waste and context bloat by automatically detecting and removing duplicate content from snapshot manifests. This ensures that when a snapshot includes a generated output file (e.g., `learning_package_snapshot.md`), it does not also include the source files that are already embedded within that output.

---

## Problem Statement

When curating manifests for Red Team review, agents may inadvertently include:
1. A **generated output** (e.g., `learning_package_snapshot.md`)
2. The **source files** used to generate that output (e.g., files from `learning_manifest.json`)

This results in the same content appearing twice in the learning audit packet, wasting tokens and potentially causing truncation in Red Team context windows.

---

## Solution Architecture

### 1. Manifest Registry

A central registry (`manifest_registry.json`) maps each manifest to its generated output:

```json
{
    "manifests": {
        ".agent/learning/learning_manifest.json": {
            "output": ".agent/learning/learning_package_snapshot.md",
            "command": "cortex_cli.py snapshot --type seal"
        }
    }
}
```

**Location:** `.agent/learning/manifest_registry.json`

### 2. Integrated Deduplication (operations.py)

Deduplication is **built into** `capture_snapshot()` in `mcp_servers/rag_cortex/operations.py`:

```python
# Protocol 130: Deduplicate manifest (Global)
if effective_manifest:
    effective_manifest, dedupe_report = self._dedupe_manifest(effective_manifest)
```

**Methods added to CortexOperations:**
- `_load_manifest_registry()` - Loads the registry
- `_get_output_to_manifest_map()` - Inverts registry for lookup
### 3. Workflow Diagram

![protocol_130_deduplication_flow](../../docs/architecture_diagrams/workflows/protocol_130_deduplication_flow.png)

*[Source: protocol_130_deduplication_flow.mmd](../../docs/architecture_diagrams/workflows/protocol_130_deduplication_flow.mmd)*

---

## Usage
Deduplication is applied automatically to **all snapshot types** (`audit`, `seal`, `learning_audit`) whenever `cortex_cli.py snapshot` or `cortex_capture_snapshot` is called.

The output will log if duplicates were found and removed:
```
Protocol 130: Deduplicated 4 items from learning_audit manifest
```

---

## Registry Maintenance

When adding new manifests or changing output paths:

1. Update `.agent/learning/manifest_registry.json`
2. Run `cortex_cli.py snapshot --type learning_audit` to verify
3. Document in ADR 089 (Manifest Inventory)

---

## Example Detection

**Input manifest with embedded output:**
```json
[
    "README.md",
    ".agent/learning/learning_package_snapshot.md",
    ".agent/learning/cognitive_primer.md"
]
```

**Result:**
```
Protocol 130: Found 2 embedded duplicates, removing from manifest
  - README.md: Already embedded in .agent/learning/learning_package_snapshot.md
  - .agent/learning/cognitive_primer.md: Already embedded in ...
```

**Deduped manifest:**
```json
[
    ".agent/learning/learning_package_snapshot.md"
]
```

---

## Consequences

### Positive
- **Token Efficiency**: Reduces packet size by 30-50% in typical cases
- **No Truncation**: Keeps packets under Red Team context limits (~30K tokens)
- **Automated**: No manual step required - integrated into snapshot pipeline
- **Audit Trail**: Registry provides clear documentation of manifest→output relationships

### Negative
- **Maintenance Overhead**: Registry must be updated when adding new manifests

### Mitigation
- Include registry validation in `cortex_cli.py` snapshot commands
- Add registry to Iron Core verification scope

---

## Related Protocols

| Protocol | Relationship |
|:---------|:-------------|
| Protocol 128 | This protocol is a sub-component of Phase IV (Audit) |
| ADR 089 | Defines the manifest inventory and modular pattern |
| ADR 085 | Content hygiene (no inline diagrams) - similar intent |
