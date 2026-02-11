---
trigger: always_on
---

# Coding Conventions & Documentation Standards
**Project Sanctuary**

## Overview

This document defines coding standards project sanctuary across Python, JavaScript/TypeScript, and C#/.NET codebases.

## 1. Documentation Standards


### Dual-Layer Documentation
To serve both AI assistants (for code analysis) and developer IDE tools (hover-tips, IntelliSense), code should include:
- **External Comments/Headers**: Brief, scannable descriptions above functions/classes
- **Internal Docstrings**: Detailed documentation within functions/classes

**Placement**: Comments sit immediately above the function/class definition. Docstrings sit immediately inside.

## 2. File-Level Headers

### Python Files
Every Python source file should begin with a header describing its purpose in detail.

**Canonical Template:** [`.agent/templates/code/python-tool-header-template.py`](../../templates/code/python-tool-header-template.py)

#### Basic Python Header
```python
#!/usr/bin/env python3
"""
[Script Name]
=====================================

Purpose:
    [Detailed description of what the script does and its role in the system]

Layer: [Investigate / Codify / Curate / Retrieve]

Usage:
    python script.py [args]
    python script.py --help

Related:
    - [Related Script 1]
    - [Related Policy/Document]
"""
```

#### Extended Python CLI/Tool Header (Gold Standard)
For CLI tools and complex scripts (especially in `tools/` and `scripts/` directories), use this comprehensive format:

```python
#!/usr/bin/env python3
"""
{{script_name}} (CLI)
=====================================

Purpose:
    Detailed multi-paragraph description of what this script does.
    Explain its role in the system and when it should be used.
    
    This tool is critical for [context] because [reason].

Layer: Investigate / Codify / Curate / Retrieve  (Pick one)

Usage Examples:
    python tools/path/to/script.py --target JCSE0004 --deep
    python tools/path/to/script.py --target MY_PKG --direction upstream --json

Supported Object Types:
    - Type 1: Description
    - Type 2: Description

CLI Arguments:
    --target        : Target Object ID (required)
    --deep          : Enable recursive/deep search (optional)
    --json          : Output in JSON format (optional)
    --direction     : Analysis direction: upstream/downstream/both (default: both)  

Input Files:
    - File 1: Description
    - File 2: Description

Output:
    - JSON to stdout (with --json flag)
    - Human-readable report (default)

Key Functions:
    - load_dependency_map(): Loads the pre-computed dependency inventory.
    - find_upstream(): Identifies incoming calls (Who calls me?).
    - find_downstream(): Identifies outgoing calls (Who do I call?).
    - deep_search(): Greps source code for loose references.

Script Dependencies:
    - dependency1.py: Purpose
    - dependency2.py: Purpose

Consumed by:
    - parent_script.py: How it uses this script
"""
```

> **Note:** This extended format enables automatic description extraction by `manage_tool_inventory.py`. The inventory tool reads the "Purpose:" section.

### TypeScript/JavaScript Files
For utility scripts and processing modules, use comprehensive headers:

```javascript
/**
 * path/to/file.js
 * ================
 * 
 * Purpose:
 *   Brief description of the component's responsibility.
 *   Explain the role in the larger system.
 * 
 * Input:
 *   - Input source 1 (e.g., XML files, JSON configs)
 *   - Input source 2
 * 
 * Output:
 *   - Output artifact 1 (e.g., Markdown files)
 *   - Output artifact 2
 * 
 * Assumptions:
 *   - Assumption about input format or state
 *   - Assumption about environment or dependencies
 * 
 * Key Functions/Classes:
 *   - functionName() - Brief description
 *   - ClassName - Brief description
 * 
 * Usage:
 *   // Code example showing how to use this module
 *   import { something } from './file.js';
 *   await something(params);
 * 
 * Related:
 *   - relatedFile.js (description)
 *   - relatedPolicy.md (description)
 * 
 * @module ModuleName
 */
```

For React components (shorter form):
```typescript
/**
 * path/to/Component.tsx
 * 
 * Purpose: Brief description of the component's responsibility.
 * Layer: Presentation layer (React component).
 * Used by: Parent components or route definitions.
 */
```

### C#/.NET Files
```csharp
// path/to/File.cs
// Purpose: Brief description of the class's responsibility.
// Layer: Service layer / Data access / API controller.
// Used by: Consuming services or controllers.
```

## 3. Function & Method Documentation

### Python Functions
Every non-trivial function should include clear documentation:

```python
def process_form_xml(xml_path: str, output_format: str = 'markdown') -> Dict[str, Any]:
    """
    Converts Oracle Forms XML to the specified output format.
    
    Args:
        xml_path: Absolute path to the Oracle Forms XML file.
        output_format: Target format ('markdown', 'json'). Defaults to 'markdown'.
    
    Returns:
        Dictionary containing converted form data and metadata.
    
    Raises:
        FileNotFoundError: If xml_path does not exist.
        XMLParseError: If the XML is malformed.
    """
    # Implementation...
```

### TypeScript/JavaScript Functions
```typescript
/**
 * Fetches RCC data from the API and updates component state.
 * 
 * @param rccId - The unique identifier for the RCC record
 * @param includeHistory - Whether to include historical data
 * @returns Promise resolving to RCC data object
 * @throws {ApiError} If the API request fails
 */
async function fetchRCCData(rccId: string, includeHistory: boolean = false): Promise<RCCData> {
  // Implementation...
}
```

### C#/.NET Methods
```csharp
/// <summary>
/// Retrieves RCC details by ID with optional related entities.
/// </summary>
/// <param name="rccId">The unique identifier for the RCC record.</param>
/// <param name="includeParticipants">Whether to include participant data.</param>
/// <returns>RCC entity with requested related data.</returns>
/// <exception cref="NotFoundException">Thrown when RCC ID is not found.</exception>
public async Task<RCC> GetRCCDetailsAsync(int rccId, bool includeParticipants = false)
{
    // Implementation...
}
```



## 4. Language-Specific Standards

### Python Standards (PEP 8)
- **Type Hints**: Use type annotations for all function signatures: `def func(name: str) -> Dict[str, Any]:`
- **Naming Conventions**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Docstrings**: Follow PEP 257 (Google or NumPy style)
- **Line Length**: Max 100 characters
- **Imports**: Group by stdlib, third-party, local (separated by blank lines)

### TypeScript/JavaScript Standards
- **Type Safety**: Use TypeScript for all new code
- **Naming Conventions**:
  - `camelCase` for functions and variables
  - `PascalCase` for classes, components, and interfaces
  - `UPPER_SNAKE_CASE` for constants
- **React Components**: Use functional components with hooks
- **Props**: Define interfaces for all component props
- **Exports**: Use named exports for utilities, default exports for pages/components

### C#/.NET Standards
- **Naming Conventions**:
  - `PascalCase` for public members, classes, methods
  - `camelCase` for private fields (with `_` prefix: `_privateField`)
  - `PascalCase` for properties
- **Documentation**: XML documentation comments for public APIs
- **Async/Await**: Use async patterns consistently
- **LINQ**: Prefer LINQ for collection operations

## 5. Code Organization

### Refactoring Threshold
If a function/method exceeds 50 lines or has more than 3 levels of nesting:
- Extract helper functions/methods
- Consider breaking into smaller, focused units
- Use meaningful names that describe purpose

### Comment Guidelines
- **Why, not What**: Explain business logic and decisions, not obvious code
- **TODO Comments**: Include ticket/issue numbers: `// TODO(#123): Add error handling`
- **Legacy Notes**: Mark Oracle Forms business logic: `// Legacy APPL4 rule: RCC_USER can only view own agency`

### Module Organization
```
module/
├── __init__.py          # Module exports
├── models.py            # Data models / DTOs
├── services.py          # Business logic
├── repositories.py      # Data access
├── utils.py             # Helper functions
└── constants.py         # Constants and enums
```

## 6. Tool Inventory Integration

### Mandatory Registration
All Python scripts in the `tools/` directory **MUST** be registered in `tools/tool_inventory.json`. This is enforced by the [Tool Inventory Policy](tool_inventory_policy.md).

### After Creating/Modifying a Script
```bash
# Register new script
python tools/curate/inventories/manage_tool_inventory.py add --path "tools/path/to/script.py"

# Update existing script description (auto-extracts from docstring)
python tools/curate/inventories/manage_tool_inventory.py update --path "tools/path/to/script.py"

# Verify registration
python tools/curate/inventories/manage_tool_inventory.py audit
```

### Docstring Auto-Extraction & RLM
The `distiller.py` engine uses the docstring as the primary input for the "Purpose" field in the RLM Cache and Tool Inventory. Explicit, high-quality docstrings ensure the Agent can discover your tool.

### Pre-Commit Checklist
Before committing changes to `tools/`:
- [ ] Script has proper header (Basic or Extended format)
- [ ] Script is registered in `tool_inventory.json`
- [ ] `manage_tool_inventory.py audit` shows 0 untracked scripts

### Related Policies
- [Tool Inventory Policy](tool_inventory_policy.md) - Enforcement triggers
- [Documentation Granularity Policy](documentation_granularity_policy.md) - Task tracking
- [/tool-inventory-manage](../../workflows/tool-inventory-manage.md) - Complete tool registration workflow

## 7. Manifest Schema (ADR 097)

When creating or modifying manifests in `.agent/learning/`, use the simple schema:

```json
{
    "title": "Bundle Name",
    "description": "Purpose of the bundle.",
    "files": [
        {"path": "path/to/file.md", "note": "Brief description"}
    ]
}
```

**Do NOT use** the deprecated `core`/`topic` pattern (ADR 089 v2.0).

**Version**: 2.0 | **Updated**: 2026-02-01