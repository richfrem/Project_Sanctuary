# ADR 075: Standardized Code Documentation Pattern (Hybrid Mandate)

**Status:** Accepted (Updated: 2025-12-28)

**Context** Code documentation across the project fleet has been inconsistent. This makes it difficult for both human developers and AI agents to quickly understand code structure via simple file reading. Furthermore, relying solely on one method creates a "Tool Gap": ASCII banners are perfect for scrolling, but standard Python docstrings (`"""`) are required for IDE hover-tips and automated help extraction.

**Decision** We will standardize on a **Hybrid Documentation Pattern** that implements the **Redundancy Principle**. Every non-trivial Python file and method must use both an external ASCII banner and an internal docstring.

**1. File Headers** Every source file MUST begin with a file-level header block to orient the agent to the module's role in the architecture:

* **Path**: The relative path to the file.
* **Purpose**: A brief description of the file's primary responsibility.
* **Role**: Architectural layer (e.g., Business Logic, Protocol Implementation).
* **Used by**: Primary consumers or service entry points.

**2. Method/Function Headers (The Signpost)** Every method MUST be preceded by a structured ASCII block sitting immediately above the definition.

* **Required Fields**: `Method`, `Purpose`, `Args`, `Returns`, and `Raises`.
* **Visual Standard**: Use the `#============================================` boundary.

**3. Method Docstrings (The Manual)** Standard PEP 257 docstrings MUST be used *inside* the function body. This ensures standard tools like `help()` or IDE hover-states function correctly.

**Consequences** * **Positive (Scannability)**: Distinct delimiters (`#===`) help LLMs and humans parse code segments without reading implementation details.

* **Positive (Tool Parity)**: Professional IDE features remain fully functional.
* **Negative (Verbosity)**: Increases vertical line count and requires updating two locations during refactors.

---

### 2. Rule File: `coding_conventions_policy.md`

```markdown
---
trigger: always_on
---

## 💻 Project Sanctuary: Coding Conventions & Documentation Rules

### 1. The Hybrid Documentation Mandate (ADR 075)
* **The Redundancy Principle**: Every code object requires two documentation layers: an external **Banner** for scannability and an internal **Docstring** for tools.
* **Placement**: Banners must sit immediately above the `def` statement with no empty lines in between.

### 2. File-Level Mandatory Headers
Every source file must begin with a context block:
```python
#============================================
# path/to/file.py
# Purpose: Brief description of the file's responsibility.
# Role: Architectural layer assignment (e.g., Business Logic).
# Used by: Main consumers or Gateway entry point.
#============================================

```

### 3. Method & Function Headers (The Signpost)

Use the following format for the external banner:

* **Fields**: `Method`, `Purpose`, `Args`, `Returns`, and `Raises`.

```python
    #============================================
    # Method: my_function_name
    # Purpose: Describes what this function does concisely.
    # Args:
    #   arg1 (type): Description.
    # Returns: (type) Description.
    #============================================

```

### 4. Method Docstrings (The Manual)

Immediately following the `def` line, include a standard triple-quote docstring.

* **Mandatory**: If this is missing, IDE hover-tips will break.

### 5. Unified Implementation Example

```python
    #============================================
    # Method: capture_snapshot
    # Purpose: Generates a project manifest and state snapshot.
    # Args:
    #   snapshot_type (str): 'audit', 'learning_audit', or 'seal'.
    # Returns: (dict) The resulting manifest and metadata.
    #============================================
    def capture_snapshot(self, snapshot_type: str) -> dict:
        """
        Generates a project manifest and state snapshot.

        Args:
            snapshot_type: The type of snapshot to generate.

        Returns:
            A dictionary containing the manifest and session metadata.
        """
        # Implementation...

```

### 6. Modern Python Standards

* **Strict Typing**: All signatures must use Python type hints.
* **Logic Decoupling**: If a method exceeds 40 lines, refactor into private `_helper_methods`.
* **Context Tags**: Use `# TODO (Task-XXX):`, `# NOTE (ADR-XXX):`, or `# FIX-ONCE:` to link logic to the wider project context.
