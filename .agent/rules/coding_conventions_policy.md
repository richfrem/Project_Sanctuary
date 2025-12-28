---
trigger: always_on
---

## 💻 Project Sanctuary: Coding Conventions & Documentation Rules

### 1. The Hybrid Documentation Mandate (ADR 075)

* **The Redundancy Principle**: To serve both AI Agents (scannability) and standard IDE tools (hover-tips), every code object requires two documentation layers: an external **Banner** and an internal **Docstring**.
* **Placement**: Banners must sit immediately above the `def` or `class` statement with no empty lines in between. Docstrings must sit immediately below the `def` or `class` line.

### 2. File-Level Mandatory Headers

Every source file MUST begin with a file-level header block to orient the agent to the module's role in the architecture:

```python
#============================================
# path/to/file.py
# Purpose: Brief description of the file's responsibility.
# Role: Architectural layer assignment (e.g., Business Logic, Data Layer).
# Used by: List of primary consumers or "Main service entry point."
#============================================

```

### 3. Method & Function Headers (The Signpost)

Every non-trivial method or function MUST be preceded by a structured ASCII banner. This is the primary source for high-level architectural skimming.

* **Required Fields**:
* `Method` / `Function`: The name of the function.
* `Purpose`: A clear, concise description of the internal logic.
* `Args`: List of arguments, their types, and their purpose.
* `Returns`: Description and type of the return value.
* `Raises`: List of expected exceptions.



### 4. Method Docstrings (The Manual)

Immediately following the function definition, you must include a standard PEP 257 docstring (`"""..."""`).

* **Purpose**: This ensures standard developer tools (VS Code, Cursor, `help()`) provide hover-state documentation and autocompletion hints.

### 5. Unified Implementation Example

```python
    #============================================
    # Method: process_snapshot
    # Purpose: Orchestrates the manifest generation and integrity check.
    # Args:
    #   session_id (str): The unique ID for the current learning loop.
    #   strict_mode (bool): If True, fails on any Tier-2 blindspots.
    # Returns: (dict) The validated session manifest.
    # Raises: IntegrityError if the Post-Flight Git check fails.
    #============================================
    def process_snapshot(self, session_id: str, strict_mode: bool = False) -> dict:
        """
        Orchestrates the manifest generation and integrity check.

        Args:
            session_id: Unique identifier for the audit session.
            strict_mode: Toggle for strict rejection of unmanifested changes.

        Returns:
            A dictionary containing the session metadata and file manifest.
        """
        # Implementation...

```

### 6. Modern Python Standards

* **Strict Typing**: All function signatures must use strict Python type hints (e.g., `-> List[str]`).
* **Variable Naming**: Use `snake_case` for functions/variables and `PascalCase` for classes (PEP 8).
* **Logic Decoupling**: If a method exceeds 40 lines of logic, it must be refactored into smaller, private helper methods (prefixed with `_`) to maintain scannability.
* **Context Tags**: Use specific tags to link code to the project state:
* `# TODO (Task-XXX):` Links directly to the `TASKS/` directory.
* `# NOTE (ADR-XXX):` Explains the architectural "why" behind a specific implementation.
* `# FIX-ONCE:` Marks core logic shared between the gateway and test suite.