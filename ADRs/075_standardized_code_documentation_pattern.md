# ADR 075: Standardized Code Documentation Pattern

## Status
Accepted

## Context
Code documentation across the project fleet has been inconsistent, ranging from inline comments to docstrings. This inconsistency makes it difficult for both human developers and AI agents to quickly understand the purpose, arguments, and return values of methods, especially when analyzing code structure via AST or simple file reading.

A pattern has emerged in `rag_cortex` and recent `sanctuary_git` refactoring that uses a distinctive ASCII header block *above* method definitions. This pattern has proven highly effective for rapid scanning and structured context extraction.

## Decision
We will standardize on the "Header-Above-Definition" documentation pattern for all Python methods and comprehensive file headers for all source files.

### 1. File Headers
Every source file MUST begin with a file-level header block:

```python
#============================================
# path/to/file.py
# Purpose: Brief description of the file's responsibility.
# Role: Architectural layer assignment (e.g., Business Logic, Data Layer)
# Used by: List of consumers or "Main service entry point"
#============================================
```

### 2. Method/Function Headers
Every non-trivial method or function MUST be preceded by a structured header block. Code immediately following the header block defines the function.

**Required Fields:**
- `Method` / `Function`: The name of the function.
- `Purpose`: A clear, conciliatory description of what it does.
- `Args` (if applicable): List of arguments and their purpose.
- `Returns` (if applicable): Description of return value.
- `Raises` (if applicable): Expected exceptions.

**Format:**
```python
    #============================================
    # Method: my_function_name
    # Purpose: Describes what this function does concisely.
    # Args:
    #   arg1: Description of argument 1
    #   arg2: Description of argument 2
    # Returns: Description of return value
    # Raises: specific_exception if condition met
    #============================================
    def my_function_name(self, arg1: str, arg2: int) -> bool:
        ...
```

### 3. Docstrings
Python docstrings (`"""..."""`) should still be used *inside* the function for IDE support, but the ASCII header block is the primary source of high-level architectural documentation.

## Consequences
### Positive
- **Scannability:** High-visibility headers make scanning files for functionality significantly faster.
- **AI Readability:** Distinct delimiters (`#===`) helps LLMs accurately parse code segments and understand intent without reading implementation details.
- **Consistency:** Uniform look and feel across the codebase.

### Negative
- **Verbosity:** Increases the vertical line count of files.
- **Maintenance:** Requires updating both the header and the signature if arguments change (though this redundancy enforces mindfulness).
