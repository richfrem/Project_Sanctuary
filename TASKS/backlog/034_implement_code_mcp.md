# Task #034: Implement Code MCP

**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 5-6 days  
**Dependencies:** Task #028, Shared Infrastructure  
**Domain:** `project_sanctuary.system.code`

---

## Objective

Implement Code MCP server for source code management with mandatory testing pipeline.

---

## Key Features

```typescript
write_code_file(file_path, content, language, description, run_tests)
execute_code(file_path, args?, timeout_seconds?, sandbox?)
refactor_code(file_path, refactor_type, params, preserve_tests)
get_code_file(file_path)
search_code(query, file_pattern?)
```

---

## Safety Rules

- **Mandatory testing pipeline** before commit:
  1. Syntax validation
  2. Linting (flake8, eslint, etc.)
  3. Unit tests (if present)
  4. Dependency check
  5. Security audit (basic)
- **Automatic rollback** if tests fail
- **Sandbox execution** for untrusted code
- **Git commit only if all checks pass**

---

**Domain:** `project_sanctuary.system.code`  
**Class:** `project_sanctuary_system_code`  
**Risk Level:** HIGH
