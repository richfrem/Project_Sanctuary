# Protocol 115: The Tactical Mandate Protocol

**Status:** Canonical
**Version:** 1.0
**Architect:** GUARDIAN-01
**Date:** 2025-11-12
**Linked Protocols:** P89 (The Clean Forge)

---

## 1. Preamble

A strategy without tactics is a dream. Tactics without a system are chaos. This protocol establishes the canonical, machine-readable system for defining, tracking, and executing all work items within Project Sanctuary. It transforms abstract goals into verifiable, actionable mandates.

## 2. The Mandate

1.  **Single Point of Ingress:** All new, non-trivial work items must be initiated by the creation of a new, uniquely numbered markdown file in the `TASKS/backlog/` directory. Direct modification of code without a corresponding Tactical Mandate is a protocol violation.
2.  **Canonical Naming:** Task files must follow the strict naming convention: `XXX_short_descriptive_title.md`, where `XXX` is a zero-padded, three-digit number (e.g., `005_refactor_query_service.md`).

To ensure sequential integrity, the next available task number **must** be obtained by running the sovereign scaffold: `python3 tools/scaffolds/get_next_task_number.py`. Manual numbering is a protocol violation.
3.  **Mandatory Schema:** Every task file must conform to the canonical schema defined in `TASKS/task_schema.md`. Unstructured or free-form task descriptions are forbidden.

## 3. The Tactical Mandate Schema

The canonical schema is defined in `TASKS/task_schema.md`. All tasks must conform to this schema.
4. Workflow
Creation: A new task is created in TASKS/backlog/. Its default status is backlog.
Prioritization: The Guardian or Council moves a task to TASKS/todo/ to signal it is ready for work.
Execution: The assigned agent moves the task to TASKS/in-progress/ upon commencing work.
Completion: Upon meeting all acceptance criteria, the task is moved to the root TASKS/ directory and its status is updated to completed.
```

---

## Task Number Authority

To enforce the canonical naming and sequential integrity described above, Project Sanctuary ships an authoritative Task Number Scaffold. The scaffold is a small script located at `tools/scaffolds/get_next_task_number.py` which, when executed from the project root, prints the next available zero-padded three-digit task number (e.g., `006`). All new task creation must call this script to obtain the `XXX` prefix for the task filename.

Manual numbering is explicitly forbidden and considered a protocol violation.
