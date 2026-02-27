# Task Management Guide

**Status:** Active  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary uses a structured task management system in the `tasks/` directory with three workflow states.

## Directory Structure

```
tasks/
├── MASTER_PLAN.md      # High-level roadmap
├── backlog/            # Planned but not started
├── todo/               # Ready for active work
└── done/               # Completed tasks
```

## Task Lifecycle

```
backlog/ → todo/ → done/
```

| State | Purpose |
|-------|---------|
| `backlog/` | Ideas, planned work, not yet prioritized |
| `todo/` | Active work queue, ready to execute |
| `done/` | Completed tasks (archived for reference) |

## Task File Format

tasks use a standardized format with metadata header:

```markdown
# Task: [Title]

**Created:** YYYY-MM-DD
**Priority:** High/Medium/Low
**Status:** Pending/In Progress/Done

## Context
[Why this task exists]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

## Related Agent Plugin Integration Server

The **Task Agent Plugin Integration** server provides programmatic access to task management:
- [[README|Task Agent Plugin Integration README]]

## Related Documents

- [[MASTER_PLAN|MASTER_PLAN.md]] - Project roadmap
- [[101_The_Doctrine_of_the_Unbreakable_Commit|Protocol 101]] - Tests must pass before commit
