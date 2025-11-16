# Sanctuary Task Schema v1.0

**Status:** Canonical
**Last Updated:** 2025-11-15
**Authority:** Protocol 115 (The Tactical Mandate Protocol)

---

## Overview

This document defines the canonical schema for all task files in Project Sanctuary. All tasks must conform to this schema to ensure consistency, verifiability, and proper integration with the task management system.

## Schema Structure

### 1. File Naming Convention
- **Format:** `XXX_descriptive_title.md`
- **XXX:** Three-digit, zero-padded sequential number (e.g., `001`, `013`)
- **Title:** Lowercase, underscore-separated descriptive name
- **Example:** `013_define_canonical_task_schema.md`

### 2. Header Block (Required)
All tasks must begin with a standardized header block containing metadata:

```markdown
# TASK: [Human-readable title]

**Status:** [backlog | todo | in-progress | complete | blocked]
**Priority:** [Critical | High | Medium | Low]
**Lead:** [Assigned agent/person, e.g., GUARDIAN-01, Unassigned]
**Dependencies:** [Task references, e.g., "Blocks #005", "Requires #012"]
**Related Documents:** [Protocol/file references, e.g., "P115, TASKS/done/005_forge_protocol_115_tactical_mandate.md"]

---
```

#### Field Definitions

**Status:**
- `backlog`: Initial state for newly created tasks
- `todo`: Task is prioritized and ready for work
- `in-progress`: Active work in progress
- `complete`: All acceptance criteria met
- `blocked`: Cannot proceed due to dependencies or issues

**Priority:**
- `Critical`: Blocks other critical work, immediate attention required
- `High`: Important for project progress, should be addressed soon
- `Medium`: Standard priority, address when resources available
- `Low`: Nice-to-have, address when higher priorities are complete

**Lead:**
- Primary responsible party or agent
- Use "Unassigned" if not yet assigned
- Examples: `GUARDIAN-01`, `Unassigned`, `AI-Assistant`

**Dependencies:**
- References to other tasks using `#XXX` format
- Use descriptive prefixes: "Blocks", "Requires", "Depends on", "Follows"
- Multiple dependencies separated by commas

**Related Documents:**
- Links to protocols, files, or external references
- Use shorthand notation (e.g., "P115" for protocols)
- Include full paths for project files

### 3. Content Sections (Required)

#### 3.1 Objective Section
```markdown
## 1. Objective

[Clear, concise statement describing the "what" and "why" of this task. What is the desired end-state upon successful completion?]
```

- Must be a single, focused paragraph
- Explain both the deliverable and the rationale
- Should answer: What problem does this solve?

#### 3.2 Deliverables Section
```markdown
## 2. Deliverables

1. [Concrete, verifiable artifact or outcome #1]
2. [Concrete, verifiable artifact or outcome #2]
3. [Additional deliverables as needed]
```

- Numbered list of specific, measurable outputs
- Each deliverable should be independently verifiable
- Focus on artifacts, not activities

#### 3.3 Acceptance Criteria Section
```markdown
## 3. Acceptance Criteria

- [Specific condition that must be true for task completion]
- [Another measurable completion condition]
- [Additional criteria as needed]
```

- Bulleted list of binary (yes/no) conditions
- Must be objectively verifiable
- Should align with deliverables but focus on validation

### 4. Optional Sections

#### 4.1 Notes Section
```markdown
## Notes

[Any additional context, implementation details, or considerations]
```

- Use for complex tasks requiring additional explanation
- May include subsections with `###` headers
- Not required but recommended for complex tasks

#### 4.2 Implementation Details
```markdown
## Implementation Details

[Technical approach, design decisions, or step-by-step plans]
```

- For complex technical tasks
- May include code snippets, diagrams, or pseudocode

## Validation Rules

### Required Fields
- All header fields must be present
- Status, Priority, and Lead are mandatory
- Dependencies and Related Documents may be "None" if not applicable
- All three main content sections (Objective, Deliverables, Acceptance Criteria) are required

### Content Standards
- Use proper Markdown formatting
- Maintain consistent indentation
- Use backticks for file paths and code references
- Keep line lengths reasonable (<100 characters where possible)

### Naming Standards
- Task numbers must be obtained via `tools/scaffolds/get_next_task_number.py`
- Titles should be descriptive but concise
- Use lowercase with underscores for file names

## Examples

### Minimal Task
```markdown
# TASK: Update README

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** README.md

---

## 1. Objective

Update the project README with current setup instructions.

## 2. Deliverables

1. README.md updated with current dependency versions.
2. Installation section includes all required steps.

## 3. Acceptance Criteria

- README.md contains accurate setup instructions.
- All links in README are functional.
```

### Complex Task
```markdown
# TASK: Implement Advanced Feature

**Status:** in-progress
**Priority:** High
**Lead:** GUARDIAN-01
**Dependencies:** "Requires #012, Blocks #015"
**Related Documents:** "P115, src/advanced_feature.py"

---

## 1. Objective

Implement the advanced feature to enhance system capabilities and address user requirements for improved performance.

## 2. Deliverables

1. New module `src/advanced_feature.py` with complete implementation.
2. Unit tests in `tests/test_advanced_feature.py` with >90% coverage.
3. Documentation updated in `docs/advanced_features.md`.
4. Integration tests pass in CI pipeline.

## 3. Acceptance Criteria

- All unit tests pass locally and in CI.
- Feature works end-to-end in development environment.
- Documentation is complete and accurate.
- Code review approval obtained.

## Notes

### Technical Approach
Using async/await pattern for performance optimization. Database queries will be batched to reduce round trips.
```

## Schema Evolution

This schema may be updated through the standard task process:
1. Create a new task proposing schema changes
2. Implement and test changes
3. Update this document
4. Update Protocol 115 if needed

All schema changes must maintain backward compatibility where possible.