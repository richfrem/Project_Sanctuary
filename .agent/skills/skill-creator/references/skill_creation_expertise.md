# Skill Creation Expertise

This document synthesizes the best practices, workflows, and technical details for creating Agent Skills, based on a review of the `plugins/skill-creator` directory.

## Core Concepts

**Skills** are modular packages that extend an agent's capabilities. They consist of:
- **`SKILL.md`**: The brain. Contains metadata (frontmatter) and instructions (markdown).
- **Resources**: Scripts, references, and assets that support the skill.

### Progressive Disclosure
To maintain context window efficiency, skills use a three-level loading system:
1.  **Metadata**: (`name`, `description`) - Always loaded (~100 tokens).
2.  **Instructions**: (`SKILL.md` body) - Loaded only when the skill is activated.
3.  **Resources**: Loaded/executed only when specifically needed by the agent.

## Anatomy of a Skill

```
skill-name/
├── SKILL.md          # REQUIRED: Metadata and Instructions
├── scripts/          # OPTIONAL: Executable code (Python/Bash)
├── references/       # OPTIONAL: Documentation/Schemas (loaded on demand)
└── assets/           # OPTIONAL: Static files for output (templates, images)
```

### `SKILL.md` Frontmatter
Only `name` and `description` are required and strictly parsed.

```yaml
---
name: my-skill
description: Comprehensive explanation of WHAT the skill does and WHEN to use it. This is the trigger.
disable-model-invocation: false # Optional: Set true to force manual /command usage
user-invocable: true            # Optional: Set false to hide from slash commands
allowed-tools: Bash, Read       # Optional: Pre-authorize tools
---
```

## The Creation Process

Don't start from an empty file. The `skill-creator` plugin includes automation to enforce structure.

### 1. Initialization
Use the provided script to generate the correct directory structure and templates.

```bash
# From the plugin directory
python scripts/init_skill.py <skill-name> --path <destination-path>
```
*Creates: `SKILL.md`, `scripts/`, `references/`, `assets/` with examples.*

### 2. Implementation Strategy

| Component | Use When... | Examples |
|-----------|-------------|----------|
| **SKILL.md Body** | Defining the core workflow or decision logic. | "How to process a refund", "Which database to query" |
| **Scripts** | Logic is deterministic, fragile, or requires exact precision. | PDF manipulation, complex data transformation, API calls |
| **References** | Documentation is large (>500 lines) or conditional. | API Specs, Database Schemas, Company Policies |
| **Assets** | Files are needed in the final output. | PPTX templates, HTML boilerplate, Logos |

**Best Practice:** Keep `SKILL.md` under 500 lines. Move details to `references/`.

### 3. Packaging
Validate and bundle the skill for distribution.

```bash
python scripts/package_skill.py <skill-path>
```
*Validates structure/frontmatter -> Creates `<skill-name>.skill` zip.*

## Design Patterns

### Output Patterns
- **Template Pattern**: Provide strict markdown templates for reports to ensure consistency.
- **Example Pattern**: Provide Input/Output pairs for style transfer (e.g., commit messages).

### Workflow Patterns
- **Sequential**: Step 1 -> Step 2 -> Step 3 (Good for deterministic tasks).
- **Conditional**: "If X, read `references/x.md`. If Y, read `references/y.md`." (Good for multi-domain skills).

## Tips for High-Quality Skills
1.  **Conciseness**: Challenge every paragraph. Does Claude *really* need to know this?
2.  **Specific Triggers**: The `description` frontmatter is the **only** thing Claude sees initially. Be precise about *when* to activate.
3.  **Determinisim**: Use Python scripts for logic that shouldn't hallucinate (math, formatting).
