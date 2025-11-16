# Sanctuary ADR Schema v1.0

**Status:** Canonical
**Last Updated:** 2025-11-15
**Authority:** Task 014 (Establish Architecture Decision Records System)

---

## Overview

This document defines the canonical schema for Architecture Decision Records (ADRs) in Project Sanctuary. ADRs document important architectural decisions, their context, and consequences. All ADRs must conform to this schema to ensure consistency and completeness.

## Schema Structure

### 1. File Naming Convention
- **Format:** `XXX_decision_title.md`
- **XXX:** Three-digit, zero-padded sequential number (e.g., `001`, `012`)
- **Title:** Lowercase, underscore-separated descriptive name
- **Example:** `001_select_qwen2_model_architecture.md`

### 2. Header Block (Required)
All ADRs must begin with a standardized header block:

```markdown
# [Decision Title]

**Status:** [proposed | accepted | deprecated | superseded]
**Date:** YYYY-MM-DD
**Deciders:** [List of people/agents involved]
**Technical Story:** [Reference to related task or issue]

---

## Context

[Description of the forces at play, including technological, business, and operational concerns]

## Decision

[Specific decision made, with rationale]

## Consequences

### Positive
- [List of positive outcomes]

### Negative
- [List of negative outcomes or trade-offs]

### Risks
- [Potential risks and mitigation strategies]

### Dependencies
- [New dependencies introduced by this decision]
```

#### Field Definitions

**Status:**
- `proposed`: Decision is under consideration
- `accepted`: Decision has been implemented and is current
- `deprecated`: Decision is no longer recommended but may still be in use
- `superseded`: Decision has been replaced by a newer decision

**Date:**
- ISO format date when the decision was made
- For proposed ADRs, use the date of proposal

**Deciders:**
- Primary decision-makers or responsible parties
- Include roles and/or specific individuals
- Examples: `GUARDIAN-01`, `AI-Assistant`, `Technical Council`

**Technical Story:**
- Reference to the task, issue, or context that prompted this decision
- Use format like `#014` for tasks, or descriptive reference

### 3. Content Sections (Required)

#### 3.1 Context Section
```markdown
## Context

[Comprehensive description of the problem or situation that required a decision]
```

- Explain the business/technical problem
- Include relevant background information
- Describe constraints and requirements
- Reference any alternatives considered

#### 3.2 Decision Section
```markdown
## Decision

[Clear statement of what was decided and why]
```

- State the decision explicitly
- Provide rationale and justification
- Reference any supporting data or analysis
- Be specific and actionable

#### 3.3 Consequences Section
```markdown
## Consequences

### Positive
- [Measurable benefits and advantages]

### Negative
- [Drawbacks, costs, or trade-offs]

### Risks
- [Potential future issues and mitigation plans]

### Dependencies
- [New technical or operational dependencies]
```

- Document both positive and negative outcomes
- Include mitigation strategies for risks
- Be honest about trade-offs and costs

### 4. Optional Sections

#### 4.1 Alternatives Considered
```markdown
## Alternatives Considered

### Option 1: [Name]
- **Description:** [Brief explanation]
- **Pros:** [Advantages]
- **Cons:** [Disadvantages]
- **Why not chosen:** [Rationale]

### Option 2: [Name]
...
```

#### 4.2 Implementation Notes
```markdown
## Implementation Notes

[Technical details about how the decision was implemented]
```

#### 4.3 Future Considerations
```markdown
## Future Considerations

[How this decision might evolve or be revisited]
```

## Validation Rules

### Required Fields
- All header fields must be present
- Status, Date, and Deciders are mandatory
- All three main content sections (Context, Decision, Consequences) are required

### Content Standards
- Use proper Markdown formatting
- Maintain consistent indentation
- Use backticks for technical terms and file references
- Keep line lengths reasonable (<100 characters where possible)

### Decision Quality
- Decisions should be reversible where possible
- Include sufficient context for future maintainers
- Document assumptions explicitly
- Consider long-term implications

## Examples

### Technology Selection ADR
```markdown
# Select Primary Large Language Model

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, AI-Assistant
**Technical Story:** #007 (Model Training Requirements)

---

## Context

Project Sanctuary requires a high-quality LLM for cognitive architecture tasks. The model must support:
- Strong reasoning capabilities
- Efficient fine-tuning
- Local deployment options
- Active community support

Current alternatives include GPT, Claude, Llama, and Qwen models.

## Decision

We will use Qwen2-7B as our primary model architecture, with Qwen2-72B as an optional larger variant for complex reasoning tasks.

**Rationale:**
- Excellent performance on reasoning benchmarks
- Efficient parameter count for fine-tuning
- Strong multilingual support
- Active development by Alibaba Cloud
- Compatible with existing tooling

## Consequences

### Positive
- High-quality reasoning capabilities
- Good balance of performance vs. resource requirements
- Strong community support and documentation

### Negative
- Higher computational requirements than smaller models
- Potential vendor lock-in concerns with Alibaba Cloud

### Risks
- Model availability and licensing changes
- Mitigation: Maintain multi-model capability

### Dependencies
- Requires CUDA-compatible hardware for training
- Depends on Hugging Face transformers library
```

### Infrastructure ADR
```markdown
# Choose Vector Database for Semantic Search

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01
**Technical Story:** Initial system design

---

## Context

The mnemonic cortex requires efficient semantic search over large document collections. We need a vector database that supports:
- High-dimensional vector similarity search
- Metadata filtering
- Horizontal scaling
- ACID transactions
- Python integration

## Decision

We will use ChromaDB as our primary vector database for development and small-scale deployment, with PostgreSQL/pgvector as the production choice.

**Rationale:**
- ChromaDB: Simple, fast, good for development
- pgvector: Production-ready, scalable, transactional
- Both support the required feature set
- Smooth migration path from development to production

## Consequences

### Positive
- Fast development iteration with ChromaDB
- Production-ready scaling with pgvector
- Consistent API across environments

### Negative
- Additional complexity of dual database setup
- Migration overhead when moving to production

### Risks
- pgvector performance bottlenecks at scale
- Mitigation: Regular performance testing and optimization

### Dependencies
- ChromaDB Python client
- PostgreSQL with pgvector extension
- Database migration tooling
```

## ADR Lifecycle

1. **Proposal**: Create ADR with `proposed` status
2. **Discussion**: Gather feedback and alternatives
3. **Decision**: Update to `accepted` status with implementation
4. **Implementation**: Execute the decision
5. **Review**: Periodically assess if decision still holds
6. **Supersession**: Create new ADR if decision changes

## Tooling

- **Numbering:** Use `tools/scaffolds/get_next_adr_number.py` for sequential numbering
- **Validation:** ADRs should be reviewed by technical leads before acceptance
- **Storage:** All ADRs stored in `ADRs/` directory
- **References:** Link ADRs in related tasks and documentation

This schema ensures that all architectural decisions in Project Sanctuary are well-documented, reasoned, and maintainable for future development.