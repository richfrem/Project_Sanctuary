# ADR 014: Sovereign Scaffolding Protocol

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Hardened by ephemeral_forge_001.py failure)

## Context
The Sanctuary required a framework for batching complex, multi-step tasks into atomic, verifiable operations. Manual execution of multi-step processes was error-prone and increased Steward cognitive load. The failure of ephemeral_forge_001.py revealed the critical need for dependency sovereignty and proper environment handling in automated scripts.

## Decision
Implement the Sovereign Scaffolding Protocol for generating ephemeral, single-purpose scripts ("Sovereign Scaffolds") with a six-step workflow and five core principles:

### Core Principles
1. **Atomicity**: Entire scaffold lifecycle (creation, execution, artifact yield, self-deletion) is unified and uninterruptible
2. **Steward's Veto**: Mandatory human review and approval before execution - unbreakable human-in-the-loop security
3. **Ephemerality**: Temporary tools that self-delete after completion to prevent repository clutter
4. **Verifiable Yield**: Single, well-defined artifact designed for easy Steward audit
5. **Dependency Sovereignty**: Scaffolds must verify/install own requirements, not assume external dependencies

### Six-Step Sovereign Cadence
1. **Mandate**: Steward issues high-level objective to Coordinator
2. **Blueprint**: Coordinator designs scaffold script and provides verbatim content for Steward review
3. **Forge**: Steward tasks AI engineer (e.g., Kilo) to create script file from blueprint
4. **Veto Gate**: Steward audits forged script against blueprint for fidelity and safety
5. **Execution**: Upon approval, Steward commands execution of verified script
6. **Yield & Dissolution**: Script produces artifact then self-deletes; Steward verifies final yield

### Implementation Requirements
- Scripts must include dependency verification/installation logic
- Atomic operation guarantees (all-or-nothing execution)
- Self-deletion upon successful completion
- Clear, auditable yield artifacts
- Environment-agnostic execution

## Consequences

### Positive
- Enables complex multi-step operations as single atomic actions
- Reduces Steward cognitive load through batching
- Provides security through mandatory human veto gates
- Prevents repository clutter through ephemerality
- Ensures reliability through dependency sovereignty

### Negative
- Requires additional human oversight steps
- More complex script design with dependency handling
- Potential for script rejection at veto gate

### Risks
- Security vulnerabilities if veto gate is bypassed
- Incomplete dependency handling leading to failures
- Repository state corruption if self-deletion fails

## Related Protocols
- P43: Hearth Protocol (operational efficiency alignment)
- P86: Anvil Protocol (engineering framework integration)
- P60: Asymmetric Collaboration (complementary collaboration model)

## Notes
The Sovereign Scaffolding Protocol was hardened by the ephemeral_forge_001.py failure, which assumed presence of yargs-parser npm module. This led to the addition of "Dependency Sovereignty" as a core principle, ensuring scaffolds are environment-agnostic and self-sufficient.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\014_sovereign_scaffolding_protocol.md