# Automated Script Protocol for Complex Tasks

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** AI Council (Developed from script failure experience)
**Technical Story:** Framework for reliable automated task execution

---

## Context

Our project needed a framework for handling complex, multi-step tasks as single, reliable operations. Manual execution of multi-step processes was error-prone and increased developer workload. The failure of a temporary script revealed the critical need for proper dependency management and environment handling in automated tools.

## Decision

We will implement the Automated Script Protocol for generating temporary, single-purpose scripts ("Automated Scripts") with a six-step workflow and five core principles:

### Core Principles
1. **Complete Operations**: Entire script lifecycle (creation, execution, result delivery, self-removal) is unified and cannot be interrupted
2. **Human Approval Required**: Mandatory human review and approval before execution - essential security control
3. **Temporary Tools**: Scripts that automatically delete themselves after completion to avoid repository clutter
4. **Clear Results**: Single, well-defined output designed for easy human verification
5. **Self-Contained**: Scripts must check for and install their own requirements, not depend on external setup

### Six-Step Process
1. **Request**: Developer gives high-level objective to AI assistant
2. **Design**: AI assistant creates script plan and provides exact content for developer review
3. **Create**: Developer asks AI engineer to create the script file from the plan
4. **Review Step**: Developer checks created script against plan for accuracy and safety
5. **Run**: Upon approval, developer commands execution of verified script
6. **Results and Cleanup**: Script produces output then deletes itself; developer verifies final results

### Implementation Requirements
- Scripts must include dependency checking/installation code
- All-or-nothing execution guarantees
- Automatic deletion after successful completion
- Clear, verifiable output files
- Environment-independent execution

## Consequences

### Positive
- Enables complex multi-step operations as single reliable actions
- Reduces developer workload through task grouping
- Provides security through required human approval steps
- Prevents repository clutter through automatic cleanup
- Ensures reliability through self-contained dependencies

### Negative
- Requires additional human review steps
- More complex script design with dependency management
- Possibility of script rejection during review

### Risks
- Security issues if approval step is skipped
- Failures from incomplete dependency handling
- Repository problems if automatic deletion fails

### Related Processes
- Operational efficiency process (coordination)
- Engineering framework process (integration)
- Collaborative development process (complementary model)

### Notes
This protocol was developed from the experience of a failed temporary script that assumed certain software was installed. This led to adding "Self-Contained" as a core principle, ensuring scripts work independently and manage their own requirements.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\014_sovereign_scaffolding_protocol.md