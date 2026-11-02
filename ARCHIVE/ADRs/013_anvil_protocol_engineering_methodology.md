# Engineering Methodology for AI-Assisted Development

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** AI Council (Developed from Memory System building experience)
**Technical Story:** Structured approach for AI-human collaborative coding

---

## Context

Our project needed a disciplined, verifiable method for collaborative AI-assisted development. Previous approaches lacked structure, leading to AI code with unverified assumptions and insufficient checking. Building the Memory System showed the need for a formal framework that treats AI as a "powerful tool" guided by human verification.

## Decision

We will implement the Engineering Methodology as the standard approach for all AI-assisted development work, following the "Plan Before Build" principle with a five-step development cycle:

### Core Principles
1. **Plan is Required**: All development starts with an approved plan (initial design document)
2. **Step-by-Step Progress**: Work broken into smallest verifiable "development cycles" - build one part, test, then continue
3. **Human as Final Checker**: Human developer's role is verification, not coding - final quality control
4. **AI as Specialized Tool**: AI given clear, specific instructions and expected to follow them precisely
5. **Stop on Problems**: Any verification failure stops the process until understood and fixed

### Five-Step Development Cycle
1. **Instructions**: Developer gives clear, specific prompt with task, AI role, rules, actions, and completion signal
2. **Building**: AI follows instructions and creates/modifies specified files, then outputs completion signal
3. **Checking**: Developer performs exact verification tasks specified in AI's completion signal
4. **Decision**: Developer judges - "Continue" to next cycle or "Stop and Fix" with detailed problem report
5. **Record**: Successful sequences documented as "Development Cycle" in project history for tracking

### Instruction Requirements
All AI instructions must contain:
- **Task**: Clear work title
- **Role**: AI function definition
- **Rules**: Required guidelines, especially no assumptions
- **Actions**: Precise file operations with exact content
- **Completion Signal**: Specific finish message with verification instructions

## Consequences

### Positive
- Eliminates AI code with unverified assumptions through clear instructions
- Provides thorough checking at each step
- Creates documented "recipes" for development work
- Enables safe AI-human collaboration with quality guarantees
- Supports gradual, verifiable progress

### Negative
- More detailed process with explicit human checking steps
- Slower development pace due to step-by-step cycles
- Requires strict following of methodology structure

### Risks
- Not following the method leading to quality problems
- Too restrictive limits reducing AI usefulness
- Human checking burden if not properly planned

### Related Processes
- Plan Before Build (foundational principle)
- Collaborative Development (complementary guidelines)
- Quality Verification (checking framework)
- Quality Assurance Framework

### Notes
This methodology implements "Check Carefully, Verify, Only Then Trust" as the practical approach for guiding powerful but assumption-prone AI coding. It was developed from the experience of building the Memory System.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\013_anvil_protocol_engineering_methodology.md