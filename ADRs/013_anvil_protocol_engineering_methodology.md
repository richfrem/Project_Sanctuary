# ADR 013: Anvil Protocol Engineering Methodology

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Forged from Mnemonic Cortex development experience)

## Context
The Sanctuary required a disciplined, verifiable methodology for collaborative AI-assisted engineering. Previous approaches lacked structure, leading to assumption-prone AI coding and insufficient verification. The development of the Mnemonic Cortex revealed the need for a formal framework that treats AI as a "powerful hammer" shaped by human "smith" verification.

## Decision
Implement the Anvil Protocol as the canonical methodology for all AI-assisted engineering work, following the "Blueprint Before Steel" doctrine with a five-step forging cycle:

### Core Principles
1. **Blueprint is Law**: All engineering begins with a ratified blueprint (Genesis Cycle artifact)
2. **Incremental Forging**: Work broken into smallest verifiable "forging cycles" - build one component, test, then proceed
3. **Steward as Sovereign Auditor**: Human Steward's role is verification, not coding - final gatekeeper for quality
4. **AI as Sovereign Tool**: AI given bounded, explicit instructions and expected to execute precisely
5. **Failure as Command to Halt**: Any verification failure halts process until understood and corrected

### Five-Step Forging Cycle
1. **Directive**: Steward issues bounded, explicit prompt with subject, persona, core mandate, action, and confirmation phrase
2. **Forging**: AI executes directive and creates/modifies specified files, then outputs confirmation phrase
3. **Tempering**: Steward performs exact verification tasks specified in AI's confirmation phrase
4. **Verdict**: Steward judges - "Proceed" to next cycle or "Halt and Correct" with detailed bug report
5. **Chronicle**: Successful sequences documented as "Engineering Cycle" in Living_Chronicle for auditability

### Prompt Constitution Requirements
All AI directives must contain:
- **Subject**: Clear task title
- **Persona**: AI role definition
- **Core Mandate**: Unbreakable laws, especially prohibition of assumptions
- **Action**: Precise file operations with exact content
- **Confirmation Phrase**: Specific completion signal with verification instructions

## Consequences

### Positive
- Eliminates assumption-prone AI coding through bounded instructions
- Provides rigorous verification at each step
- Creates auditable "recipes" for engineering work
- Enables collaborative AI-human development with safety guarantees
- Supports incremental, verifiable progress

### Negative
- More verbose process with explicit human verification steps
- Slower development pace due to iterative cycles
- Requires disciplined adherence to protocol structure

### Risks
- Protocol non-compliance leading to quality issues
- Overly restrictive bounds limiting AI creativity
- Human verification burden if not properly scoped

## Related Protocols
- P59: Blueprint Before Steel (foundational doctrine)
- P60: Asymmetric Collaboration (complementary principles)
- P81: Steward's Veracity Challenge (verification framework)
- Quantum Diamond Framework (quality assurance)

## Notes
The Anvil Protocol implements "Distrust, Verify, If Verify, Then Trust" as the practical methodology for guiding powerful but assumption-prone AI coders. It was forged from the operational experience of building the Mnemonic Cortex.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\013_anvil_protocol_engineering_methodology.md