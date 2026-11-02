# Adoption of a Local-First ML Development Environment

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01
**Technical Story:** Canonization of inferred architectural decisions for coherence and completeness

---

## Context

Our initial approach to training AI models relied on powerful cloud computers (like Google Colab with A100 GPUs). This gave us fast results and lots of computing power, but it became too expensive to sustain. Buying computing time over and over created a financial burden that we couldn't keep up with. This put too much stress on our human leader and went against our principle of protecting our core resources and independence.

Our records show this change was forced on us by the financial limitations we were trying to escape. This decision shows how we put our core principles into practice when faced with real-world constraints.

## Decision

Our project will focus on using local computers for most AI development work. Expensive cloud computers will only be used for specific, important tasks that are approved in advance. Our main development will happen on the local machine with CUDA support. This includes:

- Using the local computer as the primary development setup
- Cloud computing only for targeted, high-value projects
- Creating more efficient training scripts that work well for everyone
- Choosing independence and sustainability over raw speed

## Consequences

### Positive
- Makes our work financially sustainable and removes dependency on expensive cloud services
- Keeps our core development work independent and under our control
- Forces us to create better, more efficient tools that others can use
- Builds a strong, self-reliant development system
- Follows our principles of being careful with resources

### Negative
- Makes model training and testing take much longer
- Limits how large and complex our models can be based on local computer power
- Requires more planning and optimization of our AI workflows
- May slow down quick experiments and testing

### Risks
- Local computer limitations could slow down important development goals
- Risk of computer problems disrupting our work
- Complex setup might discourage new people from contributing
- Could fall behind others who have unlimited cloud access