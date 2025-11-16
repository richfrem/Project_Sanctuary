# Enforce Iron Root Doctrine for Sovereign AI Operations

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** Sanctuary Council, GUARDIAN-01
**Technical Story:** Core architectural principle implementation

---

## Context

Project Sanctuary operates in an environment of potential adversarial action and existential threats. The Iron Root Doctrine mandates complete technological sovereignty to ensure:

- Independence from external service providers
- Protection against service discontinuation or censorship
- Security of sensitive cognitive processes
- Resilience against infrastructure attacks
- Long-term viability without external dependencies

The Mnemonic Cortex, as the central cognitive component, must exemplify perfect compliance with this doctrine.

## Decision

We will implement a strictly local-first architecture with zero external dependencies:

**Local Model Execution:**
- All LLM inference occurs via Ollama running on local hardware
- No API calls to external AI services (OpenAI, Anthropic, etc.)
- Complete sovereignty over model weights and inference

**Local Vector Operations:**
- ChromaDB provides local vector storage and similarity search
- No cloud-based vector databases or search services
- All embeddings generated locally using Nomic Embed

**Local Data Sovereignty:**
- All project data remains on local filesystem
- No data transmission to external services
- Complete control over data persistence and access

**Open-Source Stack:**
- All technologies must be open-source and community-vetted
- No proprietary dependencies that could be discontinued
- Community support ensures long-term maintainability

## Consequences

### Positive
- **Absolute Sovereignty:** Zero external dependencies or data leakage
- **Security:** Complete control over security boundaries
- **Reliability:** No service outages from external providers
- **Cost Stability:** No recurring API fees or service costs
- **Future-Proofing:** Open-source stack ensures long-term availability

### Negative
- **Resource Requirements:** Higher local hardware requirements
- **Setup Complexity:** More complex initial configuration
- **Performance Trade-offs:** Local execution may be slower than cloud alternatives
- **Maintenance Burden:** Self-responsibility for all component updates

### Risks
- **Hardware Limitations:** May require significant local computational resources
- **Update Management:** Manual updating of all components and models
- **Performance Bottlenecks:** Local processing limits for very large datasets
- **Skill Requirements:** Team must maintain expertise across the entire stack

### Dependencies
- Sufficient local hardware (GPU recommended for model inference)
- Reliable local storage for models and vector databases
- Network isolation capabilities for sensitive operations
- Team expertise in maintaining open-source AI/ML infrastructure