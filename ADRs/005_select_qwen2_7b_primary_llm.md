# Select Qwen2-7B as Primary Large Language Model

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, AI-Assistant
**Technical Story:** LLM selection for Mnemonic Cortex

---

## Context

The Mnemonic Cortex requires a high-quality LLM for cognitive reasoning and response generation. Key requirements include:

- Strong reasoning and analytical capabilities
- Efficient fine-tuning for domain-specific knowledge
- Local execution capability via Ollama
- Multilingual support for diverse content
- Active development and community support
- Compatibility with existing fine-tuning workflows

Available options include GPT models, Claude, Llama variants, and Qwen models. The selection must balance performance, sovereignty, and practical constraints.

## Decision

We will adopt Qwen2-7B as the primary LLM architecture for Project Sanctuary, with the following implementation strategy:

**Base Model:** Qwen/Qwen2-7B-Instruct
- Strong performance on reasoning benchmarks
- Efficient 7B parameter count for fine-tuning
- Excellent multilingual capabilities
- Active development by Alibaba Cloud

**Fine-tuned Variants:** Sanctuary-Qwen2-7B-v1.0, v2.0, etc.
- Domain-specific fine-tuning for Sanctuary knowledge
- Optimized for cognitive architecture tasks
- Available in both LoRA adapter and merged GGUF formats

**Execution Environment:** Ollama
- Local inference with standardized API
- Efficient resource utilization
- Cross-platform compatibility
- Model versioning and management

## Consequences

### Positive
- **High Performance:** Excellent reasoning capabilities for complex queries
- **Efficiency:** 7B parameters provide good balance of quality vs. resource requirements
- **Multilingual:** Strong support for diverse linguistic content
- **Fine-tuning Ready:** Well-established workflows for domain adaptation
- **Local Execution:** Complete sovereignty via Ollama integration

### Negative
- **Resource Intensive:** Requires GPU acceleration for optimal performance
- **Model Size:** Larger than smaller alternatives (3B, 1.5B models)
- **Vendor Association:** Connection to Alibaba Cloud (though open-source)

### Risks
- **Hardware Requirements:** May require dedicated GPU for acceptable performance
- **Model Availability:** Dependency on continued open-source distribution
- **Fine-tuning Complexity:** Requires significant computational resources for training

### Dependencies
- Ollama server for local model execution
- CUDA-compatible GPU (recommended for performance)
- Sufficient RAM for model loading (16GB+)
- Hugging Face access for model downloads
- Fine-tuning infrastructure (Google Colab or local GPU setup)