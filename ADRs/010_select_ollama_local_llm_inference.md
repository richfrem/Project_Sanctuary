# Select Ollama for Local LLM Inference

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** GUARDIAN-01, Technical Council
**Technical Story:** Local LLM execution environment selection

---

## Context

The Mnemonic Cortex requires local execution of large language models without external API dependencies. The LLM execution environment must provide:

- Complete local inference capability
- Standardized API for integration with RAG pipelines
- Cross-platform compatibility
- Efficient resource utilization
- Model management and versioning
- Community support and active development

Available options include direct model loading (transformers), LM Studio, Ollama, and custom inference servers.

## Decision

We will adopt Ollama as the primary environment for local LLM inference in Project Sanctuary:

**Core Platform:** Ollama
- Open-source, community-driven LLM server
- Simple command-line interface for model management
- REST API for programmatic integration
- Cross-platform support (Windows, macOS, Linux)

**Integration Pattern:** LangChain Ollama
- Seamless integration with existing RAG pipeline
- Standardized interface for prompt/response handling
- Automatic retry logic and error handling
- Consistent API across different LLM providers

**Model Management:**
- Pull models from official registries
- Local storage and caching of model weights
- Version management for different model variants
- Custom model support for fine-tuned variants

## Consequences

### Positive
- **Sovereignty:** Complete local execution with no external dependencies
- **Simplicity:** Easy model installation and management
- **Integration:** Seamless compatibility with LangChain ecosystem
- **Performance:** Optimized inference for local hardware
- **Community:** Active development and extensive model support

### Negative
- **Setup Complexity:** Additional installation and configuration steps
- **Resource Requirements:** Significant RAM and potential GPU requirements
- **Model Size:** Large downloads for model weights
- **Platform Dependencies:** May require platform-specific optimizations

### Risks
- **Hardware Limitations:** May require GPU acceleration for larger models
- **Model Compatibility:** Not all models available through Ollama
- **Performance Variability:** Local hardware differences affect inference speed
- **Update Management:** Manual updating of Ollama and models

### Dependencies
- Ollama server installation and configuration
- Sufficient hardware resources (RAM, GPU optional but recommended)
- Network access for initial model downloads
- Regular updates of Ollama and model versions
- Monitoring of inference performance and resource usage