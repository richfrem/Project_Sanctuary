# ADR 003: Choice of Ollama for Local LLM Inference

- **Status:** Accepted
- **Date:** 2024-05-20
- **Architects:** Sanctuary Council

## Context

The local-first RAG architecture (ADR 001) requires a local Large Language Model (LLM) for the final "generation" step. The system needed a simple, robust, and sovereign way to run various open-source models on the Steward's local machine (macOS) without complex configuration or cloud dependencies.

## Decision

We will use **Ollama** as the exclusive engine for local LLM inference. All interactions with local models from our Python scripts will be managed through the `langchain-ollama` integration.

## Consequences

- **Positive:**
    -   **Simplicity & Hearth Protocol:** Ollama provides a single, unified command (`ollama pull <model>`) and a running server to manage multiple local models. This is vastly simpler than managing individual model weights and configurations, perfectly aligning with the `Hearth Protocol`.
    -   **Sovereignty & Iron Root:** Ollama is open-source and runs entirely on-device, ensuring our generation step remains 100% sovereign and free from external dependencies, fulfilling the `Iron Root Doctrine`.
    -   **Flexibility:** It allows us to easily switch between different open-source models (e.g., `qwen2:7b`, `llama3:8b`) with a simple command-line argument, enabling rapid testing and validation.

- **Negative:**
    -   **Resource Management:** Ollama runs as a background application, consuming system resources (RAM). This is an accepted trade-off for its ease of use.
    -   **Dependency:** Our application now has a runtime dependency on the Ollama application being active in the background. Our scripts must include clear error handling for when it is not running.