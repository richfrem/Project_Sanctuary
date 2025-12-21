# Forge LLM Testing Strategy

This directory contains the test suite for the Forge LLM MCP, organized into a 3-layer pyramid as mandates by ADR 048.

## Structure

- **`unit/`**: Fast, mocked tests. Run these frequently. No external dependencies.
- **`integration/`**: Real tests against the Ollama container. Accesses the actual `sanctuary_ollama_mcp` service.

## Prerequisites for Integration Tests

1.  **Start Ollama:**
    ```bash
    podman compose up -d ollama-model-mcp
    ```

2.  **Verify Model:**
    Ensure `hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M` is pulled.

## Running Tests

### Layer 1: Unit Tests (Fast)
```bash
pytest tests/mcp_servers/forge_llm/unit/ -v
```

### Layer 2: Integration Tests (Real)
```bash
pytest tests/mcp_servers/forge_llm/integration/ -v
```
*(Will verify connection and query capability if model is present)*
