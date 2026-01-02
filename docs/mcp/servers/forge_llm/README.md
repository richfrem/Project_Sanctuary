# Forge LLM MCP Server Documentation

## Overview

Forge LLM MCP provides access to the fine-tuned Sanctuary-Qwen2 model for specialized knowledge and decision-making. Currently limited to query operations only.

## Key Concepts

- **Fine-Tuned Model:** Sanctuary-Qwen2-7B trained on Project Sanctuary data
- **Specialized Knowledge:** Protocol-aware, strategic insights
- **Query Only:** Fine-tuning operations explicitly out of scope
- **Hardware Requirements:** CUDA GPU required for fine-tuning (not for queries)

## Server Implementation

- **Server Code:** [mcp_servers/forge_llm/server.py](../../../mcp_servers/forge_llm/server.py)
- **Operations:** [mcp_servers/forge_llm/operations.py](../../../mcp_servers/forge_llm/operations.py)

## Testing

- **Test Suite:** [tests/mcp_servers/forge_llm/](../../../tests/mcp_servers/forge_llm/)
- **Status:** ⚠️ Requires CUDA GPU for testing
- **Environment:** `CUDA_FORGE_ACTIVE=true` required

## Operations

### `query_sanctuary_model`
Query the fine-tuned Sanctuary-Qwen2 model

**Example:**
```python
query_sanctuary_model(
    prompt="What is the strategic priority for Q1 2025?",
    temperature=0.7,
    max_tokens=2048,
    system_prompt="You are a Sanctuary protocol expert"
)
```

### `check_sanctuary_model_status`
Verify model availability in Ollama

## Scope Limitation

> [!WARNING]
> Currently only `query_sanctuary_model` and `check_sanctuary_model_status` are authorized for MCP usage. Automated fine-tuning operations are explicitly **out of scope** until further trust verification.

## Hardware Requirements

- **Query Operations:** Standard hardware (CPU/GPU)
- **Fine-Tuning Operations:** CUDA GPU required (not exposed via MCP)

## Status

✅ **Query Operations Operational** - Fine-tuning out of scope
