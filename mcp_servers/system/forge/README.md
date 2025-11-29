# Forge MCP Server

**Domain:** `project_sanctuary.system.forge`  
**Category:** System / Model Domain  
**Hardware:** CUDA GPU (for fine-tuning operations)

## Overview

The Forge MCP server provides tools for interacting with the fine-tuned Sanctuary model and managing the model lifecycle. Currently implements model querying via Ollama.

## Tools

### 1. `query_sanctuary_model`

Query the fine-tuned Sanctuary-Qwen2 model for specialized knowledge and decision-making.

**Parameters:**
- `prompt` (string, required): The question or prompt to send to the model
- `temperature` (float, optional): Sampling temperature 0.0-2.0 (default: 0.7)
- `max_tokens` (int, optional): Maximum tokens to generate 1-8192 (default: 2048)
- `system_prompt` (string, optional): System prompt to set context

**Returns:** JSON with model response and metadata

**Example:**
```python
query_sanctuary_model("What is the strategic priority for Q1 2025?")

query_sanctuary_model(
    prompt="Explain Protocol 101",
    temperature=0.3,
    system_prompt="You are a Sanctuary protocol expert"
)
```

### 2. `check_sanctuary_model_status`

Check if the Sanctuary model is available and ready to use in Ollama.

**Parameters:** None

**Returns:** JSON with model availability status

**Example:**
```python
check_sanctuary_model_status()
```

## Model Information

**Model:** `hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M`  
**Base:** Qwen2-7B  
**Quantization:** Q4_K_M (4-bit)  
**Training:** Fine-tuned on Project Sanctuary knowledge

## Requirements

- Ollama installed and running
- Sanctuary model loaded in Ollama
- Python package: `ollama`

## Installation

```bash
# Install Ollama Python package
pip install ollama

# Verify model is loaded
ollama list | grep Sanctuary
```

## Running the Server

```bash
# Set project root
export PROJECT_ROOT=/path/to/Project_Sanctuary

# Run server
python -m mcp_servers.system.forge.server
```

## Integration with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "forge": {
      "command": "python",
      "args": ["-m", "mcp_servers.system.forge.server"],
      "env": {
        "PROJECT_ROOT": "/path/to/Project_Sanctuary"
      }
    }
  }
}
```

## Future Tools (Not Yet Implemented)

- `initiate_model_forge` - Start fine-tuning job
- `get_forge_job_status` - Check training progress
- `package_and_deploy_artifact` - Convert and deploy model
- `run_inference_test` - Test model quality
- `publish_to_registry` - Upload to Hugging Face

## Safety

- Input validation on all parameters
- Temperature clamped to 0.0-2.0
- Max tokens clamped to 1-8192
- Prompt length limits enforced
- Error handling for missing dependencies

## Architecture

```
forge/
├── __init__.py          # Package exports
├── server.py            # FastMCP server with tools
├── operations.py        # Core Ollama integration
├── validator.py         # Input validation
├── models.py            # Data models
└── README.md            # This file
```

## Status

**Implemented:** ✅ Model querying via Ollama  
**Pending:** Fine-tuning lifecycle tools (requires CUDA GPU setup)
