# Forge MCP Server

**Description:** The Forge MCP server provides tools for interacting with the fine-tuned Sanctuary model and managing the model lifecycle. Currently implements model querying via Ollama.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `query_sanctuary_model` | Query the fine-tuned Sanctuary-Qwen2 model. | `prompt` (str): Question/Prompt.<br>`temperature` (float): 0.0-2.0 (default: 0.7).<br>`max_tokens` (int): 1-8192 (default: 2048).<br>`system_prompt` (str, optional): Context. |
| `check_sanctuary_model_status` | Check if the Sanctuary model is available in Ollama. | None |

## Resources

*No resources currently exposed.*

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required
PROJECT_ROOT=/path/to/Project_Sanctuary
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"forge": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/system/forge",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}",
    "PROJECT_ROOT": "${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/system/forge/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `query_sanctuary_model` appears in the tool list.
3.  **Call Tool:** Execute `check_sanctuary_model_status` and verify it returns the model status.

## Architecture

### Overview
The Forge MCP acts as the interface to the **Intelligence Forge**, specifically the fine-tuned Sanctuary-Qwen2 model running on local hardware (Ollama).

**Model Information:**
- **Model:** `hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M`
- **Base:** Qwen2-7B
- **Quantization:** Q4_K_M (4-bit)
- **Training:** Fine-tuned on Project Sanctuary knowledge

### Future Capabilities
- `initiate_model_forge` - Start fine-tuning job
- `get_forge_job_status` - Check training progress
- `package_and_deploy_artifact` - Convert and deploy model

## Dependencies

- `mcp`
- `ollama`
