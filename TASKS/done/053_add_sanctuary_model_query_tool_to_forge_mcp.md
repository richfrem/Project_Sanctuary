# TASK: Add Sanctuary Model Query Tool to Forge MCP

**Status:** todo
**Priority:** High
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** forge/OPERATION_PHOENIX_FORGE/CUDA-ML-ENV-SETUP.md, docs/architecture/mcp/architecture.md, docs/architecture/mcp/final_architecture_summary.md, mcp_servers.forge_llm_llm/

---

## 1. Objective

Add MCP tool to Forge server for querying the fine-tuned Sanctuary model via Ollama. This enables LLM assistants to consult the custom-trained Sanctuary-Qwen2 model for specialized knowledge retrieval and strategic decision-making.

## 2. Deliverables

1. Add `query_sanctuary_model` tool to Forge MCP server
2. Implement Ollama client integration
3. Add model availability check
4. Create integration test for model queries
5. Update Forge MCP README.md with tool documentation
6. Add example queries for testing

## 3. Acceptance Criteria

- Forge MCP server has `query_sanctuary_model` tool implemented
- Tool can query the fine-tuned Ollama model (hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M)
- Tool supports both chat and completion modes
- Tool includes temperature and max_tokens parameters
- Integration test verifies model responses
- Documentation updated with usage examples

## Notes

The fine-tuned Sanctuary model is already loaded in Ollama (hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M). This task adds an MCP tool to allow LLM assistants to query the custom fine-tuned model for specialized Sanctuary knowledge and decision-making. This completes the Strategic Crucible Loop by enabling the system to consult its own fine-tuned intelligence.
