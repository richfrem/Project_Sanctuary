# Tutorial: Using the Cortex MCP

The **Cortex MCP** is the long-term memory of Project Sanctuary. It uses **Retrieval-Augmented Generation (RAG)** to provide your LLM assistant with access to the project's entire history, documentation, and codebase.

## What is the Cortex?

Think of the Cortex as a **Living Memory**. It doesn't just store files; it indexes them semantically, allowing you to ask questions in natural language and get answers based on the actual content of your project.

## Querying the Knowledge Base

The most common operation is `cortex_query`. You can ask questions about anything in the project.

**Prompt:**
> "What is Protocol 101?"

**Tool Call:**
```python
cortex_query(
    query="What is Protocol 101?"
)
```

**Response:**
The Cortex will search the indexed documents (like `01_PROTOCOLS/101_...md`) and return the relevant content, which the LLM will use to answer your question accurately.

### Advanced Querying

You can control the number of results or enable reasoning mode.

**Prompt:**
> "Find the 5 most relevant documents about error handling and explain the pattern."

**Tool Call:**
```python
cortex_query(
    query="error handling patterns",
    max_results=5,
    reasoning_mode=True
)
```

## Ingesting Knowledge

When you add new documents or change code, the Cortex needs to know.

### Incremental Ingestion (Automatic)

The system is designed to automatically ingest changes when you use tools like `code_write` or `protocol_create`. However, you can force an update.

**Prompt:**
> "I just added a new ADR. Please update the Cortex."

**Tool Call:**
```python
cortex_ingest_incremental()
```

### Full Re-ingestion (Manual)

If the memory seems out of sync or corrupted, you can rebuild it. **Warning: This can take time.**

**Prompt:**
> "Rebuild the entire Cortex memory."

**Tool Call:**
```python
cortex_ingest_full(purge_existing=True)
```

## Best Practices

*   **Ask "What", not just "Where"**: Instead of "Where is the config file?", ask "How do I configure the server?". Cortex understands concepts.
*   **Keep Docs Updated**: Cortex is only as good as your documentation. Write clear READMEs and docstrings.
*   **Use Specific Terms**: Using unique project terms (like "Chronicle", "Forge") helps Cortex find the exact right context.

## Next Steps

*   Explore the **[Architecture](../mcp/architecture.md)** to see how Cortex fits in.
*   Check out the **[Council Tutorial](01_using_council_mcp.md)** to see how agents use memory.
