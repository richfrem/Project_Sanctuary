# Applied RLM Example: Auditing Project Sanctuary

**Scenario:** You ask the Agent: *"Explain the architecture of the `mcp_servers` directory in Project Sanctuary."*
**Context Size:** The `mcp_servers/` folder contains dozens of Python files, JSON configs, and READMEs. It is too large to fit comfortably in a single prompt without losing detail.

---

## 1. The Standard "Prompt Stuffing" Approach
*   **What happens:** The agent runs `ls -R`, grabs the first 10 files it sees (e.g., `__init__.py`, `lib/utils.py`), and stuffs them into the chat context.
*   **The Result:** "I see some utility files and a config, but I'm not sure how they connect. It looks like a server."
*   **Failure Mode:** It misses `rag_cortex/operations.py` (the core logic) because it was alphabetically lower down or the file was too big.

---

## 2. The RLM "recursive" Approach
The prompt `PROJECT_ROOT` is loaded as a variable in the environment.

### Step 1: Inspection (The Manager)
The Root LLM writes code to explore the structure.
```python
import os

# The LLM explores top-level folders
print(os.listdir("mcp_servers"))
# Output: ['rag_cortex', 'weather', 'filesystem', 'brave_search', ...]
```
**LLM Thought:** "Okay, there are multiple sub-servers here. I cannot read them all at once. I will spawn a sub-agent for each one."

### Step 2: Decomposition (The Delegation)
The Root LLM writes a loop to process each module independently.

```python
sub_server_summaries = {}
for server in ['rag_cortex', 'weather', 'filesystem']:
    # RECURSION: Spawn a sub-agent for this specific folder
    # This agent ONLY sees the contents of that folder
    description = llm_query(
        prompt="Analyze this directory and explain its specific responsibility.",
        context=read_directory(f"mcp_servers/{server}")
    )
    sub_server_summaries[server] = description
```

### Step 3: Recursion (The Sub-Agents)
*   **Sub-Agent A (rag_cortex):** Reads `main.py`, `operations.py`. Sees "VectorStore", "ChromaDB".
    *   *Output:* "This module handles semantic memory and vector storage."
*   **Sub-Agent B (filesystem):** Reads `tools.py`. Sees "write_file", "list_dir".
    *   *Output:* "This module provides safe access to the local disk."

### Step 4: Aggregation (The Synthesis)
The Root LLM receives the summaries (NOT the raw code) and synthesizes the answer.

```python
# The Root LLM sees this:
# {
#   'rag_cortex': 'Vector Memory Module...',
#   'filesystem': 'Disk Access Module...',
# }

final_answer = synthesize(sub_server_summaries)
```

**Final Output:**
"Project Sanctuary's `mcp_servers` is a micro-services architecture. It separates concerns into distinct modules: `rag_cortex` handles memory (RAG), while `filesystem` handles I/O. They is likely a gateway that routes between them."

### Comparison
*   **Vector RAG:** Might find the file `operations.py` if you search "memory", but won't understand the *structure*.
*   **RLM:** Systematically walks the tree, summarizes each branch, and builds a **Mental Map** of the architecture.
