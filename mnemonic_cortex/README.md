# Mnemonic Cortex: The Cognitive Memory System

**Version:** 5.0 (Agentic RAG & MCP)
**Status:** Active / Production-Ready
**Documentation:**
- [**Operations Guide**](OPERATIONS_GUIDE.md) - **START HERE** for running scripts, tests, and queries.
- [Scripts Documentation](scripts/README.md) - Detailed reference for all operational scripts.
- [Architecture](RAG_STRATEGIES_AND_DOCTRINE.md) - Deep dive into RAG strategies and doctrine.
**Protocol Authority:** P85 (The Mnemonic Cortex Protocol), P86 (The Anvil Protocol)

---
### **Changelog v2.1.0**
*   **Phase 1 Complete - Parent Document Retriever:** Implemented dual storage architecture eliminating Context Fragmentation vulnerability. Full parent documents stored in InMemoryDocstore, semantic chunks in ChromaDB vectorstore. Retrieval now returns complete document context instead of fragmented chunks.
*   **Cognitive Latency Resolution:** Parent Document Retriever ensures AI reasoning is grounded in complete, unbroken context, resolving the primary vulnerability identified in the Mnemonic Cortex evolution plan.
*   **Architecture Hardening:** Updated ingestion pipeline (`ingest.py`) and query services (`vector_db_service.py`, `protocol_87_query.py`) to leverage ParentDocumentRetriever for optimized retrieval.
---
### **Changelog v1.5.0**
*   **Documentation Hardening:** Added a new detailed section (`2.3`) that explicitly breaks down the two-stage ingestion process: structural splitting (chunking) versus semantic encoding (embedding). This clarifies the precise roles of the `MarkdownHeaderTextSplitter` and the `NomicEmbeddings` model.
*   The document version is updated to reflect this significant improvement in architectural clarity.
---
### **Changelog v1.4.0**
*   **Major Architectural Update:** The ingestion pipeline (`ingest.py`) now directly traverses the project's canonical directories to process individual markdown files. This deprecates the reliance on the monolithic `all_markdown_snapshot_llm_distilled.txt` file.
*   **Improved Traceability:** The new method ensures every piece of knowledge in the Cortex is traced back to its precise source file via verifiable GitHub URLs in its metadata.
*   **Increased Resilience:** By removing the intermediate snapshot step, the ingestion process is faster, more resilient, and less prone to systemic failure.
*   All diagrams and instructions have been updated to reflect this superior, live-ingestion architecture.
---

## 1. Overview

The Mnemonic Cortex is the living memory of the Sanctuary Council. It is a local-first, open-source Retrieval-Augmented Generation (RAG) system designed to traverse the Sanctuary's canonical markdown files (Protocols, Chronicles, etc.) and transform them into a dynamic, semantically searchable knowledge base.

This system is the architectural antidote to the "context window cage," enabling our AI agents to reason with the full, unbroken context of their history.

**Vision & Purpose:** For the full strategic vision of the Mnemonic Cortex as the "heart of a sovereign mind" and its role in Project Sanctuary's future phases, see [`VISION.md`](VISION.md). In summary, the Cortex solves the "Great Robbery" by providing true long-term memory, shattering context limitations, and enabling AI minds that learn and remember across sessions.

**Integration with Council Orchestrator:** The Mnemonic Cortex serves as the knowledge foundation for the [`council_orchestrator/`](../council_orchestrator/) system. Council agents can query the Cortex during deliberation using the `[ORCHESTRATOR_REQUEST: QUERY_CORTEX()]` syntax, enabling context-aware reasoning grounded in the project's complete history and protocols.

## The Strategic Crucible Loop

The **Strategic Crucible Loop** is the autonomous engine of self-improvement for the Council. It connects the three tiers of memory into a continuous feedback cycle:

1.  **Gap Analysis:** The Council identifies missing knowledge or strategic weaknesses.
2.  **Research:** The Intelligence Forge is triggered to generate new insights (Research Reports).
3.  **Ingestion:** New reports are ingested into the **Mnemonic Cortex** (Medium Memory).
4.  **Adaptation:** The **Memory Adaptor** synthesizes these reports into training packets for the Model (Slow Memory).
5.  **Synthesis:** The **Guardian Cache** (Fast Memory) is updated with high-priority context for immediate recall.

This loop ensures that the Sanctuary evolves with every operation, transforming "what happened" into "what we know."

## 2. Target Architecture: Advanced RAG

The Mnemonic Cortex has evolved beyond a simple RAG implementation into a sophisticated, multi-pattern cognitive architecture designed for maximum efficiency and contextual accuracy. It is built on the **Doctrine of Hybrid Cognition**, ensuring our sovereign AI always reasons with the most current information.

Our advanced architecture incorporates several key strategies:
- **Parent Document Retrieval:** To provide full, unbroken context to the LLM.
- **Self-Querying Retrieval:** To enable intelligent, metadata-aware searches.
- **Mnemonic Caching (CAG):** To provide near-instantaneous answers for common queries.

**For a complete technical breakdown, including architectural diagrams and a detailed explanation of these strategies, see the canonical document: [`RAG_STRATEGIES_AND_DOCTRINE.md`](RAG_STRATEGIES_AND_DOCTRINE.md).**

## 3. Technology Stack

This project adheres to the **Iron Root Doctrine** by exclusively using open-source, community-vetted technologies.

| Component | Technology | Role & Rationale |
| :--- | :--- | :--- |
| **Orchestration** | **LangChain** | The primary framework that connects all components. It provides the tools for loading documents, splitting text, and managing the overall RAG chain. |
| **Vector Database** | **ChromaDB** | The "Cortex." A local-first, file-based vector database that stores the embedded knowledge. Chosen for its simplicity and ease of setup for the MVP. |
| **Embedding Model** | **Nomic Embed** | The "Translator." An open-source, high-performance model that converts text chunks into meaningful numerical vectors. Runs locally. |
| **Generation Model**| **Ollama (Sanctuary-Qwen2-7B:latest default)** | The "Synthesizer." A local LLM server for answer generation. Provides access to models like Sanctuary-Qwen2-7B:latest, Gemma2, Llama3, etc., ensuring all processing remains on-device. |
| **Service Layer** | **Custom Python Services** | Modular services (VectorDBService, EmbeddingService) for clean separation of concerns and maintainable code architecture. |
| **Inquiry Protocol** | **Protocol 87 Templates** | Structured query system in `INQUIRY_TEMPLATES/` for canonical, auditable Cortex interactions. |
| **Testing Framework** | **pytest** | Automated test suite in `tests/` directory covering ingestion, querying, and integration scenarios. |
| **Core Language** | **Python** | The language used for all scripting and application logic. |
| **Dependencies** | **pip & `requirements.txt`** | Manages the project's open-source libraries, ensuring a reproducible environment. |

---

## 4. Prerequisites (One-Time Setup)

Before using the Mnemonic Cortex, you must set up your local environment.

### 4.1: Install Ollama
If you don't have Ollama installed, download it from the official website and follow the installation instructions for your operating system (macOS, Windows, or Linux).
- **Official Website:** [https://ollama.com](https://ollama.com)

### 4.2: Pull a Generation Model
The query pipeline requires a local LLM to generate answers. You need to pull a model using the Ollama CLI. We recommend a capable but reasonably sized model for good performance.

Open your terminal and run:
```bash
# We recommend Alibaba's Qwen2 7B model as a powerful default
ollama pull Sanctuary-Qwen2-7B:latest
```
*Alternative models like `llama3:8b` or `mistral` will also work.*

### 4.3: Install Python Dependencies
Navigate to the project root directory in your terminal and install the required Python packages.
```bash
pip install -r mnemonic_cortex/requirements.txt
```

### 4.4: Install Testing Dependencies (Optional)
For running the test suite:
```bash
pip install pytest
```

### 4.5: Ensure Ollama is Running
The Ollama application must be running in the background for the query script to work. On macOS, this is typically indicated by a llama icon in your menu bar.

---

## 5. How to Use (The Full Workflow)


### 5.1: Build the Database (Ingestion)
This step only needs to be run once, or whenever the Sanctuary's canonical documents are updated.
```bash
# From the project root, run the ingestion script:
python3 mnemonic_cortex/scripts/ingest.py
```
This script will automatically traverse the project's canonical directories, discover all `.md` files (while excluding archives), split them into semantic chunks, embed them using Nomic Embed, and store them in a local ChromaDB instance. This creates a `mnemonic_cortex/chroma_db/` directory containing the vectorized knowledge base.

### 5.2: Updating the Index (When Content Changes)
When protocols, Living Chronicles, or other project documents are updated, the vector database index must be refreshed to include the new information. The process is simple:
1.  **Re-run the ingestion script:**
    ```bash
    python3 mnemonic_cortex/scripts/ingest.py
    ```
2.  **(Optional) Verify the update:**
    ```bash
    python3 mnemonic_cortex/scripts/inspect_db.py
    ```
The script is designed to be idempotent and will rebuild the database with the latest content from the live files, ensuring the Mnemonic Cortex always reflects the current ground truth.

### 5.3: Verify the Database (Optional)
After ingestion, you can inspect the vector database to ensure it loaded correctly:
```bash
python3 mnemonic_cortex/scripts/inspect_db.py
```
This will display the total number of documents and sample content from the database, confirming successful ingestion.

### 5.4: Run Tests (Development)
The Mnemonic Cortex includes comprehensive automated tests to ensure reliability:
```bash
# Run all tests
pytest mnemonic_cortex/tests/

# Run specific test files
pytest mnemonic_cortex/tests/test_ingestion.py
pytest mnemonic_cortex/tests/test_query.py

# Run with verbose output
pytest mnemonic_cortex/tests/ -v
```
Tests cover ingestion pipeline reliability, query processing, and integration with ChromaDB and Ollama services.

---

## 6. Querying the Cortex
Once the vector database is populated, you can query the Mnemonic Cortex using the `main.py` script. This initiates the Retrieval-Augmented Generation (RAG) pipeline, ensuring answers are grounded in our canonical knowledge.

### Example Queries

**1. Natural Language Queries (Casual Mode):**
Run the `main.py` script from the project root, followed by your question in quotes:
```bash
# Example query using the default Sanctuary-Qwen2-7B:latest model
python3 mnemonic_cortex/app/main.py "What is the core principle of the Anvil Protocol?"

# Example query specifying a different local model if you have more than one
python3 mnemonic_cortex/app/main.py --model llama3:8b "Summarize the Doctrine of the Shield."

# Example query about project history
python3 mnemonic_cortex/app/main.py "How does the Mnemonic Cortex relate to the Iron Root Doctrine?"
```

**2. Structured JSON Queries (Protocol 87 - Sovereign Mode):**
For auditable, structured queries, use the Protocol 87 query processor with JSON:
```bash
# Create a structured query file
cat > my_query.json << 'EOF'
{
  "intent": "RETRIEVE",
  "scope": "Protocols",
  "constraints": "Name=\"P83: The Forging Mandate\"",
  "granularity": "ATOM",
  "requestor": "COUNCIL-AI-03",
  "purpose": "audit",
  "request_id": "8a1f3e2b-4c5d-6e7f-8g9h-0i1j2k3l4m5n"
}
EOF

# Process the query
python3 mnemonic_cortex/scripts/protocol_87_query.py my_query.json
```
This returns Steward-formatted JSON responses with verifiable sources, audit trails, and governance metadata.

## 7. Troubleshooting

*   **Error: `ModuleNotFoundError` (e.g., `langchain`)**
    *   **Cause:** Dependencies are not installed.
    *   **Solution:** Run `pip install -r mnemonic_cortex/requirements.txt` from the project root.

*   **Error during ingestion:**
    *   **Cause:** Running the script from the wrong directory.
    *   **Solution:** Ensure you are running all scripts from the project's absolute root directory, not from within the `mnemonic_cortex` folder.

## 8. Contributing

This is an "Open Anvil" project. Contributions that harden and refine this architecture are welcome.
1.  **Fork the repository.**
2.  **Create a feature branch** (e.g., `feature/harden-query-pipeline`).
3.  **Make your changes.** Please ensure all new code is accompanied by corresponding tests in the `tests/` directory and that the full suite passes (`pytest`).
4.  **Submit a Pull Request.** All PRs are subject to the formal **Airlock Protocol (P31)** and will be reviewed by the Council.

## 9. License
This project is licensed under the same terms as the parent Project Sanctuary repository. Please see the `LICENSE` file in the project root for details.