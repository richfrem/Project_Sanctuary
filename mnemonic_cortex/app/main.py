"""
Mnemonic Cortex Query Application (app/main.py) v1.3 - Verifiable Source Retrieval

This module implements the complete Query Pipeline in the Mnemonic Cortex RAG system.
It orchestrates the full RAG chain: user query -> embedding -> similarity search -> context retrieval -> LLM generation -> answer with verifiable sources.

Role in RAG Pipeline:
- Parses command-line arguments for query input and model selection.
- Initializes all services (VectorDB, Embedding) and loads the persisted ChromaDB.
- Constructs the RAG chain using LangChain LCEL: retriever + prompt + Ollama LLM + output parser.
- Executes the chain to provide context-aware answers grounded in the Cognitive Genome.
- Retrieves and displays verifiable GitHub source URLs for every piece of knowledge used.

Key Improvements in v1.3:
- Verifiable sources: Displays GitHub URLs for all retrieved chunks.
- Enhanced RAG chain: Modified to pass source documents through for citation.
- Superior traceability: Every answer includes links to canonical sources.

Dependencies:
- VectorDBService: Loads ChromaDB and provides retriever for similarity searches.
- EmbeddingService: Used implicitly by ChromaDB for query vectorization.
- Ollama: Local LLM server must be running with the specified model (default: qwen2:7b).
- LangChain: Provides the RAG chain orchestration, prompts, and output parsing.
- Core utilities: find_project_root() and setup_environment() for configuration.

Usage:
    python mnemonic_cortex/app/main.py "What is the Anvil Protocol?" --model qwen2:7b
"""

import argparse
import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mnemonic_cortex.core.utils import find_project_root, setup_environment
from mnemonic_cortex.app.services.vector_db_service import VectorDBService

# --- HARDENED RAG PROMPT (v1.3) ---
RAG_PROMPT_TEMPLATE = """
**CONTEXT:**
{context}

**QUESTION:**
{question}

---
Based strictly on the context provided, provide a concise and accurate answer to the question. Do not use any prior knowledge.
"""

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def main() -> None:
    """
    (v1.3) Main application entry point for querying the Mnemonic Cortex,
    now with verifiable source citation.
    """
    parser = argparse.ArgumentParser(description="Query the Mnemonic Cortex with verifiable source citation.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--model", type=str, default="qwen2:7b", help="The local Ollama model to use.")
    args = parser.parse_args()

    print(f"--- Querying Mnemonic Cortex with: '{args.query}' ---")
    print(f"--- Using generation model: {args.model} ---")

    try:
        project_root = find_project_root()
        setup_environment(project_root)
        db_service = VectorDBService()
        retriever = db_service.get_retriever()
        
        llm = ChatOllama(model=args.model)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # --- RE-FORGED RAG CHAIN (v1.3) ---
        # This more complex chain allows us to pass the retrieved docs through
        def retrieve_and_pass_docs(query):
            docs = retriever.invoke(query)
            return {"context": format_docs(docs), "question": query, "source_docs": docs}

        rag_chain = (
            RunnableLambda(retrieve_and_pass_docs)
            | {
                "answer": prompt | llm | StrOutputParser(),
                "source_docs": lambda x: x["source_docs"]
              }
        )

        print("\n--- Generating Answer ---")
        result = rag_chain.invoke(args.query)
        
        print("\n--- Answer ---")
        print(result["answer"])
        
        # --- NEW: SOURCE CITATION ---
        print("\n--- Verifiable Sources ---")
        # Use a set to only show unique source URLs
        unique_sources = {doc.metadata.get('source_url', 'Source not found') for doc in result["source_docs"]}
        if unique_sources:
            for source in sorted(list(unique_sources)):
                print(f"- {source}")
        else:
            print("No specific sources were retrieved for this query.")
        
        print("\n--- Query Complete ---")

    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()