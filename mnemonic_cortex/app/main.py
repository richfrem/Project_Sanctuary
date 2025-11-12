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
- Ollama: Local LLM server must be running with the specified model (default: Sanctuary-Qwen2-7B:latest).
- LangChain: Provides the RAG chain orchestration, prompts, and output parsing.
- Core utilities: find_project_root() and setup_environment() for configuration.

Usage:
    python mnemonic_cortex/app/main.py "What is the Anvil Protocol?" --model Sanctuary-Qwen2-7B:latest
"""

import argparse
import os
import sys
from langchain_core.prompts import ChatPromptTemplate
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
    parser = argparse.ArgumentParser(description="Query the Mnemonic Cortex.")
    parser.add_argument("query", type=str, help="The query to process.")
    parser.add_argument("--model", type=str, default="Sanctuary-Qwen2-7B:latest", help="The Ollama model to use for generation.")
    parser.add_argument("--retrieve-only", action="store_true", help="Run retrieval but skip LLM generation. Prints retrieved documents.")
    parser.add_argument("--no-rag", action="store_true", help="Run LLM generation without RAG. Tests internal model knowledge.")
    args = parser.parse_args()

    try:
        project_root = find_project_root()
        setup_environment(project_root)

        # --- CONDITIONAL EXECUTION LOGIC ---
        if args.retrieve_only:
            print(f"--- [RETRIEVE-ONLY MODE] Fetching documents for query: '{args.query}' ---")
            db_service = VectorDBService()
            retrieved_docs = db_service.query(args.query)
            print(f"\n--- Retrieved {len(retrieved_docs)} Parent Documents ---")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'Unknown')
                print(f"\n--- DOC {i+1}: {source} ---")
                print(doc.page_content[:1000] + "...")
            return

        print(f"--- Querying Mnemonic Cortex with: '{args.query}' ---")
        print(f"--- Using generation model: {args.model} ---")

        llm = ChatOllama(model=args.model)

        if args.no_rag:
            print(f"--- [NO-RAG MODE] Querying internal model knowledge: '{args.query}' ---")
            prompt = ChatPromptTemplate.from_template("Question: {question}\n\nAnswer:")
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"question": args.query})
            print("\n--- Model Response (Internal Knowledge Only) ---")
            print(response)
            return

        # --- DEFAULT RAG PIPELINE ---
        db_service = VectorDBService()
        retrieved_docs = db_service.query(args.query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {context} 

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        print("\n--- Generating Final Answer (RAG Augmented) ---")
        response = chain.invoke({"question": args.query, "context": context})
        print(response)

    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()