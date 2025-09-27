"""
Mnemonic Cortex Query Application (app/main.py)

This module implements the complete Query Pipeline in the Mnemonic Cortex RAG system.
It orchestrates the full RAG chain: user query -> embedding -> similarity search -> context retrieval -> LLM generation -> answer.

Role in RAG Pipeline:
- Parses command-line arguments for query input and model selection.
- Initializes all services (VectorDB, Embedding) and loads the persisted ChromaDB.
- Constructs the RAG chain using LangChain LCEL: retriever + prompt + Ollama LLM + output parser.
- Executes the chain to provide context-aware answers grounded in the Cognitive Genome.

Dependencies:
- VectorDBService: Loads ChromaDB and provides retriever for similarity searches.
- EmbeddingService: Used implicitly by ChromaDB for query vectorization.
- Ollama: Local LLM server must be running with the specified model (default: gemma2:9b).
- LangChain: Provides the RAG chain orchestration, prompts, and output parsing.
- Core utilities: find_project_root() and setup_environment() for configuration.

Usage:
    python mnemonic_cortex/app/main.py "What is the Anvil Protocol?" --model llama3:8b
"""

import argparse
import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mnemonic_cortex.core.utils import find_project_root, setup_environment
from mnemonic_cortex.app.services.vector_db_service import VectorDBService

# --- RAG Chain Definition ---

# Define the prompt template for the LLM
# This guides the model to use the retrieved context to answer the question.
RAG_PROMPT_TEMPLATE = """
**CONTEXT:**
{context}

**QUESTION:**
{question}

---
Based on the context provided, please provide a concise and accurate answer to the question.
"""

def main():
    """
    Main application entry point for querying the Mnemonic Cortex.
    """
    parser = argparse.ArgumentParser(
        description="Query the Mnemonic Cortex using a local RAG pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("query", type=str, help="The question to ask the Mnemonic Cortex.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemma2:9b", 
        help="The local Ollama model to use for generation (e.g., 'gemma2:9b', 'llama3:8b').\nEnsure you have pulled the model with 'ollama pull <model_name>'."
    )
    args = parser.parse_args()

    print(f"--- Querying Mnemonic Cortex with: '{args.query}' ---")
    print(f"--- Using generation model: {args.model} ---")

    try:
        # Setup environment and services
        project_root = find_project_root()
        setup_environment(project_root)
        db_service = VectorDBService()
        retriever = db_service.get_retriever()
        
        # Initialize the local LLM via Ollama
        llm = ChatOllama(model=args.model)

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Define the RAG chain using LangChain Expression Language (LCEL)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the chain with the user's query
        print("\n--- Generating Answer ---")
        answer = rag_chain.invoke(args.query)
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Query Complete ---")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n--- QUERY FAILED ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()