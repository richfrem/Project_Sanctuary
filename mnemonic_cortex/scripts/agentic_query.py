# mnemonic_cortex/scripts/agentic_query.py
import sys
import subprocess
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_rag_query(query: str):
    main_script_path = project_root / "mnemonic_cortex" / "app" / "main.py"
    print(f"\n--- [AGENT] Passing hardened query to Mnemonic Cortex RAG pipeline ---")
    print(f"--- [AGENT] Query: '{query}' ---")
    subprocess.run([sys.executable, str(main_script_path), query])

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 agentic_query.py \"<your high-level goal>\"")
        sys.exit(1)

    high_level_goal = sys.argv[1]
    print(f"--- [AGENT] Received high-level goal: '{high_level_goal}' ---")
    load_dotenv(dotenv_path=project_root / ".env")
    
    llm = Ollama(model="Sanctuary-Qwen2-7B:latest")

    # --- HARDENED PROMPT TEMPLATE V2 ---
    # This prompt is highly directive, forcing the LLM to act as a keyword extractor.
    template = """
You are a search query extraction engine. Your only function is to analyze the user's goal and extract a single-line, keyword-rich search query.

CRITICAL INSTRUCTIONS:
1.  Read the user's goal carefully.
2.  Identify and extract all named entities, specific protocol numbers (e.g., "P101", "Protocol 63"), and unique doctrinal phrases (e.g., "Unbreakable Commit", "Cognitive Diversity", "Steward's Litmus Test").
3.  Combine these extracted keywords into a single, space-separated line. This is for a semantic vector search.
4.  DO NOT answer the user's goal. DO NOT add any commentary or explanation. Your entire output must be ONLY the query itself.

High-level goal: {goal}
Refined Query:
"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    print("--- [AGENT] Using LLM with hardened prompt to extract precise query... ---")
    refined_query = chain.invoke({"goal": high_level_goal})
    
    run_rag_query(refined_query.strip())

if __name__ == "__main__":
    main()
