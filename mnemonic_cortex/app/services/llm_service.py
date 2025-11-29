"""
LLM Service
Provides an interface for interacting with Large Language Models (Ollama/Gemini)
to perform reasoning tasks such as query structuring and decomposition.
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class StructuredQuery(BaseModel):
    semantic_query: str = Field(description="The keyword-rich search query optimized for vector search")
    reasoning: str = Field(description="Brief explanation of why this query was constructed")
    filters: Dict[str, Any] = Field(description="Metadata filters to apply (e.g., {'source': 'protocol'})")

class LLMService:
    """Service for LLM interactions."""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize LLM Service.
        
        Args:
            project_root: Path to project root for loading .env
        """
        if project_root:
            self.project_root = Path(project_root)
            load_dotenv(dotenv_path=self.project_root / ".env")
        
        # Initialize LLM (Default to Ollama/Qwen as per agentic_query.py)
        # TODO: Make model configurable via env vars
        self.model_name = os.getenv("LLM_MODEL", "Sanctuary-Qwen2-7B:latest")
        self.llm = Ollama(model=self.model_name, temperature=0.1)
        
        # Initialize parsers
        self.query_parser = JsonOutputParser(pydantic_object=StructuredQuery)

    def generate_structured_query(self, natural_query: str) -> Dict[str, Any]:
        """
        Translate a natural language query into a structured query.
        
        Args:
            natural_query: The user's raw query
            
        Returns:
            Dict containing semantic_query, reasoning, and filters
        """
        template = """
        You are an expert search query optimizer for the Project Sanctuary knowledge base.
        Your goal is to translate a natural language user request into a precise, structured search query.
        
        The knowledge base contains:
        - Protocols (e.g., "Protocol 101", "P87")
        - Chronicles (Daily logs, "Entry #123")
        - ADRs (Architecture Decision Records)
        - Code documentation
        
        INSTRUCTIONS:
        1. Analyze the user's request to understand the core intent.
        2. Extract specific keywords, protocol numbers, and technical terms.
        3. Formulate a 'semantic_query' optimized for vector search (keyword-heavy).
        4. Identify any implicit filters (e.g., if user asks about "Protocols", filter by source).
        5. Provide a brief 'reasoning' for your choices.
        
        User Request: {query}
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={"format_instructions": self.query_parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | self.query_parser
        
        try:
            print(f"--- [LLM Service] Generating structured query for: '{natural_query}' ---")
            result = chain.invoke({"query": natural_query})
            return result
        except Exception as e:
            print(f"--- [LLM Service] Error generating query: {e} ---")
            # Fallback to simple pass-through
            return {
                "semantic_query": natural_query,
                "reasoning": "LLM generation failed, falling back to raw query.",
                "filters": {}
            }
