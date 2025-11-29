import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from mnemonic_cortex.app.services.vector_db_service import VectorDBService

class RAGService:
    """
    Service for RAG operations: Retrieval and Generation.
    Encapsulates the logic previously found in main.py.
    """
    def __init__(self, project_root: str, model_name: str = "Sanctuary-Qwen2-7B:latest"):
        self.project_root = project_root
        self.model_name = model_name
        self.vector_db = VectorDBService()
        self.llm = ChatOllama(model=model_name)

    def query(self, query_text: str, retrieve_only: bool = False) -> str | list:
        """
        Execute a RAG query.
        
        Args:
            query_text: The user's question.
            retrieve_only: If True, returns the retrieved documents instead of an answer.
            
        Returns:
            Generated answer string (default) or list of Documents (if retrieve_only=True).
        """
        # 1. Retrieve context
        retrieved_docs = self.vector_db.query(query_text)
        
        if retrieve_only:
            return retrieved_docs
            
        if not retrieved_docs:
            return "I could not find any relevant information in the Cortex to answer your question."

        # 2. Format context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 3. Generate answer
        template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {context} 

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({"question": query_text, "context": context})
