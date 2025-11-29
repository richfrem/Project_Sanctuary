import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path to allow importing modules
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--real-llm", 
        action="store_true", 
        default=False, 
        help="Run tests against the real local LLM (Ollama). Default is to mock."
    )

@pytest.fixture
def real_llm(request):
    """Return True if --real-llm flag is set."""
    return request.config.getoption("--real-llm")

@pytest.fixture
def mock_llm_response():
    """Return a default mock response."""
    return "This is a mocked response from the Mnemonic Cortex."

@pytest.fixture
def llm_service(real_llm, mock_llm_response):
    """
    Fixture that returns a context manager for patching ChatOllama.
    If --real-llm is set, it does nothing (uses real class).
    If not set, it mocks ChatOllama to return a fixed response.
    """
    if real_llm:
        # No-op context manager
        class RealLLMContext:
            def __enter__(self): return None
            def __exit__(self, *args): pass
        return RealLLMContext()
    else:
        # Mock the ChatOllama class
        with patch("mnemonic_cortex.app.services.rag_service.ChatOllama") as mock_class:
            mock_instance = mock_class.return_value
            # Mock the invoke method of the chain (which is what RAGService calls)
            # RAGService: chain = prompt | self.llm | StrOutputParser()
            # This is tricky to mock perfectly because of the LCEL pipe syntax.
            # Easier to mock the RAGService.query method or the LLM's invoke if we can intercept it.
            
            # Alternative: Mock the invoke method of the LLM instance itself
            # But RAGService constructs a chain.
            
            # Let's mock the entire chain execution in RAGService if possible, 
            # OR we can mock ChatOllama to return a MagicMock that behaves like a runnable.
            
            # When chain.invoke is called, it calls invoke on the last element (StrOutputParser)
            # which calls invoke on the previous...
            
            # Simplest approach for RAGService unit testing:
            # Mock the `invoke` method of the chain. But we don't have access to the chain object easily.
            
            # Let's try patching ChatOllama to return a mock that produces a specific AIMessage
            from langchain_core.messages import AIMessage
            mock_instance.invoke.return_value = AIMessage(content=mock_llm_response)
            
            # Also need to handle the pipe operator `|` if we want to be robust, 
            # but usually mocking the instance is enough if the chain construction uses it.
            # However, `prompt | llm` creates a RunnableSequence.
            
            yield mock_class
