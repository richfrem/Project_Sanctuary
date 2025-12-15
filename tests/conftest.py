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


@pytest.fixture(scope="session")
def mcp_servers():
    """
    Start all 12 MCP servers for E2E tests.
    Tears down after test session completes.
    
    This fixture enables full MCP client call lifecycle testing by:
    1. Starting all MCP servers using the standard start_mcp_servers.py script
    2. Waiting for them to initialize
    3. Yielding control to tests
    4. Cleaning up all server processes on teardown
    
    Usage:
        @pytest.mark.e2e
        async def test_via_mcp_client(mcp_servers):
            async with ClientSession() as session:
                result = await session.call_tool("tool_name", {...})
    
    Note: This uses the same start_mcp_servers.py script that VS Code uses,
    ensuring consistency between development and testing environments.
    """
    import subprocess
    import time
    import signal
    
    project_root = Path(__file__).parent.parent
    start_script = project_root / "mcp_servers" / "start_mcp_servers.py"
    venv_python = project_root / ".venv" / "bin" / "python"
    
    # Start servers using the standard launcher script
    print("\n🚀 Starting all 12 MCP servers for E2E tests...")
    print(f"   Using: {start_script}")
    
    # Use --run flag for foreground mode (easier to manage in tests)
    proc = subprocess.Popen(
        [str(venv_python), str(start_script), "--run"],
        cwd=project_root,
        stdin=subprocess.PIPE, # Keep stdin open so servers don't exit on EOF
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create new process group for clean shutdown
    )
    
    # Give servers time to boot (adjust if needed)
    time.sleep(5)
    
    # Check if process is still running
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"MCP servers failed to start:\n"
            f"STDOUT: {stdout.decode()}\n"
            f"STDERR: {stderr.decode()}"
        )
    
    print("✓ MCP servers started successfully")
    
    yield  # Tests run here
    
    # Teardown: Kill all server processes
    print("\n🛑 Stopping MCP servers...")
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
        print("✓ MCP servers stopped cleanly")
    except subprocess.TimeoutExpired:
        print("⚠ MCP servers did not stop gracefully, forcing kill...")
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        # Process already terminated
        print("✓ MCP servers already stopped")
