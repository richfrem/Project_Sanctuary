import pytest
from unittest.mock import patch, MagicMock

# By using `@patch`, we can replace external systems (like the DB and the LLM)
# with predictable "mock" objects during our test. This makes the test fast,
# reliable, and independent of external services.

@patch('mnemonic_cortex.app.main.ChatOllama')
@patch('mnemonic_cortex.app.main.VectorDBService')
@patch('langchain_core.runnables.RunnablePassthrough')
@patch('langchain_core.prompts.ChatPromptTemplate')
@patch('argparse.ArgumentParser.parse_args')
def test_main_query_flow_successfully(
    mock_parse_args, mock_prompt_template, mock_runnable, mock_db_service, mock_chat_ollama, capsys
):
    """
    Tests the entire main RAG pipeline, mocking all external dependencies to ensure
    the logic flows correctly from query to a printed answer.
    This is a comprehensive integration test of our application's core logic.
    """
    from mnemonic_cortex.app import main

    # --- 1. SETUP THE MOCKS ---

    # Simulate the user running: `python main.py "What is the Anvil Protocol?"`
    mock_parse_args.return_value = MagicMock(
        query="What is the Anvil Protocol?", 
        model="mock_model"
    )

    # Mock the retriever from our Vector DB Service
    mock_retriever = MagicMock()
    mock_db_service.return_value.get_retriever.return_value = mock_retriever

    # Mock the LLM to return a predictable, fake answer
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "This is the final mocked answer from the LLM."
    mock_chat_ollama.return_value = mock_llm

    # --- 2. EXECUTE THE SCRIPT ---

    main.main()

    # --- 3. VERIFY THE RESULTS ---

    # Capture what was printed to the console
    captured = capsys.readouterr()

    # Assert that the query flow executed with the expected output structure
    assert "--- Querying Mnemonic Cortex with: 'What is the Anvil Protocol?' ---" in captured.out
    assert "--- Generating Answer ---" in captured.out
    assert "--- Answer ---" in captured.out
    assert "--- Query Complete ---" in captured.out