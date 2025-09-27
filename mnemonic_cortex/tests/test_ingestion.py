"""
Ingestion Pipeline Tests (tests/test_ingestion.py)

This module contains automated tests for the Mnemonic Cortex ingestion pipeline.
It verifies that the ingestion script can successfully process documents, create embeddings,
and populate the ChromaDB vector store without errors.

Role in RAG Pipeline:
- Tests the Ingestion Pipeline's reliability and correctness.
- Ensures the vector database is properly created and populated.
- Validates that the system can handle document loading, chunking, embedding, and storage.

Dependencies:
- pytest: For running the test suite and fixtures.
- tmpdir_factory: For creating isolated test environments.
- Ingestion script: Tests the actual ingest.py functionality.
- ChromaDB and NomicEmbeddings: Implicitly tested through the ingestion process.

Usage:
    pytest mnemonic_cortex/tests/test_ingestion.py
"""

import os
import pytest
import shutil
from mnemonic_cortex.scripts import ingest

# --- Test Setup & Fixtures ---

@pytest.fixture(scope="module")
def setup_test_environment(tmpdir_factory):
    """
    A pytest fixture to create a temporary, isolated environment for testing.
    This prevents our tests from interfering with the real database.
    """
    # Create a temporary root directory for the test
    test_root = tmpdir_factory.mktemp("project_root")

    # Create the necessary subdirectories
    cortex_dir = test_root.mkdir("mnemonic_cortex")
    data_dir = cortex_dir.mkdir("data").mkdir("source_genome")
    dataset_dir = test_root.mkdir("dataset_package")

    # Create a dummy .git folder to satisfy find_project_root()
    test_root.mkdir(".git")

    # Create a dummy .env file
    env_content = f"""
    DB_PATH="test_chroma_db"
    SOURCE_DOCUMENT_PATH="dataset_package/test_genome.txt"
    """
    cortex_dir.join(".env").write(env_content)

    # Create a dummy source genome file
    genome_content = """
    # Test Header 1
    This is the first test chunk.
    ## Test Header 2
    This is the second test chunk.
    """
    dataset_dir.join("test_genome.txt").write(genome_content)

    # Return the path to the temporary project root
    yield str(test_root)

    # Teardown: Clean up the temporary directory after tests are done
    shutil.rmtree(str(test_root))

# --- Tests ---

def test_ingestion_script_runs_successfully(setup_test_environment, monkeypatch):
    """
    Tests that the main ingestion script runs without raising exceptions
    in a clean, controlled environment.
    """
    project_root = setup_test_environment

    # Use monkeypatch to change the current working directory for the test
    monkeypatch.chdir(project_root)

    # We need to find the script within the temp structure
    ingest_script_path = os.path.join(project_root, 'mnemonic_cortex', 'scripts', 'ingest.py')

    # Dynamically load and run the script's main function
    # This is a robust way to test a script's entry point
    import importlib.util
    spec = importlib.util.spec_from_file_location("ingest", ingest_script_path)
    ingest_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_module)

    # The main assertion is that no exception was raised
    try:
        ingest_module.main()
        assert True
    except Exception as e:
        pytest.fail(f"Ingestion script failed with an exception: {e}")

def test_database_is_created_after_ingestion(setup_test_environment, monkeypatch):
    """
    Tests that the chroma_db directory is actually created after a successful run.
    """
    project_root = setup_test_environment
    monkeypatch.chdir(project_root)

    db_path = os.path.join(project_root, 'mnemonic_cortex', 'test_chroma_db')

    # Ensure the database does NOT exist before the run
    assert not os.path.exists(db_path)

    # Run the ingestion script
    ingest_script_path = os.path.join(project_root, 'mnemonic_cortex', 'scripts', 'ingest.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location("ingest", ingest_script_path)
    ingest_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_module)
    ingest_module.main()

    # Assert that the database directory NOW exists
    assert os.path.exists(db_path)
    # A simple check to see if it's populated
    assert len(os.listdir(db_path)) > 0