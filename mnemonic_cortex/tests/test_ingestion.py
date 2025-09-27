import os
import pytest
import shutil
import subprocess

# --- Test Setup & Fixtures ---

@pytest.fixture(scope="function")
def setup_test_environment(tmpdir_factory):
    """
    A pytest fixture to create a temporary, isolated project structure
    AND populate it with the necessary source files for testing.
    """
    test_root = tmpdir_factory.mktemp("project_root")

    # Create the full directory structure
    cortex_dir = test_root.mkdir("mnemonic_cortex")
    data_dir = cortex_dir.mkdir("data").mkdir("source_genome")
    dataset_dir = test_root.mkdir("dataset_package")
    scripts_dir = cortex_dir.mkdir("scripts")

    test_root.mkdir(".git") # Satisfies find_project_root()

    # --- CRITICAL FIX: Copy the actual source code into the test environment ---
    real_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Copy the script to be tested
    shutil.copy(
        os.path.join(real_project_root, 'mnemonic_cortex', 'scripts', 'ingest.py'),
        str(scripts_dir)
    )
    # Copy the utility functions it depends on
    shutil.copytree(
        os.path.join(real_project_root, 'mnemonic_cortex', 'core'),
        str(cortex_dir.join('core'))
    )
    # Create empty __init__.py files to make them importable packages
    cortex_dir.join("__init__.py").write("")
    scripts_dir.join("__init__.py").write("")
    cortex_dir.join('core').join("__init__.py").write("")
    # --- END FIX ---

    # Create a dummy .env and source file for the test
    cortex_dir.join(".env").write("DB_PATH=test_chroma_db\nSOURCE_DOCUMENT_PATH=dataset_package/test_genome.txt")
    dataset_dir.join("test_genome.txt").write("# Test Header\nTest content.")

    yield str(test_root)

    # Teardown
    shutil.rmtree(str(test_root))

# --- Tests ---

def test_ingestion_script_runs_successfully(setup_test_environment, monkeypatch):
    project_root = setup_test_environment
    monkeypatch.chdir(project_root)

    ingest_script_path = os.path.join('mnemonic_cortex', 'scripts', 'ingest.py')

    # Run the script as a subprocess for maximum isolation
    result = subprocess.run(['python3', ingest_script_path], capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
    assert "--- Ingestion Process Complete ---" in result.stdout

def test_database_is_created_after_ingestion(setup_test_environment, monkeypatch):
    project_root = setup_test_environment
    monkeypatch.chdir(project_root)

    db_path = os.path.join('mnemonic_cortex', 'test_chroma_db')
    assert not os.path.exists(db_path)

    ingest_script_path = os.path.join('mnemonic_cortex', 'scripts', 'ingest.py')
    subprocess.run(['python3', ingest_script_path], check=True)

    assert os.path.exists(db_path)
    assert len(os.listdir(db_path)) > 0