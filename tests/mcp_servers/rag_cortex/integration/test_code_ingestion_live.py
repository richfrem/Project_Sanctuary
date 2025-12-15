
import pytest
import shutil
from pathlib import Path
from mcp_servers.rag_cortex.operations import CortexOperations
import chromadb

@pytest.fixture
def temp_cortex_env(tmp_path):
    """
    Creates a temporary environment for Cortex integration testing.
    Returns a dict with paths and the initialized operations object.
    """
    # Setup paths
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    
    cortex_data = project_root / "mcp_servers/rag_cortex/data"
    cortex_data.mkdir(parents=True)
    
    # Initialize Operations with temp root
    # We use a persistent client in a temp dir to simulate real behavior
    ops = CortexOperations(str(project_root))
    
    yield {
        "root": project_root,
        "ops": ops,
        "data_dir": cortex_data
    }
    
    # Cleanup (handled by tmp_path, but explicitly closing clients helps)
    # ops.chroma_client = None 

def test_full_ingestion_polyglot(temp_cortex_env):
    """
    Test full ingestion of a directory containing mixed Python and JavaScript code.
    Verifies that files are converted, ingested, and queryable.
    """
    root = temp_cortex_env["root"]
    ops = temp_cortex_env["ops"]
    
    # 1. Create Source Files
    src_dir = root / "src"
    src_dir.mkdir()
    
    # Python File
    py_content = '''
def calculate_orbit(radius: float) -> float:
    """
    Calculate the orbital period.
    """
    import math
    return 2 * math.pi * radius
'''
    (src_dir / "physics.py").write_text(py_content, encoding="utf-8")
    
    # JavaScript File
    js_content = '''
/**
 * Renders the dashboard UI.
 * @param {string} userId - The user ID
 */
function renderDashboard(userId) {
    console.log("Rendering...");
    const data = fetchData(userId);
    return <div>{data}</div>;
}

const fetchData = (id) => {
    return { status: "active" };
}
'''
    (src_dir / "ui.js").write_text(js_content, encoding="utf-8")
    
    # 2. Run Full Ingestion
    print("\n--- Running Ingest Full ---")
    result = ops.ingest_full(purge_existing=True, source_directories=[str(src_dir)])
    
    assert result.status == "success"
    assert result.documents_processed >= 2
    assert result.chunks_created > 0
    
    # 3. Verify Artifacts
    assert (src_dir / "physics.py.md").exists()
    assert (src_dir / "ui.js.md").exists()
    
    # 4. Verify Retrieval (Python)
    print("\n--- Querying Python ---")
    py_results = ops.query("orbit", max_results=1)
    assert len(py_results.results) > 0
    assert "calculate_orbit" in py_results.results[0].content
    
    # 5. Verify Retrieval (JS)
    print("\n--- Querying JavaScript ---")
    js_results = ops.query("dashboard userId", max_results=1)
    assert len(js_results.results) > 0
    assert "renderDashboard" in js_results.results[0].content
    assert "fetchData" in js_results.results[0].content or "Rendering" in js_results.results[0].content


def test_incremental_ingestion_js(temp_cortex_env):
    """Test incremental ingestion of a new Typescript file."""
    root = temp_cortex_env["root"]
    ops = temp_cortex_env["ops"]
    
    # 1. Create file
    ts_file = root / "types.ts"
    ts_content = '''
interface User {
    id: number;
    name: string;
}

function validateUser(u: User): boolean {
    return u.id > 0;
}
'''
    ts_file.write_text(ts_content, encoding="utf-8")
    
    # 2. Ingest Incrementally
    print("\n--- Running Ingest Incremental ---")
    result = ops.ingest_incremental([str(ts_file)])
    
    assert result.status == "success"
    assert result.documents_added == 1
    
    # 3. Verify Retrieval
    print("\n--- Querying TypeScript ---")
    ts_results = ops.query("validateUser", max_results=1)
    assert len(ts_results.results) > 0
    assert "validateUser" in ts_results.results[0].content
