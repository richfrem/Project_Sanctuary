"""
Container lifecycle management for RAG Cortex ChromaDB service.

Automatically starts and manages the ChromaDB container when the MCP server starts.
"""
import subprocess
import time
import os
import requests
from pathlib import Path


def check_podman_available() -> bool:
    """Check if Podman is installed and machine is running."""
    try:
        result = subprocess.run(
            ["podman", "machine", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "Currently running" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_container_running(container_name: str = "sanctuary-vector-db") -> bool:
    """Check if ChromaDB container is running."""
    try:
        result = subprocess.run(
            ["podman", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return container_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_chromadb_healthy(host: str = "localhost", port: int = 8000, timeout: int = 2) -> bool:
    """Check if ChromaDB service is responding."""
    try:
        import chromadb
        client = chromadb.HttpClient(host=host, port=port)
        # Try to list collections as a health check
        client.list_collections()
        return True
    except Exception:
        return False


def start_chromadb_container(
    project_root: str,
    container_name: str = "sanctuary-vector-db",
    port: int = 8000
) -> tuple[bool, str]:
    """
    Start ChromaDB container using Podman.
    
    Args:
        project_root: Path to project root
        container_name: Name for the container
        port: Port to expose ChromaDB on
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    project_path = Path(project_root).resolve()
    
    # Get data path from env or default to .vector_data
    data_dir_name = os.getenv("CHROMA_DATA_PATH", ".vector_data")
    
    # Handle absolute vs relative paths
    if os.path.isabs(data_dir_name):
        vector_data_path = Path(data_dir_name)
    else:
        vector_data_path = project_path / data_dir_name
    
    # Ensure data directory exists
    vector_data_path.mkdir(parents=True, exist_ok=True)
    
    # Check if container exists but is stopped
    try:
        result = subprocess.run(
            ["podman", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if container_name in result.stdout:
            # Container exists, just start it
            subprocess.run(
                ["podman", "start", container_name],
                capture_output=True,
                timeout=10
            )
            return True, f"Started existing container: {container_name}"
    except subprocess.TimeoutExpired:
        return False, "Timeout checking for existing container"
    
    # Create new container
    try:
        cmd = [
            "podman", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:8000",
            "-v", f"{vector_data_path}:/chroma/chroma:Z",
            "-e", "IS_PERSISTENT=TRUE",
            "-e", "ANONYMIZED_TELEMETRY=FALSE",
            "--restart", "unless-stopped",
            "chromadb/chroma:latest"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, f"Created and started container: {container_name}"
        else:
            return False, f"Failed to create container: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout creating container"
    except Exception as e:
        return False, f"Error creating container: {str(e)}"


def _wait_for_chromadb(max_wait_seconds: int = 30) -> bool:
    """
    Wait for ChromaDB to become responsive.
    
    Args:
        max_wait_seconds: Maximum time to wait in seconds
        
    Returns:
        True if ChromaDB is responsive, False otherwise
    """
    import time
    import chromadb
    
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            # Try to connect using ChromaDB client
            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            # Try to list collections as a health check
            client.list_collections()
            return True
        except Exception:
            time.sleep(1)
    
    return False


def ensure_chromadb_running(project_root: str) -> tuple[bool, str]:
    """
    Ensure ChromaDB container is running, starting it if necessary.
    
    This is the main entry point for container lifecycle management.
    
    Args:
        project_root: Path to project root
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Get configuration from environment
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    
    # If CHROMA_HOST is not localhost/vector-db, assume external service
    if host not in ["localhost", "127.0.0.1", "vector-db"]:
        if check_chromadb_healthy(host, port):
            return True, f"Connected to external ChromaDB at {host}:{port}"
        else:
            return False, f"Cannot connect to external ChromaDB at {host}:{port}"
    
    # Check if Podman is available
    if not check_podman_available():
        return False, "Podman is not running. Please start Podman Desktop or run 'podman machine start'"
    
    # Check if container is already running
    if check_container_running():
        if check_chromadb_healthy("localhost", port):
            return True, "ChromaDB container already running and healthy"
        else:
            return False, "ChromaDB container running but not responding"
    
    # Start the container
    success, message = start_chromadb_container(project_root, port=port)
    if not success:
        return False, message
    
    # Wait for ChromaDB to be ready
    if _wait_for_chromadb(max_wait_seconds=30):
        return True, f"{message} - ChromaDB is ready"
    else:
        return False, f"{message} - ChromaDB started but not responding"
