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


def check_container_running(container_name: str = "sanctuary_vector_db") -> bool:
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



def start_container(
    container_name: str,
    image_name: str,
    port_mapping: str,
    volumes: list[str] = None,
    env_vars: list[str] = None,
    cmd_args: list[str] = None
) -> tuple[bool, str]:
    """
    Generic function to start a Podman container.
    """
    # Check if container exists but is stopped
    try:
        result = subprocess.run(
            ["podman", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if container_name in result.stdout:
            subprocess.run(
                ["podman", "start", container_name],
                capture_output=True,
                timeout=10
            )
            return True, f"Started existing container: {container_name}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout checking for existing container {container_name}"

    # Create new container
    try:
        cmd = [
            "podman", "run", "-d",
            "--name", container_name,
            "-p", port_mapping,
        ]
        
        if volumes:
            for v in volumes:
                cmd.extend(["-v", v])
        
        if env_vars:
            for e in env_vars:
                cmd.extend(["-e", e])
                
        cmd.append("--restart=unless-stopped")
        cmd.append(image_name)
        
        if cmd_args:
            cmd.extend(cmd_args)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, f"Created and started container: {container_name}"
        else:
            return False, f"Failed to create container {container_name}: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout creating container {container_name}"
    except Exception as e:
        return False, f"Error creating container {container_name}: {str(e)}"

def check_service_healthy(host: str, port: int, service_type: str = "chroma", timeout: int = 5) -> bool:
    """Check if a service is healthy."""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            if service_type == "chroma":
                import chromadb
                client = chromadb.HttpClient(host=host, port=port)
                client.list_collections()
                return True
            elif service_type == "ollama":
                resp = requests.get(f"http://{host}:{port}/api/tags")
                if resp.status_code == 200:
                    return True
        except Exception:
            time.sleep(1)
            
    return False

def ensure_chromadb_running(project_root: str) -> tuple[bool, str]:
    """Ensure ChromaDB is running."""
    project_path = Path(project_root).resolve()
    data_dir = os.getenv("CHROMA_DATA_PATH", ".vector_data")
    if os.path.isabs(data_dir):
        vector_data_path = Path(data_dir)
    else:
        vector_data_path = project_path / data_dir
    vector_data_path.mkdir(parents=True, exist_ok=True)
    
    container_name = "sanctuary_vector_db"
    port = int(os.getenv("CHROMA_PORT", "8000"))
    
    if not check_podman_available():
        return False, "Podman not available"
        
    if check_container_running(container_name):
        return True, "ChromaDB container running"
        
    start_success, start_msg = start_container(
        container_name=container_name,
        image_name="chromadb/chroma:latest",
        port_mapping=f"{port}:8000",
        volumes=[f"{vector_data_path}:/chroma/chroma:Z"],
        env_vars=["IS_PERSISTENT=TRUE", "ANONYMIZED_TELEMETRY=FALSE"]
    )
        
    if not start_success:
        return False, f"Failed to start ChromaDB: {start_msg}"
        
    # Wait for healthy
    if check_service_healthy("127.0.0.1", port, "chroma", timeout=15):
        return True, "ChromaDB started and healthy"
    return False, "ChromaDB started but unhealthy"

def ensure_ollama_running(project_root: str) -> tuple[bool, str]:
    """Ensure Ollama is running."""
    container_name = "sanctuary-ollama-mcp"
    port = 11434
    
    if not check_podman_available():
        return False, "Podman not available"
        
    if check_container_running(container_name):
        if check_service_healthy("127.0.0.1", port, "ollama", timeout=2):
            return True, "Ollama container running"
        # If running but unhealthy, we might want to restart? For now, just report.
        
    # Standard Ollama setup
    # Note: Requires volume for models
    ollama_models_path = Path(project_root).resolve() / ".ollama_models"
    ollama_models_path.mkdir(parents=True, exist_ok=True)
    
    start_success, start_msg = start_container(
        container_name=container_name,
        image_name="ollama/ollama:latest",
        port_mapping=f"{port}:11434",
        volumes=[f"{ollama_models_path}:/root/.ollama:Z"]
    )
    
    if not start_success:
        return False, f"Failed to start Ollama: {start_msg}"
        
    if check_service_healthy("localhost", port, "ollama", timeout=15):
        return True, "Ollama started and healthy"
    return False, "Ollama started but unhealthy"
