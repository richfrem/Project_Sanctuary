import logging
import sys
import os
from pathlib import Path

def setup_mcp_logging(name: str, log_file: str = "logs/mcp_server.log"):
    """
    Setup logging for MCP servers to write to a file and console.
    
    Args:
        name: Logger name
        log_file: Path to log file (relative to project root)
    """
    # Find project root
    current = Path(__file__).resolve().parent
    project_root = None
    
    while current.parent != current:
        if (current / "mcp_servers").exists():
            project_root = current
            break
        current = current.parent
    
    if project_root is None:
        project_root = Path.cwd()

    log_path = project_root / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        # File Handler - Only if MCP_LOGGING is true
        if os.getenv("MCP_LOGGING", "").lower() == "true":
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Console Handler (stderr)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
