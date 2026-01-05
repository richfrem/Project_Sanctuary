#============================================
# forge/scripts/upload_to_huggingface.py
# Purpose: Manages the upload of model weights, GGUF files, and metadata to Hugging Face Hub.
# Role: Deployment / Artifact Layer
# Used by: Phase 6 of the Forge Pipeline
#============================================

import sys
import argparse
import logging
import asyncio
import atexit
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import yaml

# --- Project Utilities Bootstrap ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT_PATH = FORGE_ROOT.parent

if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

try:
    from mcp_servers.lib.path_utils import find_project_root
    from mcp_servers.lib.logging_utils import setup_mcp_logging
    from mcp_servers.lib.hf_utils import upload_to_hf_hub
    from mcp_servers.lib.env_helper import get_env_variable
    # Use find_project_root() for consistent root discovery
    PROJECT_ROOT = Path(find_project_root())
except ImportError as e:
    print(f"❌ FATAL ERROR: Could not import core libraries: {e}")
    sys.exit(1)

# --- Logging ---
try:
    log = setup_mcp_logging("upload_to_huggingface", log_file="logs/upload_to_huggingface.log")
    log.info("Upload to Hugging Face started - using setup_mcp_logging")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("upload_to_huggingface")
    log.info("Upload to Hugging Face started - local logging fallback")

atexit.register(logging.shutdown)

# --- Configuration Constants ---
DEFAULT_UPLOAD_CONFIG: Path = FORGE_ROOT / "config" / "upload_config.yaml"


#============================================
# Function: load_config
# Purpose: Loads the upload configuration from a YAML file.
# Args: None
# Returns: (Dict[str, Any]) The configuration dictionary.
#============================================
def load_config() -> Dict[str, Any]:
    """
    Loads the upload configuration from a YAML file.

    Returns:
        The loaded configuration as a dictionary, or empty if not found.
    """
    if not DEFAULT_UPLOAD_CONFIG.exists():
        log.warning(f"Config file not found at {DEFAULT_UPLOAD_CONFIG}, using defaults.")
        return {}
    
    with open(DEFAULT_UPLOAD_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
        log.info(f"Loaded config from {DEFAULT_UPLOAD_CONFIG}")
        return config or {}


#============================================
# Function: load_environment
# Purpose: Retrieves necessary Hugging Face credentials from environment variables.
# Args: None
# Returns: (Tuple[str, Optional[str], Optional[str]]) Token, username, and repo name.
# Raises: SystemExit if HF Token is missing.
#============================================
def load_environment() -> Tuple[str, Optional[str], Optional[str]]:
    """
    Retrieves necessary Hugging Face credentials from environment variables.

    Returns:
        A tuple of (token, username, repo_name).
    """
    try:
        token = get_env_variable("HUGGING_FACE_TOKEN", required=True)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)
        
    username = get_env_variable("HUGGING_FACE_USERNAME", required=False)
    repo_name = get_env_variable("HUGGING_FACE_REPO", required=False)
    
    if not username or not repo_name:
        log.warning("HUGGING_FACE_USERNAME or HUGGING_FACE_REPO not set. Will require --repo argument.")
    
    return token, username, repo_name


#============================================
# Function: perform_upload
# Purpose: Synchronous wrapper for the asynchronous HF upload utility.
# Args:
#   repo_id (str): The destination repository ID.
#   file_paths (List[str]): List of files/folders to upload.
#   token (str): HF API Token.
#   private (bool): Whether to ensure the repo is private (default: False).
# Returns: None
# Raises: SystemExit if the upload fails.
#============================================
def perform_upload(repo_id: str, file_paths: List[str], token: str, private: bool = False) -> None:
    """
    Synchronous wrapper for the asynchronous HF upload utility.

    Args:
        repo_id: The destination repository ID.
        file_paths: List of local paths to upload.
        token: Hugging Face authentication token.
        private: Flag to mark repository as private.
    """
    log.info(f"Delegating upload to mcp_servers.lib.hf_utils...")
    
    # Run async function in sync wrapper
    result = asyncio.run(upload_to_hf_hub(
        repo_id=repo_id,
        paths=file_paths,
        token=token,
        private=private
    ))
    
    if result.success:
        log.info(f"🏆 SUCCESS: Upload complete. Repository: {result.repo_url}")
    else:
        log.error(f"❌ FATAL ERROR: Upload failed: {result.error}")
        sys.exit(1)


#============================================
# Function: main
# Purpose: Orchestrates the upload of specified Forge artifacts to Hugging Face.
# Args: None
# Returns: None
# Raises: SystemExit if no files are specified or path resolution fails.
#============================================
def main() -> None:
    """
    Orchestrates the upload of specified Forge artifacts to Hugging Face.
    
    Processes command-line arguments, resolves file paths based on configuration,
    and initiates the upload process.
    """
    parser = argparse.ArgumentParser(description="Upload Forge artifacts to Hugging Face.")
    parser.add_argument("--repo", help="HF Repo ID (username/repo). Overrides defaults.")
    parser.add_argument("--files", nargs="+", help="Explicit file/folder paths to upload.")
    parser.add_argument("--private", action="store_true", help="Mark repository as private.")
    parser.add_argument("--gguf", action="store_true", help="Upload GGUF artifacts.")
    parser.add_argument("--modelfile", action="store_true", help="Upload Ollama Modelfile.")
    parser.add_argument("--readme", action="store_true", help="Upload README.md.")
    parser.add_argument("--model-card", action="store_true", help="Upload model_card.yaml.")
    parser.add_argument("--lora", action="store_true", help="Upload LoRA adapter directory.")
    args = parser.parse_args()

    config = load_config()
    token, env_username, env_repo = load_environment()

    # Determine Repo ID
    if args.repo:
        repo_id = args.repo
    elif env_username and env_repo:
        repo_id = f"{env_username}/{env_repo}"
    elif config.get('repository', {}).get('default_repo'):
        repo_id = config['repository']['default_repo']
    else:
        log.error("No repo ID found. Specify --repo or HUGGING_FACE_USERNAME/REPO.")
        sys.exit(1)

    file_paths: List[str] = args.files or []
    files_config = config.get('files', {})

    # Path Resolution Logic
    if args.gguf:
        path = PROJECT_ROOT / files_config.get('gguf_path', "models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf")
        file_paths.append(str(path))

    if args.modelfile:
        path = PROJECT_ROOT / files_config.get('modelfile_path', "Modelfile")
        file_paths.append(str(path))

    if args.readme:
        target = 'readme_lora_path' if args.lora else 'readme_path'
        fallback = "huggingface/README_LORA.md" if args.lora else "huggingface/README.md"
        path = FORGE_ROOT / files_config.get(target, fallback)
        file_paths.append(str(path))

    if args.model_card:
        path = FORGE_ROOT / files_config.get('model_card_path', "huggingface/model_card.yaml")
        file_paths.append(str(path))

    if args.lora:
        path = PROJECT_ROOT / files_config.get('lora_path', "forge/models/Sanctuary-Qwen2-7B-v1.0-adapter")
        file_paths.append(str(path))

    # Verify paths exist before starting
    for p in file_paths:
        if not Path(p).exists():
            log.error(f"Path does not exist: {p}")
            sys.exit(1)

    if not file_paths:
        log.error("No files specified for upload.")
        sys.exit(1)

    log.info("=== Hugging Face Upload Session ===")
    log.info(f"Target Repository: {repo_id}")
    log.info(f"Artifact Count: {len(file_paths)}")

    perform_upload(repo_id, file_paths, token, args.private)


if __name__ == "__main__":
    main()