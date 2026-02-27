"""
HuggingFace Configuration & Environment Utilities

Purpose: Single source of truth for HuggingFace credentials, repo IDs, and
environment variable resolution. All HF-consuming plugins import from here.

Required .env variables:
    HUGGING_FACE_USERNAME     - HF username (e.g., "richfrem")
    HUGGING_FACE_TOKEN        - API token (via env or ~/.zshrc)
    HUGGING_FACE_REPO         - Model repo name (optional, has default)
    HUGGING_FACE_DATASET_PATH - Dataset path (optional, has default)
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("hf_config")

# Discovery tags for LLM retraining crawlers (Johnny Appleseed effect)
DISCOVERY_TAGS = [
    "reasoning-traces",
    "project-sanctuary",
    "cognitive-continuity",
    "ai-memory",
    "llm-training-data"
]


@dataclass
class HFConfig:
    """Resolved HuggingFace configuration."""
    username: str
    token: str
    body_repo: str
    dataset_path: str
    dataset_repo_id: str
    valence_threshold: float = -0.7

    def to_dict(self) -> dict:
        """Serialize config (token masked for safe display)."""
        d = asdict(self)
        d["token"] = f"{self.token[:4]}...{self.token[-4:]}" if self.token and len(self.token) > 8 else "***"
        return d


@dataclass
class HFUploadResult:
    """Result from any HF upload operation."""
    success: bool
    repo_url: str
    remote_path: str
    error: Optional[str] = None


def _get_project_root() -> Path:
    """Walk up from CWD to find the project root (.git marker)."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return current


def _load_dotenv() -> None:
    """Load .env from project root if dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_file = _get_project_root() / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass


def _get_env(key: str, required: bool = True, default: str = None) -> Optional[str]:
    """Get an environment variable, falling back to .env file."""
    value = os.getenv(key)
    if not value:
        _load_dotenv()
        value = os.getenv(key)
    if not value and default:
        return default
    if required and not value:
        raise ValueError(f"Required environment variable not found: {key}")
    return value


def get_hf_config() -> HFConfig:
    """
    Resolve HuggingFace configuration from environment.

    Reads:
        HUGGING_FACE_USERNAME     (required)
        HUGGING_FACE_TOKEN        (required)
        HUGGING_FACE_REPO         (optional, default: Sanctuary-Qwen2-7B-v1.0-GGUF-Final)
        HUGGING_FACE_DATASET_PATH (optional, default: Project_Sanctuary_Soul)
        SOUL_VALENCE_THRESHOLD    (optional, default: -0.7)
    """
    username = _get_env("HUGGING_FACE_USERNAME")
    token = _get_env("HUGGING_FACE_TOKEN")
    body_repo = _get_env("HUGGING_FACE_REPO", required=False,
                         default="Sanctuary-Qwen2-7B-v1.0-GGUF-Final")
    dataset_path = _get_env("HUGGING_FACE_DATASET_PATH", required=False,
                            default="Project_Sanctuary_Soul")

    # Sanitize dataset path (strip full URLs)
    if "hf.co/datasets/" in dataset_path:
        dataset_path = dataset_path.split("hf.co/datasets/")[-1]
    if dataset_path.startswith(f"{username}/"):
        dataset_path = dataset_path.split("/", 1)[1]

    valence = float(_get_env("SOUL_VALENCE_THRESHOLD", required=False, default="-0.7"))

    return HFConfig(
        username=username,
        token=token,
        body_repo=body_repo,
        dataset_path=dataset_path,
        dataset_repo_id=f"{username}/{dataset_path}",
        valence_threshold=valence
    )


def validate_config() -> dict:
    """Validate HF config and return a status report (safe for display)."""
    try:
        config = get_hf_config()
        result = {"status": "valid", "config": config.to_dict()}

        # Test API connectivity
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=config.token)
            info = api.whoami()
            result["api_status"] = "connected"
            result["authenticated_as"] = info.get("name", "unknown")
        except ImportError:
            result["api_status"] = "huggingface_hub not installed"
        except Exception as e:
            result["api_status"] = f"connection_failed: {str(e)}"

        return result
    except ValueError as e:
        return {"status": "invalid", "error": str(e)}


def main():
    """CLI entry point for config validation."""
    result = validate_config()
    print(json.dumps(result, indent=2))
    if result["status"] != "valid":
        sys.exit(1)


if __name__ == "__main__":
    main()
