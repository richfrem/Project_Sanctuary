import re
import json
from pathlib import Path

# ---------------------------------------------
# Global Exclusion Configuration
# ---------------------------------------------

# Load configuration from manifest
try:
    # Resolve path relative to this file
    _manifest_path = Path(__file__).parent / "exclusion_manifest.json"
    if _manifest_path.exists():
        with open(_manifest_path, "r") as f:
            _config = json.load(f)
    else:
        # Fallback if file missing (should not happen in prod)
        print(f"⚠️ Warning: exclusion_manifest.json not found at {_manifest_path}")
        _config = {}
except Exception as e:
    print(f"⚠️ Error loading exclusion_manifest.json: {e}")
    _config = {}

# 1. Directory Names (Set for O(1) lookup)
EXCLUDE_DIR_NAMES = set(_config.get("exclude_dir_names", []))

# 2. Allowed Extensions (Set)
ALLOWED_EXTENSIONS = set(_config.get("allowed_extensions", [
    '.md', '.py', '.js', '.ts', '.jsx', '.tsx',
    '.json', '.yaml', '.yml', '.toml',
    '.sh', '.bash', '.zsh', '.ps1', '.bat',
    '.txt', '.cfg', '.ini'
]))

# 3. Markdown Extensions (Set)
MARKDOWN_EXTENSIONS = set(_config.get("markdown_extensions", {'.md', '.txt', '.markdown'}))

# 4. Protected Seeds (Set)
# These files act as a BYPASS to the exclusion rules.
# If explicitly requested by a manifest, they are allowed even if their directory is excluded.
PROTECTED_SEEDS = set(_config.get("protected_seeds", []))

# 5. Always Exclude Files (Mixed List: Strings + Regex)
_static_excludes = _config.get("always_exclude_files", [])
_regex_patterns = _config.get("exclude_patterns", [])

# 6. Recursive Artifacts (Protocol 128)
RECURSIVE_ARTIFACTS = _config.get("recursive_artifacts", [])

# Compile regex patterns
_compiled_patterns = []
for pattern in _regex_patterns:
    try:
        flags = re.IGNORECASE if any(ext in pattern for ext in ["gguf", "bin", "safetensors", "log", "pyc"]) else 0
        _compiled_patterns.append(re.compile(pattern, flags))
    except re.error as e:
        print(f"⚠️ Invalid regex in exclusion_manifest: {pattern} - {e}")

# ALWAYS_EXCLUDE_FILES should include static ones, the recursive artifacts, and regex patterns
ALWAYS_EXCLUDE_FILES = _static_excludes + RECURSIVE_ARTIFACTS + _compiled_patterns
