#!/usr/bin/env python3
"""
<<<<<<< HEAD
rlm_config.py
=======
rlm_config.py (Shared Module)
>>>>>>> origin/main
=====================================

Purpose:
    Centralized configuration and utility logic for the RLM Toolchain.
<<<<<<< HEAD
    Loads profile settings exclusively from `.agent/learning/rlm_profiles.json`,
    making this module the Single Source of Truth for all RLM operations.

Layer: Curate / Rlm

Usage:
    # No direct usage (Shared Module)
    from rlm_config import RLMConfig, load_cache, save_cache, collect_files

Related:
    - distiller.py
    - query_cache.py
    - cleanup_cache.py
    - inventory.py
=======
    Implement the "Manifest Factory" pattern (ADR-0024) to dynamically
    resolve manifests and cache files based on the Analysis Type (Legacy vs Tool).

    This module is the Single Source of Truth for RLM logic.

Layer: Curate / Rlm

Supported Object Types:
    - RLM Config (Legacy Documentation)
    - RLM Config (Tool Discovery)

CLI Arguments (Consumed by Scripts):
    --type  : [legacy|tool] Selects the configuration profile.

Usage Examples:
    # No direct usage (Shared Module)

Input Files:
    - plugins/rlm-factory/resources/manifest-index.json
    - (Manifests referenced by the index)

Output:
    - RLMConfig object (Typed configuration)
    - Shared Utility Pointers (load_cache, save_cache, etc.)

Key Classes/Functions:
    - RLMConfig: Loads and validates configuration from the factory index.
    - load_cache(): Shared cache loader.
    - save_cache(): Shared cache persister.
    - collect_files(): Centralized file discovery logic (Glob vs Inventory).

Script Dependencies:
    - None (this is a dependency for others)

Consumed by:
    - plugins/rlm-factory/scripts/distiller.py
    - plugins/rlm-factory/scripts/query_cache.py
    - plugins/rlm-factory/scripts/cleanup_cache.py
>>>>>>> origin/main
"""
import os
import sys
import json
<<<<<<< HEAD
import glob as globmod
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

# ============================================================
# PATHS
# File is at: plugins/rlm-factory/skills/rlm-curator/scripts/rlm_config.py
# Root is 6 levels up (scripts→rlm-curator→skills→rlm-factory→plugins→ROOT)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PROFILES_PATH = PROJECT_ROOT / ".agent" / "learning" / "rlm_profiles.json"


class RLMConfig:
    """
    Profile-based configuration for the RLM distillation toolchain.

    Loads all settings from a named profile in `rlm_profiles.json`. This
    class is the single configuration entry point for all RLM scripts.

    Attributes:
        profile_name: Name of the loaded profile.
        root: Resolved project root path.
        manifest_path: Resolved path to the manifest JSON.
        cache_path: Resolved path to the cache JSON.
        include_patterns: Glob patterns to include during file collection.
        exclude_patterns: Substrings to exclude during file collection.
        allowed_suffixes: File extensions to process.
        llm_model: Ollama model name to use for distillation.
        prompt_template: Loaded prompt text for the LLM.
    """

    def __init__(
        self,
        profile_name: str,
        project_root: Optional[Path] = None
    ) -> None:
        """
        Initialize RLMConfig from a named profile.

        Args:
            profile_name: Name of the profile to load from rlm_profiles.json.
            project_root: Optional override for the project root path.

        Raises:
            SystemExit: If the profile is not found or required keys are missing.
        """
        self.root = project_root or PROJECT_ROOT
        self.profile_name = profile_name

        # Load and validate the named profile from JSON
        profiles_data = self._load_profiles_json()
        profile = profiles_data.get("profiles", {}).get(profile_name)

        if not profile:
            available = list(profiles_data.get("profiles", {}).keys())
            print(f"❌ RLM Profile '{profile_name}' not found. Available: {available}")
            sys.exit(1)

        self.description = profile.get("description", f"RLM Cache: {profile_name}")
        self.allowed_suffixes = profile.get("extensions", [".md", ".py"])
        self.llm_model = profile.get("llm_model", "granite3.2:8b")

        # Resolve all paths relative to project root
        manifest_rel = profile.get("manifest")
        cache_rel = profile.get("cache")

        if not manifest_rel or not cache_rel:
            print(f"❌ Profile '{profile_name}' is missing 'manifest' or 'cache' path.")
            sys.exit(1)

        self.manifest_path = (self.root / manifest_rel).resolve()
        self.cache_path = (self.root / cache_rel).resolve()

        # Parser type and prompt configuration
        self.parser_type = profile.get("parser", "directory_glob")
        prompt_path_rel = profile.get(
            "prompt_path",
            "plugins/rlm-factory/resources/prompts/rlm/rlm_summarize_legacy.md"
        )
        self.prompt_full_path = (self.root / prompt_path_rel).resolve()
        self.prompt_template = self._load_prompt()

        # File collection patterns (populated from manifest)
        self.include_patterns: List[str] = []
        self.exclude_patterns: List[str] = []
        self.recursive: bool = True
        self._load_manifest_content()

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _load_profiles_json(self) -> dict:
        """
        Load the raw rlm_profiles.json data from disk.

        Returns:
            Parsed JSON as a dict, or empty dict on failure.
        """
        if not DEFAULT_PROFILES_PATH.exists():
            return {}
        try:
            with open(DEFAULT_PROFILES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading profiles JSON: {e}")
            return {}

    def _load_prompt(self) -> str:
        """
        Load the prompt template from the configured path, with a fallback.

        Returns:
            Prompt template string, or empty string if not found.
        """
        if self.prompt_full_path.exists():
            return self.prompt_full_path.read_text(encoding="utf-8")

        # Fallback: check relative to the skill's resources directory
        internal_fallback = (
            Path(__file__).resolve().parents[2]
            / "resources" / "prompts" / "rlm" / "rlm_summarize_legacy.md"
        )
        if internal_fallback.exists():
            return internal_fallback.read_text(encoding="utf-8")

        return ""

    def _load_manifest_content(self) -> None:
        """
        Populate include/exclude patterns from the manifest JSON file.
        Logs a warning if the manifest does not exist.
        """
        if not self.manifest_path.exists():
            print(f"⚠️ Manifest not found: {self.manifest_path}")
            return
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.include_patterns = data.get("include", [])
            self.exclude_patterns = data.get("exclude", [])
            self.recursive = data.get("recursive", True)
        except Exception as e:
            print(f"⚠️ Error reading manifest {self.manifest_path}: {e}")

    @classmethod
    def from_profile(cls, profile_name: str, project_root: Optional[Path] = None) -> "RLMConfig":
        """
        Standard factory entry point for profile-based config.

        Args:
            profile_name: Name of the profile to load.
            project_root: Optional override for the project root path.

        Returns:
            Initialized RLMConfig instance.
        """
        return cls(profile_name, project_root)

=======
from pathlib import Path

# ============================================================
# CONSTANTS & PROMPTS
# ============================================================

current_dir = Path(__file__).parent.resolve()

def find_project_root(start_path: Path) -> Path:
    """Heuristic to find the project root containing .agent or tools/."""
    curr = start_path
    for _ in range(5):
        if (curr / ".agent").exists() or (curr / "tools").exists():
            return curr
        if curr.parent == curr:
            break
        curr = curr.parent
    # Fallback: plugins/name/scripts/ (depth 3) or tools/name/ (depth 2)
    return start_path.parents[2] if "plugins" in str(start_path) else start_path.parents[1]

PROJECT_ROOT = find_project_root(current_dir)

def find_factory_index(script_dir: Path) -> Path:
    """Find manifest-index.json in local or parent resources/."""
    # Option 1: script_dir/resources/ (Flat structure like tools/name/)
    opt1 = script_dir / "resources" / "manifest-index.json"
    if opt1.exists():
        return opt1
    # Option 2: script_dir/../resources/ (Plugin structure like plugins/name/scripts/)
    opt2 = script_dir.parent / "resources" / "manifest-index.json"
    if opt2.exists():
        return opt2
    # Default to current_dir / resources (will error gracefully if not found)
    return opt1

FACTORY_INDEX_PATH = find_factory_index(current_dir)

class RLMConfig:
    def __init__(self, run_type="project", override_targets=None):
        self.type = run_type
        self.manifest_data = {}
        self.cache_path = None
        self.parser_type = "directory_glob"
        self.prompt_template = "" # Loaded dynamically
        self.targets = []
        self.exclude_patterns = []
        self.allowed_suffixes = []
        
        # Load Factory Index
        if not FACTORY_INDEX_PATH.exists():
            print(f"❌ Factory Index not found at {FACTORY_INDEX_PATH}")
            sys.exit(1)
            
        try:
            with open(FACTORY_INDEX_PATH, "r") as f:
                factory_index = json.load(f)
        except Exception as e:
            print(f"❌ Invalid Factory Index JSON: {e}")
            sys.exit(1)
            
        # Resolve Config Definition
        config_def = factory_index.get(run_type)

        if not config_def:
            print(f"❌ Unknown RLM Type: '{run_type}'. Available: {list(factory_index.keys())}")
            sys.exit(1)

        self.description = config_def.get("description", "RLM Configuration")
        self.allowed_suffixes = config_def.get("allowed_suffixes", [".md", ".txt"])
            
        # Resolve Paths
        # Manifest is relative to Project Root (Standardized)
        manifest_path_raw = config_def["manifest"]
        self.manifest_path = (PROJECT_ROOT / manifest_path_raw).resolve()
        
        # Cache Path Resolution (Env Overrides Manifest)
        # Dedicated to General Cache for rlm-factory plugin
        env_prefix = config_def.get("env_prefix", "RLM_SUMMARY")
        env_cache_path = os.getenv(f"{env_prefix}_CACHE")
        
        if env_cache_path:
             # If absolute, use as is. If relative, resolve from Project Root.
             self.cache_path = Path(env_cache_path)
             if not self.cache_path.is_absolute():
                   self.cache_path = PROJECT_ROOT / env_cache_path
        else:
             # Fallback to manifest default
             cache_path_raw = config_def["cache"]
             self.cache_path = PROJECT_ROOT / cache_path_raw
            
        self.parser_type = config_def.get("parser", "directory_glob")
        
        # Load LLM Model (Env > Manifest > Default)
        self.llm_model = os.getenv("OLLAMA_MODEL") or config_def.get("llm_model") or "granite3.2:8b"
        
        # Load Prompt from Path (Relative to Project Root)
        prompt_rel_path = config_def.get("prompt_path")
        if prompt_rel_path:
            # Resolve relative to Project Root
            prompt_full_path = (PROJECT_ROOT / prompt_rel_path).resolve()
            if prompt_full_path.exists():
                try:
                    self.prompt_template = prompt_full_path.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"⚠️  Error reading prompt file {prompt_full_path}: {e}")
            else:
                print(f"⚠️  Prompt file not found: {prompt_full_path}")
        else:
            print("⚠️  No 'prompt_path' defined in configuration.")
            
        # Fallback if loading failed
        if not self.prompt_template:
            print(f"❌ Critical Error: Failed to load prompt template for type '{run_type}'.")
            sys.exit(1)
        
        # Load the actual Manifest
        self.load_manifest_content()
        
        if override_targets:
            self.targets = override_targets
            
    def load_manifest_content(self):
        if not self.manifest_path.exists():
            print(f"⚠️  Manifest not found: {self.manifest_path}")
            return

        try:
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
                
            if self.parser_type == "directory_glob":
                self.targets = data.get("include", [])
                self.exclude_patterns = data.get("exclude", [])
            elif self.parser_type == "inventory_dict":
                 # For inventory, we treat the keys or specific fields as targets
                 # Assuming tool_inventory.json structure: {"category": {"tool_name": { "path": "..." }}}
                 # We will extract valid paths from it in collect_files
                 self.manifest_data = data
                 self.targets = ["INVENTORY_ROOT"] # Dummy target to trigger collection
                 self.exclude_patterns = data.get("exclude", []) # Should be empty or relevant
                 
        except Exception as e:
            print(f"⚠️  Error reading manifest {self.manifest_path}: {e}")
>>>>>>> origin/main

# ============================================================
# SHARED UTILITIES
# ============================================================

<<<<<<< HEAD
# ----------------------------------------------------------
# compute_hash — content fingerprinting for cache freshness
# ----------------------------------------------------------
def compute_hash(content: str) -> str:
    """
    Compute a short SHA256 hash of the file content.

    Args:
        content: Raw file content string.

    Returns:
        First 16 characters of the SHA256 hex digest.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ----------------------------------------------------------
# load_cache / save_cache — crash-resilient JSON persistence
# ----------------------------------------------------------
def load_cache(cache_path: Path) -> Dict:
    """
    Load the cache JSON from disk.

    Args:
        cache_path: Path to the cache JSON file.

    Returns:
        Parsed cache dict, or empty dict if the file doesn't exist or is invalid.
    """
=======
import hashlib
from typing import List, Dict

def compute_hash(content: str) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def load_cache(cache_path: Path) -> Dict:
    """Load existing cache or return empty dict."""
>>>>>>> origin/main
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
<<<<<<< HEAD
            print(f"⚠️ Error loading cache at {cache_path}: {e}")
    return {}


def save_cache(cache: Dict, cache_path: Path) -> None:
    """
    Persist the cache dict to disk, creating parent directories as needed.

    Args:
        cache: Cache data to serialize.
        cache_path: Target path for the JSON file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")


# ----------------------------------------------------------
# should_skip — file exclusion predicate
# ----------------------------------------------------------
def should_skip(file_path: Path, config: RLMConfig) -> bool:
    """
    Determine whether a file should be excluded from processing.

    Checks both the exclude pattern list (substring match on absolute path)
    and the allowed suffix list.

    Args:
        file_path: Absolute path to the file.
        config: Active RLMConfig instance.

    Returns:
        True if the file should be skipped, False otherwise.
    """
    path_str = str(file_path.resolve())

    # 1. Exclude patterns (substring match)
    for pattern in config.exclude_patterns:
        if pattern in path_str:
            return True

    # 2. Extension allowlist
    if config.allowed_suffixes:
        if file_path.suffix.lower() not in config.allowed_suffixes:
            return True

    return False


# ----------------------------------------------------------
# collect_files — manifest-driven file discovery
# ----------------------------------------------------------
def collect_files(config: RLMConfig) -> List[Path]:
    """
    Collect eligible files from the project root based on manifest include patterns.

    Supports glob wildcards (e.g. `plugins/*/README.md`) and directory targets.
    Files are deduplicated and returned in sorted order.

    Args:
        config: Active RLMConfig instance providing include patterns and filters.

    Returns:
        Sorted, deduplicated list of absolute Path objects.
    """
    all_files: List[Path] = []
    root = config.root

    for pattern in config.include_patterns:
        pattern_norm = pattern.replace("\\", "/")

        if "*" in pattern_norm or "?" in pattern_norm:
            # Glob pattern — resolve from root
            matches = list(root.glob(pattern_norm))
        else:
            path = root / pattern_norm
            if not path.exists():
                continue
            if path.is_file():
                matches = [path]
            else:
                # Directory target — scan for all allowed extensions
                matches = []
                for ext in config.allowed_suffixes:
                    if config.recursive:
                        matches.extend(path.glob(f"**/*{ext}"))
                    else:
                        matches.extend(path.glob(f"*{ext}"))

        for match in matches:
            if match.is_file() and not should_skip(match, config):
                all_files.append(match)

    return sorted(list(set(all_files)))
=======
            print(f"⚠️  Error loading cache: {e}")
    return {}

def save_cache(cache: Dict, cache_path: Path):
    """Persist cache to disk immediately (crash-resilient)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

def should_skip(file_path: Path, config: RLMConfig, debug_fn=None) -> bool:
    """Check if file should be excluded from processing."""
    
    def log(msg):
        if debug_fn:
            debug_fn(msg)
            
    try:
        # Canonicalize path for consistent matching
        path_obj = file_path.resolve()
        path_str = str(path_obj)
    except Exception as e:
        print(f"⚠️  SKIP CHECK FAILED (path resolution): {file_path} — {e}")
        return False  # fail-open (do not skip if we can't check)

    # Check exclude patterns
    for pattern in config.exclude_patterns:
        if pattern in path_str:
            log(f"Skipping {path_str} (exclude pattern: {pattern})")
            return True
    
    # Allowed Suffixes Check
    if config.allowed_suffixes:
        if file_path.suffix.lower() not in config.allowed_suffixes:
            log(f"Skipping due to unsupported suffix: {file_path.suffix}")
            return True
    
    return False

def collect_files(config: RLMConfig) -> List[Path]:
    """Collect all eligible files based on parser type."""
    all_files = []
    
    if config.parser_type == "inventory_dict":
        # Recursively search for "path" keys in the JSON inventory
        def recursive_search(data):
            if isinstance(data, dict):
                # Check if this node looks like a tool definition
                if "path" in data and isinstance(data["path"], str):
                    path_str = data["path"]
                    # If relative, resolve from project root
                    if not os.path.isabs(path_str):
                        full_path = (PROJECT_ROOT / path_str).resolve()
                    else:
                        full_path = Path(path_str)
                    
                    if full_path.exists():
                        if full_path.is_file():
                            if not should_skip(full_path, config):
                                all_files.append(full_path)
                        elif full_path.is_dir():
                            for ext in config.allowed_suffixes:
                                for f in full_path.glob(f"**/*{ext}"):
                                    if f.is_file() and not should_skip(f, config):
                                        all_files.append(f)
                
                # Recurse into values
                for v in data.values():
                    recursive_search(v)
            
            elif isinstance(data, list):
                # Recurse into list items
                for item in data:
                    recursive_search(item)

        recursive_search(config.manifest_data)
        
        # Post-process: Filter by targets if overridden
        if config.targets and "INVENTORY_ROOT" not in config.targets:
             filtered_files = []
             for f in all_files:
                  for t in config.targets:
                       # Normalize
                       t_path = (PROJECT_ROOT / t).resolve()
                       # Check if file is same or inside target dir
                       try:
                           # is_relative_to is Python 3.9+
                           # f.relative_to(t_path)
                           # But explicit check for parents is safer across versions
                           if f == t_path or t_path in f.parents:
                               filtered_files.append(f)
                               break
                       except:
                           pass
             all_files = filtered_files
                    
    elif config.parser_type == "directory_glob":
        # Default Directory Globbing
        for target in config.targets:
            # Normalize path separators for cross-platform compatibility
            target_normalized = target.replace("\\", "/")
            path = PROJECT_ROOT / target_normalized
            
            if not path.exists():
                continue
                
            # If target is file
            if path.is_file():
                if not should_skip(path, config):
                    all_files.append(path)
                continue
                
            # If target is dir
            for ext in config.allowed_suffixes:
                for f in path.glob(f"**/*{ext}"):
                    if f.is_file() and not should_skip(f, config):
                        all_files.append(f)
                        
    return all_files
            

>>>>>>> origin/main
