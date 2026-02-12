#!/usr/bin/env python3
"""
rlm_config.py (Shared Module)
=====================================

Purpose:
    Centralized configuration and utility logic for the RLM Toolchain.
    Implements the "Manifest Factory" pattern (ADR-0024) to dynamically
    resolve manifests, cache files, and vector DB configs based on the 
    Analysis Type (Sanctuary vs Tool).

    This module is the Single Source of Truth for RLM logic.

Layer: Codify / Rlm

Usage Examples:
    from tools.codify.rlm.rlm_config import RLMConfig
    config = RLMConfig(run_type="tool")

Supported Object Types:
    - RLM Config (Sanctuary Documentation)
    - RLM Config (Tool Discovery)

Input Files:
    - tools/standalone/rlm-factory/manifest-index.json
    - (Manifests referenced by the index)

Output:
    - RLMConfig object (Typed configuration)
    - Shared Utility Pointers (load_cache, save_cache, etc.)

Key Classes:
    - RLMConfig: Loads and validates configuration from the factory index.

Key Functions:
    - load_cache(): Shared cache loader.
    - save_cache(): Shared cache persister.
    - collect_files(): Centralized file discovery logic (Glob vs Inventory).

Script Dependencies:
    - None (this is a dependency for others)

Consumed by:
    - tools/codify/rlm/distiller.py
    - tools/retrieve/rlm/query_cache.py
    - tools/curate/rlm/cleanup_cache.py
"""
import os
import sys
import json
from pathlib import Path

# ============================================================
# CONSTANTS & PROMPTS
# ============================================================

# Heuristic to find project root (fallback)
current_dir = Path(__file__).parent.resolve()
# tools/codify/rlm -> tools/codify -> tools -> root
PROJECT_ROOT = current_dir.parents[2] 
FACTORY_INDEX_PATH = PROJECT_ROOT / "tools" / "standalone" / "rlm-factory" / "manifest-index.json"

class RLMConfig:
    def __init__(self, run_type="tool", override_targets=None):
        self.type = run_type
        self.manifest_data = {}
        self.cache_path = None
        self.parser_type = "directory_glob"
        self.prompt_template = "" # Loaded dynamically
        self.targets = []
        self.exclude_patterns = []
        
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
            
        config_def = factory_index.get(run_type)
        if not config_def:
            print(f"❌ Unknown RLM Type: '{run_type}'. Available: {list(factory_index.keys())}")
            sys.exit(1)

        self.description = config_def.get("description", "RLM Configuration")
            
        # Resolve Paths
        # Manifest is relative to Project Root (Standardized)
        manifest_path_raw = config_def["manifest"]
        self.manifest_path = (PROJECT_ROOT / manifest_path_raw).resolve()
        
        # Cache is relative to Project Root
        cache_path_raw = config_def["cache"]
        self.cache_path = PROJECT_ROOT / cache_path_raw
            
        self.parser_type = config_def.get("parser", "directory_glob")
        
        # Load LLM Model (Manifest > Env > Default)
        self.llm_model = config_def.get("llm_model") or os.getenv("OLLAMA_MODEL", "granite3.2:8b")
        
        # Load Vector DB Configuration (New Schema)
        self.vector_config = config_def.get("vector_config", {})
        
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
                self.targets = []
                
                # 1. RLM Manifest Schema v2.0 (target_directories + core_files)
                if "target_directories" in data:
                    t_dirs = data.get("target_directories", [])
                    for d in t_dirs:
                        if isinstance(d, dict) and "path" in d:
                            self.targets.append(d["path"])
                        elif isinstance(d, str):
                            self.targets.append(d)
                            
                if "core_files" in data:
                    c_files = data.get("core_files", [])
                    for f in c_files:
                        if isinstance(f, dict) and "path" in f:
                            self.targets.append(f["path"])
                        elif isinstance(f, str):
                            self.targets.append(f)
                            
                # 2. ADR 097 Simple Schema (files array)
                if "files" in data:
                    files = data.get("files", [])
                    for item in files:
                        if isinstance(item, str):
                            self.targets.append(item)
                        elif isinstance(item, dict) and "path" in item:
                            self.targets.append(item["path"])

                # 3. Original Simple Schema (include array)
                if "include" in data:
                     self.targets.extend(data.get("include", []))

                # 4. Fallback for Classic Schema (core + topic) (Legacy ADR 089)
                if not self.targets and ("core" in data or "topic" in data):
                    self.targets = data.get("core", []) + data.get("topic", [])
                
                self.exclude_patterns = data.get("exclude_patterns", data.get("exclude", []))

                # Fallback: Inherit Global Exclusions from Ingest Manifest if missing
                if not self.exclude_patterns:
                    # Try local exclusion manifest first (new standard)
                    excl_path = PROJECT_ROOT / "mcp_servers" / "lib" / "exclusion_manifest.json"
                    if excl_path.exists():
                         try:
                            with open(excl_path, "r") as f:
                                excl_data = json.load(f)
                                self.exclude_patterns = excl_data.get("global_exclusions", [])
                         except Exception:
                             pass
                    
                    # Fallback to older location
                    if not self.exclude_patterns:
                        ingest_path = PROJECT_ROOT / "tools" / "standalone" / "vector-db" / "ingest_manifest.json"
                        if ingest_path.exists():
                            try:
                                with open(ingest_path, "r") as f:
                                    ingest_data = json.load(f)
                                    self.exclude_patterns = ingest_data.get("exclude", [])
                            except Exception as e:
                                print(f"⚠️  Failed to load global exclusions: {e}")

            elif self.parser_type == "inventory_dict":
                 # For inventory, we treat the keys or specific fields as targets
                 # Assuming tool_inventory.json structure: {"category": {"tool_name": { "path": "..." }}}
                 # We will extract valid paths from it in collect_files
                 self.manifest_data = data
                 self.targets = ["INVENTORY_ROOT"] # Dummy target to trigger collection
                 self.exclude_patterns = data.get("exclude", []) # Should be empty or relevant
                 
        except Exception as e:
            print(f"⚠️  Error reading manifest {self.manifest_path}: {e}")

# ============================================================
# SHARED UTILITIES
# ============================================================

import hashlib
from typing import List, Dict

def compute_hash(content: str) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def load_cache(cache_path: Path) -> Dict:
    """Load existing cache or return empty dict."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
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
    
    # For Tools, allow .py, .js, .sh, .ts
    if config.type == "tool":
        if file_path.suffix.lower() not in [".py", ".js", ".sh", ".ts"]:
            log(f"Skipping due to unsupported suffix: {file_path.suffix}")
            return True
    elif config.type == "sanctuary":
        # Allow documentation and whitelisted code/config files
        if file_path.suffix.lower() not in [".md", ".txt", ".py", ".json", ".mmd"]:
             log(f"Skipping due to unsupported suffix: {file_path.suffix}")
             return True
    else:
        # Fallback for strict documentation profiles
        if file_path.suffix.lower() not in [".md", ".txt"]:
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
                            for ext in ["**/*.py", "**/*.js", "**/*.sh", "**/*.ts", "**/*.md", "**/*.txt"]:
                                for f in full_path.glob(ext):
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
            for ext in ["**/*.md", "**/*.txt", "**/*.py", "**/*.js", "**/*.ts"]:
                for f in path.glob(ext):
                    if f.is_file() and not should_skip(f, config):
                        all_files.append(f)
                        
    return all_files
            

