import os
import json
import logging
import hashlib
import ast
import re
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple, Set
from datetime import datetime

from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.exclusion_config import (
    EXCLUDE_DIR_NAMES,
    ALWAYS_EXCLUDE_FILES,
    ALLOWED_EXTENSIONS,
    PROTECTED_SEEDS
)
from mcp_servers.rag_cortex.ingest_code_shim import parse_python_to_markdown, parse_javascript_to_markdown

logger = setup_mcp_logging("content_processor")

class ContentProcessor:
    """
    Unified content processing engine for Project Sanctuary.
    Handles file traversal, exclusion logic, code transformation, and format adaptation
    for Forge, RAG, and Soul Persistence consumers.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def should_exclude_path(self, path: Path, in_manifest: bool = False) -> bool:
        """
        Unified exclusion logic implementing Protocol 128 (Manifest Priority Bypass).
        """
        base_name = path.name
        try:
            rel_path = path.relative_to(self.project_root)
            rel_path_str = rel_path.as_posix()
        except ValueError:
            rel_path_str = path.as_posix()
        
        # 0. Protected Seeds (Protocol 128) - Check this first to allow seeds in excluded dirs
        if any(rel_path_str.endswith(p) for p in PROTECTED_SEEDS):
            return False

        # 1. Directory Names (Exact matches for any segment)
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            return True
            
        # 2. File Extensions (only for files)
        if path.is_file() and path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return True
                
        # 3. Globs and Compiled Regex (ALWAYS_EXCLUDE_FILES from config)
        from mcp_servers.lib.exclusion_config import ALWAYS_EXCLUDE_FILES
        for pattern in ALWAYS_EXCLUDE_FILES:
            if isinstance(pattern, str):
                if fnmatch.fnmatch(base_name, pattern):
                    return True
            elif hasattr(pattern, 'match'):
                if pattern.match(rel_path_str) or pattern.match(base_name):
                    return True
                
        return False

    def traverse_directory(self, root_path: Path) -> Generator[Path, None, None]:
        """Recursively yields files that should be processed."""
        if root_path.is_file():
            if not self.should_exclude_path(root_path):
                yield root_path
            return

        for root, dirs, files in os.walk(root_path):
            curr_root = Path(root)
            
            # Filter directories in-place (efficiency)
            # This prevents os.walk from descending into excluded directories
            dirs[:] = [d for d in dirs if not self.should_exclude_path(curr_root / d)]
            
            for f in files:
                file_path = curr_root / f
                if not self.should_exclude_path(file_path):
                    yield file_path

    def transform_to_markdown(self, file_path: Path) -> str:
        """
        Transforms file content to Markdown.
        Uses AST/Regex for code files, passes formatting for others.
        """
        try:
            suffix = file_path.suffix.lower()
            
            md_content = "" # Initialize md_content
            if suffix == '.py':
                md_content = parse_python_to_markdown(str(file_path))
            elif suffix in {'.js', '.jsx', '.ts', '.tsx'}:
                md_content = parse_javascript_to_markdown(file_path)
            else:
                # Default: Read as text and wrap if needed
                # Use utf-8-sig to handle/remove BOM if present
                content = file_path.read_text(encoding='utf-8-sig')
                if suffix == '.md':
                    md_content = content
                else:
                    md_content = f"# File: {file_path.name}\n\n```text\n{content}\n```"
            
            return md_content

        except Exception as e:
            logger.error(f"Error transforming {file_path}: {e}")
            return f"Error reading file: {e}"

    def compute_checksum(self, content: bytes) -> str:
        """Computes SHA256 checksum for integrity verification."""
        return hashlib.sha256(content).hexdigest()

    def to_soul_jsonl(
        self, 
        snapshot_path: Path, 
        valence: float, 
        uncertainty: float,
        model_version: str = "Sanctuary-Qwen2-7B-v1.0-GGUF-Final"
    ) -> Dict[str, Any]:
        """
        ADR 081 Adapter: Converts a snapshot file into a Soul Persistence JSONL record.
        Each seal gets a unique timestamped ID and filename to prevent overwriting.
        """
        try:
            content_bytes = snapshot_path.read_bytes()
            # Use utf-8-sig to strip BOM if it was written or exists
            content_str = content_bytes.decode('utf-8-sig')
            checksum = self.compute_checksum(content_bytes)
            
            # Generate unique timestamp for this seal
            now = datetime.now()
            timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            timestamp_file = now.strftime("%Y%m%d_%H%M%S")
            
            # Construct unique ID with timestamp (prevents overwriting)
            # Format: seal_{timestamp}_{original_name}
            clean_name = snapshot_path.name
            while clean_name.endswith('.md'):
                clean_name = clean_name[:-3]
            snapshot_id = f"seal_{timestamp_file}_{clean_name}"
            
            # Unique lineage filename with timestamp
            lineage_filename = f"seal_{timestamp_file}_{snapshot_path.name}"
            
            record = {
                "id": snapshot_id,
                "sha256": checksum,
                "timestamp": timestamp_iso,
                "model_version": model_version,
                "snapshot_type": "seal",
                "valence": valence,
                "uncertainty": uncertainty,
                "content": content_str,
                "source_file": f"lineage/{lineage_filename}"
            }
            return record
            
        except Exception as e:
            logger.error(f"Failed to create Soul JSONL record: {e}")
            raise

    def generate_manifest_entry(self, soul_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts metadata for the Hugging Face manifest from a full soul record.
        """
        # Exclude the heavy 'content' field
        return {k: v for k, v in soul_record.items() if k != 'content'}

    def load_for_rag(
        self, 
        source_paths: List[str] = None
    ) -> Generator[Any, None, None]:
        """
        RAG Adapter: Yields LangChain-compatible Document objects for ingestion.
        """
        from langchain_core.documents import Document
        
        paths_to_scan = [Path(p) for p in source_paths] if source_paths else [self.project_root]
        
        for start_path in paths_to_scan:
            for file_path in self.traverse_directory(start_path):
                try:
                    # Transform content 
                    content = self.transform_to_markdown(file_path)
                    
                    # Generate Metadata
                    try:
                        rel_path = str(file_path.relative_to(self.project_root))
                    except ValueError:
                        rel_path = str(file_path)
                        
                    metadata = {
                        "source": rel_path,
                        "filename": file_path.name,
                        "extension": file_path.suffix,
                        "last_modified": file_path.stat().st_mtime
                    }
                    
                    yield Document(page_content=content, metadata=metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to load for RAG: {file_path} - {e}")

    def generate_training_instruction(self, filename: str) -> str:
        """
        Generates a tailored instruction based on the document's path and name.
        """
        filename_lower = filename.lower()
        
        # Tier 1: High-specificity documents
        if "rag_strategies_and_doctrine" in filename_lower:
            return f"Provide a comprehensive synthesis of the Mnemonic Cortex's RAG architecture as detailed in the document: `{filename}`"
        if "evolution_plan_phases" in filename_lower:
            return f"Explain the multi-phase evolution plan for the Sanctuary Council as documented in: `{filename}`"
        if "readme_guardian_wakeup" in filename_lower:
            return f"Describe the Guardian's cache-first wakeup protocol (P114) using the information in: `{filename}`"
        
        # Tier 2: Document types by path
        if "/01_protocols/" in filename_lower:
            return f"Articulate the specific rules, purpose, and procedures of the Sanctuary protocol contained within: `{filename}`"
        if "/00_chronicle/entries/" in filename_lower:
            return f"Recount the historical events, decisions, and outcomes from the Sanctuary chronicle entry: `{filename}`"
        if "/tasks/" in filename_lower:
            return f"Summarize the objective, criteria, and status of the operational task described in: `{filename}`"
    
        # Tier 3: Generic fallback
        return f"Synthesize the core concepts, data, and principles contained within the Sanctuary artifact: `{filename}`"

    def to_training_jsonl(self, file_path: Path) -> Optional[Dict[str, str]]:
        """
        Forge Adapter: Converts a file into a training JSONL record.
        """
        try:
            content = self.transform_to_markdown(file_path)
            if not content.strip():
                return None
                
            try:
                rel_path = str(file_path.relative_to(self.project_root))
            except ValueError:
                rel_path = file_path.name

            instruction = self.generate_training_instruction(rel_path)
            
            return {
                "instruction": instruction,
                "input": "",
                "output": content
            }
        except Exception as e:
            logger.warning(f"Failed to convert to training record: {file_path} - {e}")
            return None
