#!/usr/bin/env python3
"""
RAG Cortex Validator
=====================================

    Input validation for Mnemonic Cortex RAG operations.
    Ensures structural integrity of requests before processing.

Layer: Validation (Logic)

Key Classes:
    - CortexValidator: Main safety logic
        - __init__(project_root)
        - validate_ingest_full(purge, dirs)
        - validate_query(query, max_results)
        - validate_ingest_incremental(files, meta)
        - validate_capture_snapshot(manifest, type)
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any


class ValidationError(Exception):
    #============================================
    # Class: ValidationError
    # Purpose: Custom exception raised when validation fails.
    #============================================
    pass


class CortexValidator:
    #============================================
    # Class: CortexValidator
    # Purpose: Validator for Cortex MCP operations.
    # Patterns: Strategy / Validator
    #============================================

    def __init__(self, project_root: str):
        #============================================
        # Method: __init__
        # Purpose: Initialize validator with project context.
        # Args:
        #   project_root: Absolute path to project root
        #============================================
        self.project_root = Path(project_root)
    
    def validate_ingest_full(
        self,
        purge_existing: bool = True,
        source_directories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        #============================================
        # Method: validate_ingest_full
        # Purpose: Validate full ingestion request parameters.
        # Args:
        #   purge_existing: Whether to purge existing database
        #   source_directories: Optional list of source directories
        # Returns: Dictionary of validated parameters
        # Raises: ValidationError if path verification fails
        #============================================
        # Validate source directories if provided
        if source_directories:
            for directory in source_directories:
                dir_path = self.project_root / directory
                if not dir_path.exists():
                    raise ValidationError(f"Source directory does not exist: {directory}")
                if not dir_path.is_dir():
                    raise ValidationError(f"Path is not a directory: {directory}")
        
        return {
            "purge_existing": purge_existing,
            "source_directories": source_directories
        }
    
    def validate_query(
        self,
        query: str,
        max_results: int = 5,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        #============================================
        # Method: validate_query
        # Purpose: Validate query request parameters.
        # Args:
        #   query: Search query string
        #   max_results: Maximum results to return (1-100)
        #   use_cache: Cache activation flag
        # Returns: Dictionary of validated parameters
        # Raises: ValidationError if constraints are violated
        #============================================
        # Validate query string
        if not query or not query.strip():
            raise ValidationError("Query string cannot be empty")
        
        if len(query) > 10000:
            raise ValidationError("Query string too long (max 10000 characters)")
        
        # Validate max_results
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")
        
        if max_results > 100:
            raise ValidationError("max_results cannot exceed 100")
        
        return {
            "query": query.strip(),
            "max_results": max_results,
            "use_cache": use_cache
        }
    
    def validate_ingest_incremental(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        #============================================
        # Method: validate_ingest_incremental
        # Purpose: Validate incremental ingestion parameters.
        # Args:
        #   file_paths: List of file paths to ingest
        #   metadata: Optional metadata to attach
        #   skip_duplicates: Deduplication flag
        # Returns: Dictionary of validated parameters
        # Raises: ValidationError if files are missing or unsupported
        #============================================
        # Validate file_paths
        if not file_paths:
            raise ValidationError("file_paths cannot be empty")
        
        if len(file_paths) > 1000:
            raise ValidationError("Cannot ingest more than 1000 files at once")
        
        # Validate each file path
        validated_paths = []
        for file_path in file_paths:
            # Convert to absolute path if relative
            if not os.path.isabs(file_path):
                abs_path = self.project_root / file_path
            else:
                abs_path = Path(file_path)
            
            # Check file exists
            if not abs_path.exists():
                raise ValidationError(f"File does not exist: {file_path}")
            
            # Check it's a file
            if not abs_path.is_file():
                raise ValidationError(f"Path is not a file: {file_path}")
            
            # Check extension (md or allowed code files)
            valid_extensions = ('.md', '.py', '.js', '.jsx', '.ts', '.tsx')
            if not str(abs_path).lower().endswith(valid_extensions):
                raise ValidationError(f"File type not supported: {file_path}. Must be one of {valid_extensions}")
            
            validated_paths.append(str(abs_path))
        
        # Validate metadata if provided
        if metadata:
            if not isinstance(metadata, dict):
                raise ValidationError("metadata must be a dictionary")
        
        return {
            "file_paths": validated_paths,
            "metadata": metadata,
            "skip_duplicates": skip_duplicates
        }
    
    def validate_stats(self) -> Dict[str, Any]:
        #============================================
        # Method: validate_stats
        # Purpose: Validate statistics request (no parameters needed).
        # Returns: Empty dict
        #============================================
        return {}

    def validate_capture_snapshot(
        self,
        manifest_files: List[str],
        snapshot_type: str = "audit",
        strategic_context: Optional[str] = None
    ) -> Dict[str, Any]:
        #============================================
        # Method: validate_capture_snapshot
        # Purpose: Validate tool-driven snapshot parameters.
        # Args:
        #   manifest_files: List of file paths to include
        #   snapshot_type: 'audit' or 'seal'
        #   strategic_context: Optional context string
        # Returns: Dictionary of validated parameters
        # Raises: ValidationError if constraints are violated
        #============================================
        if manifest_files is None:
             manifest_files = []
        
        if not isinstance(manifest_files, list):
            raise ValidationError("manifest_files must be a list of strings")
            
        if snapshot_type not in ["audit", "seal"]:
            raise ValidationError("snapshot_type must be either 'audit' or 'seal'")
            
        if strategic_context and not isinstance(strategic_context, str):
            raise ValidationError("strategic_context must be a string")
            
        return {
            "manifest_files": manifest_files,
            "snapshot_type": snapshot_type,
            "strategic_context": strategic_context
        }
