#============================================
# mcp_servers/rag_cortex/file_store.py
# Purpose: Simple File-Based Document Store.
#          Replacement for LangChain's removed LocalFileStore.
# Role: Single Source of Truth
# Used as a module by operations.py
# Calling example:
#   store = SimpleFileStore(root_path="./data")
#   store.mset([("key1", doc1)])
#   docs = store.mget(["key1"])
# LIST OF FUNCTIONS IMPLEMENTED:
#   - __init__
#   - mdelete
#   - mget
#   - mset
#   - yield_keys
#============================================

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from mcp_servers.lib.logging_utils import setup_mcp_logging

# Configure logging
logger = setup_mcp_logging("rag_cortex.file_store")


#============================================
# Class: SimpleFileStore
# Purpose: Simple file-based key-value store for documents.
# Storage: JSON files in a directory.
# Components:
#   root_path: Directory path to store files
#============================================
class SimpleFileStore:
    
    #============================================
    # Method: __init__
    # Purpose: Initialize the file store.
    # Args:
    #   root_path: Directory path to store files
    #============================================
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
    
    #============================================
    # Method: mset
    # Purpose: Set multiple key-value pairs.
    # Args:
    #   key_value_pairs: List of (key, document) tuples
    #============================================
    def mset(self, key_value_pairs: List[Tuple[str, Document]]) -> None:
        for key, doc in key_value_pairs:
            file_path = self.root_path / f"{key}.json"
            
            # Serialize document
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)
    
    #============================================
    # Method: mget
    # Purpose: Get multiple values by keys.
    # Args:
    #   keys: List of keys to retrieve
    # Returns: List of documents (None for missing keys)
    #============================================
    def mget(self, keys: List[str]) -> List[Document]:
        results = []
        for key in keys:
            file_path = self.root_path / f"{key}.json"
            
            if not file_path.exists():
                results.append(None)
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_dict = json.load(f)
                
                doc = Document(
                    page_content=doc_dict["page_content"],
                    metadata=doc_dict.get("metadata", {})
                )
                results.append(doc)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                # In case of error, just skip this file
                continue
                    
        return results
    
    #============================================
    # Method: mdelete
    # Purpose: Delete multiple keys.
    # Args:
    #   keys: List of keys to delete
    #============================================
    def mdelete(self, keys: List[str]) -> None:
        for key in keys:
            file_path = self.root_path / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
    
    #============================================
    # Method: yield_keys
    # Purpose: Yield all keys in the store.
    # Yields: Key strings (filenames without .json extension)
    #============================================
    def yield_keys(self) -> Iterator[str]:
        for file_path in self.root_path.glob("*.json"):
            yield file_path.stem
