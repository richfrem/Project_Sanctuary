"""
Simple File-Based Document Store

Replacement for LangChain's removed LocalFileStore.
Stores documents as JSON files in a directory.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

# Configure logging
logger = logging.getLogger("rag_cortex.file_store")


class SimpleFileStore:
    """Simple file-based key-value store for documents."""
    
    def __init__(self, root_path: str):
        """Initialize the file store.
        
        Args:
            root_path: Directory path to store files
        """
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
    
    def mset(self, key_value_pairs: List[Tuple[str, Document]]) -> None:
        """Set multiple key-value pairs.
        
        Args:
            key_value_pairs: List of (key, document) tuples
        """
        for key, doc in key_value_pairs:
            file_path = self.root_path / f"{key}.json"
            
            # Serialize document
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)
    
    def mget(self, keys: List[str]) -> List[Document]:
        """Get multiple values by keys.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            List of documents (None for missing keys)
        """
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
    
    def mdelete(self, keys: List[str]) -> None:
        """Delete multiple keys.
        
        Args:
            keys: List of keys to delete
        """
        for key in keys:
            file_path = self.root_path / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
    
    def yield_keys(self) -> Iterator[str]:
        """Yield all keys in the store.
        
        Yields:
            Key strings (filenames without .json extension)
        """
        for file_path in self.root_path.glob("*.json"):
            yield file_path.stem
