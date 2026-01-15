#!/usr/bin/env python3
"""
Standalone ChromaDB Ingestion Script for RLM Kit.
Provides both Full and Incremental ingestion capabilities.
"""
import os
import sys
import logging
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Third-party imports (Install: pip install langchain-chroma langchain-huggingface python-dotenv)
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chroma_ingest")

load_dotenv()

class VectorMemory:
    def __init__(self, persist_directory: str = ".vector_data"):
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Client
        self.vector_store = Chroma(
            collection_name="project_memory",
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        
    def ingest_file(self, file_path: Path):
        """Ingest a single file with chunking."""
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to DB
            self.vector_store.add_documents(chunks)
            logger.info(f"✅ Ingested {file_path} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest {file_path}: {e}")

    def ingest_directory(self, directory: Path, extensions: List[str] = [".md", ".txt", ".py"]):
        """Recursively ingest a directory."""
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return
            
        logger.info(f"📂 Scanning {directory}...")
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    self.ingest_file(file_path)

    def query(self, query_text: str, n_results: int = 5):
        """Search the vector database."""
        results = self.vector_store.similarity_search_with_score(query_text, k=n_results)
        
        print(f"\n🔍 Query: '{query_text}'\n")
        for doc, score in results:
            print(f"--- [Score: {score:.4f}] {doc.metadata.get('source', 'Unknown')} ---")
            print(doc.page_content[:400] + "...\n")

    def stats(self):
        """Get collection statistics."""
        try:
            # Direct access to underlying Chroma collection for raw stats
            count = self.vector_store._collection.count()
            print(f"📊 Vector Memory Stats:")
            print(f"   - Collection: {self.vector_store._collection.name}")
            print(f"   - Total Chunks: {count}")
            print(f"   - Location: {self.persist_directory}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Ingest: python chroma_ingest.py ingest <file_or_directory>")
        print("  Query:  python chroma_ingest.py query \"search text\"")
        print("  Stats:  python chroma_ingest.py stats")
        sys.exit(1)
        
    command = sys.argv[1]
    memory = VectorMemory()
    
    if command == "ingest":
        if len(sys.argv) < 3:
             print("Error: Missing target directory/file")
             sys.exit(1)
        target = Path(sys.argv[2])
        if target.is_dir():
            memory.ingest_directory(target)
        else:
            memory.ingest_file(target)
            
    elif command == "query":
        if len(sys.argv) < 3:
             print("Error: Missing query text")
             sys.exit(1)
        q = sys.argv[2]
        memory.query(q)
        
    elif command == "stats":
        memory.stats()
        
    else:
        print(f"Unknown command: {command}")
