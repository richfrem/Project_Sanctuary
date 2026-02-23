#!/usr/bin/env python3
"""
RAG Cortex Operations
=====================================

Purpose:
    Core domain logic for the Mnemonic Cortex (RAG) system.
    Orchestrates ingestion, semantic search, and Mnemonic Cache (CAG) management.
    Manages vector store interactions and LLM context building.

Layer: Business Logic

Key Classes:
    - CortexOperations: Main manager
        - __init__(project_root, client)
        - ingest_incremental(file_paths, metadata, skip_duplicates)
        - query(query, max_results, use_cache, reasoning_mode)
        - query_structured(query_string, request_id)
        - get_stats(include_samples, sample_count)
        - cache_get(query)
        - cache_set(query, answer)
        - cache_warmup(genesis_queries)
        - get_cache_stats()
        
        # Internal Methods
        - _chunked_iterable(seq, size)
        - _safe_add_documents(retriever, docs, max_retries)
        - _load_manifest_registry()
        - _get_output_to_manifest_map(registry)
        - _dedupe_manifest(manifest)
        - _check_mermaid_cli()
        - _render_single_diagram(mmd_path)
        - _ensure_diagrams_rendered()
        - _get_mcp_name(mcp_class_str)

Dependencies:
    - ChromaDB: Vector Store
    - LangChain: Text Splitting & Embedding
    - HuggingFace: Embedding Models
"""


import os
import re # Added for parsing markdown headers
from typing import List, Tuple # Added Tuple
# Disable tqdm globally to prevent stdout pollution - MUST BE FIRST
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import subprocess
import contextlib
import io
import logging
import json
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Optional



# Setup logging
# This block is moved to the top and modified to use standard logging
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from mcp_servers.lib.logging_utils import setup_mcp_logging
# logger = setup_mcp_logging(__name__)

# Configure logging
logger = logging.getLogger("rag_cortex.operations")
if not logger.handlers:
    # Add a default handler if none exist (e.g., when running directly)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


from .models import (
    IngestFullResponse,
    QueryResponse,
    QueryResult,
    StatsResponse,
    CollectionStats,
    IngestIncrementalResponse,
    to_dict,
    CacheGetResponse,
    CacheSetResponse,

)
from mcp_servers.lib.content_processor import ContentProcessor

# Imports that were previously inside methods, now moved to top for class initialization
# Silence stdout/stderr during imports to prevent MCP protocol pollution
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import chromadb
    from dotenv import load_dotenv
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from mcp_servers.rag_cortex.file_store import SimpleFileStore
    from langchain_core.documents import Document
    from mcp_servers.lib.env_helper import get_env_variable


class CortexOperations:
    #============================================
    # Class: CortexOperations
    # Purpose: Main backend for the Mnemonic Cortex RAG service.
    # Patterns: Facade / Orchestrator
    #============================================
    
    def __init__(self, project_root: str, client: Optional[chromadb.ClientAPI] = None):
        #============================================
        # Method: __init__
        # Purpose: Initialize Mnemonic Cortex backend.
        # Args:
        #   project_root: Path to project root
        #   client: Optional injected ChromaDB client
        #============================================
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "mcp_servers" / "rag_cortex" / "scripts"
        self.data_dir = self.project_root / ".agent" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Network configuration using env_helper
        self.chroma_host = get_env_variable("CHROMA_HOST", required=False) or "localhost"
        self.chroma_port = int(get_env_variable("CHROMA_PORT", required=False) or "8110")
        self.chroma_data_path = get_env_variable("CHROMA_DATA_PATH", required=False) or ".vector_data"
        
        self.child_collection_name = get_env_variable("CHROMA_CHILD_COLLECTION", required=False) or "child_chunks_v5"
        self.parent_collection_name = get_env_variable("CHROMA_PARENT_STORE", required=False) or "parent_documents_v5"

        # Initialize ChromaDB client
        if client:
            self.chroma_client = client
        else:
            self.chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        
        # Initialize embedding model (HuggingFace/sentence-transformers for ARM64 compatibility - ADR 069)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize child splitter (smaller chunks for retrieval)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize parent splitter (larger chunks for context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize vectorstore (Chroma)
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.child_collection_name,
            embedding_function=self.embedding_model
        )

        # Parent document store (file-based, using configurable data path)
        docstore_path = str(self.project_root / self.chroma_data_path / self.parent_collection_name)
        self.store = SimpleFileStore(root_path=docstore_path)

        # Initialize Content Processor
        self.processor = ContentProcessor(self.project_root)
    
    #============================================
    # Method: _chunked_iterable
    # Purpose: Yield successive n-sized chunks from seq.
    # Args:
    #   seq: Sequence to chunk
    #   size: Chunk size
    # Returns: Generator of chunks
    #============================================
    def _chunked_iterable(self, seq: List, size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]
    
    def _safe_add_documents(self, retriever, docs: List, max_retries: int = 5):
        #============================================
        # Method: _safe_add_documents
        # Purpose: Recursively retry adding documents to handle ChromaDB 
        #          batch size limits.
        # Args:
        #   retriever: ParentDocumentRetriever instance
        #   docs: List of documents to add
        #   max_retries: Maximum number of retry attempts
        #============================================
        try:
            retriever.add_documents(docs, ids=None, add_to_docstore=True)
            return
        except Exception as e:
            # Check for batch size or internal errors
            err_text = str(e).lower()
            if "batch size" not in err_text and "internalerror" not in e.__class__.__name__.lower():
                raise
            
            if len(docs) <= 1 or max_retries <= 0:
                raise
            
            mid = len(docs) // 2
            left = docs[:mid]
            right = docs[mid:]
            self._safe_add_documents(retriever, left, max_retries - 1)
            self._safe_add_documents(retriever, right, max_retries - 1)



    #============================================
    # Methods: _should_skip_path and _load_documents_from_directory
    # DEPRECATED: Replaced by ContentProcessor.load_for_rag()
    #============================================

    def ingest_full(
        self,
        purge_existing: bool = True,
        source_directories: List[str] = None
    ):
        #============================================
        # Method: ingest_full
        # Purpose: Perform full ingestion of knowledge base.
        # Args:
        #   purge_existing: Whether to purge existing database
        #   source_directories: Optional list of source directories
        # Returns: IngestFullResponse with accurate statistics
        #============================================
        try:
            start_time = time.time()
            
            # Purge existing collections if requested
            if purge_existing:
                logger.info("Purging existing database collections...")
                try:
                    self.chroma_client.delete_collection(name=self.child_collection_name)
                    logger.info(f"Deleted child collection: {self.child_collection_name}")
                except Exception as e:
                    logger.warning(f"Child collection '{self.child_collection_name}' not found or error deleting: {e}")
                
                # Also clear the parent document store
                if Path(self.store.root_path).exists():
                    import shutil
                    shutil.rmtree(self.store.root_path)
                    logger.info(f"Cleared parent document store at: {self.store.root_path}")
                else:
                    logger.info(f"Parent document store path '{self.store.root_path}' does not exist, no need to clear.")
                
                # Recreate the directory to ensure it exists for new writes
                Path(self.store.root_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Recreated parent document store directory at: {self.store.root_path}")
                
            # Re-initialize vectorstore to ensure it connects to a fresh/existing collection
            # This is critical after a delete_collection operation
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.child_collection_name,
                embedding_function=self.embedding_model
            )
            
            # Default source directories from Manifest (ADR 082 Harmonization - JSON)
            import json
            manifest_path = self.project_root / "mcp_servers" / "lib" / "ingest_manifest.json"
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                base_dirs = manifest.get("common_content", [])
                unique_targets = manifest.get("unique_rag_content", [])
                default_source_dirs = list(set(base_dirs + unique_targets))
            except Exception as e:
                logger.warning(f"Failed to load ingest manifest from {manifest_path}: {e}")
                # Fallback to critical defaults if manifest fails
                default_source_dirs = ["00_CHRONICLE", "01_PROTOCOLS"]
            
            # Determine directories
            dirs_to_process = source_directories or default_source_dirs
            paths_to_scan = [str(self.project_root / d) for d in dirs_to_process]
            
            # Load documents using ContentProcessor
            logger.info(f"Loading documents via ContentProcessor from {len(paths_to_scan)} directories...")
            all_docs = list(self.processor.load_for_rag(paths_to_scan))
            
            total_docs = len(all_docs)
            if total_docs == 0:
                logger.warning("No documents found for ingestion.")
                return IngestFullResponse(
                    documents_processed=0,
                    chunks_created=0,
                    ingestion_time_ms=(time.time() - start_time) * 1000,
                    vectorstore_path=f"{self.chroma_host}:{self.chroma_port}",
                    status="success",
                    error="No documents found."
                )
            
            logger.info(f"Processing {len(all_docs)} documents with parent-child splitting...")
            
            child_docs = []
            parent_count = 0
            
            for doc in all_docs:
                # Split into parent chunks
                parent_chunks = self.parent_splitter.split_documents([doc])
                
                for parent_chunk in parent_chunks:
                    # Generate parent ID
                    parent_id = str(uuid4())
                    parent_count += 1
                    
                    # Store parent document
                    self.store.mset([(parent_id, parent_chunk)])
                    
                    # Split parent into child chunks
                    sub_docs = self.child_splitter.split_documents([parent_chunk])
                    
                    # Add parent_id to child metadata
                    for sub_doc in sub_docs:
                        sub_doc.metadata["parent_id"] = parent_id
                        child_docs.append(sub_doc)
            
            # Add child chunks to vectorstore in batches
            # ChromaDB has a maximum batch size of ~5461
            logger.info(f"Adding {len(child_docs)} child chunks to vectorstore...")
            batch_size = 5000  # Safe batch size under the limit
            
            for i in range(0, len(child_docs), batch_size):
                batch = child_docs[i:i + batch_size]
                logger.info(f"  Adding batch {i//batch_size + 1}/{(len(child_docs)-1)//batch_size + 1} ({len(batch)} chunks)...")
                self.vectorstore.add_documents(batch)
            
            # Get actual counts
            # Re-initialize vectorstore to ensure it reflects the latest state
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.child_collection_name,
                embedding_function=self.embedding_model
            )
            child_count = self.vectorstore._collection.count()
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"✓ Ingestion complete!")
            logger.info(f"  - Parent documents: {parent_count}")
            logger.info(f"  - Child chunks: {child_count}")
            logger.info(f"  - Time: {elapsed_ms/1000:.2f}s")
            
            return IngestFullResponse(
                documents_processed=total_docs,
                chunks_created=child_count,
                ingestion_time_ms=elapsed_ms,
                vectorstore_path=f"{self.chroma_host}:{self.chroma_port}",
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Full ingestion failed: {e}", exc_info=True)
            return IngestFullResponse(
                documents_processed=0,
                chunks_created=0,
                ingestion_time_ms=0,
                vectorstore_path="",
                status="error",
                error=str(e)
            )

    
    def query(
        self,
        query: str,
        max_results: int = 5,
        use_cache: bool = False,
        reasoning_mode: bool = False
    ):
        #============================================
        # Method: query
        # Purpose: Perform semantic search query using RAG infrastructure.
        # Args:
        #   query: Search query string
        #   max_results: Maximum results to return
        #   use_cache: Whether to use semantic cache
        #   reasoning_mode: Use reasoning model if True
        # Returns: QueryResponse with results and metadata
        #============================================
        try:
            start_time = time.time()
            
            # Initialize ChromaDB client (already done in __init__)
            collection = self.chroma_client.get_collection(name=self.child_collection_name)
            
            # Initialize embedding model (already done in __init__)
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results with Parent Document lookup
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i, doc_content in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    parent_id = metadata.get("parent_id")
                    
                    # If we have a parent_id, retrieve the full document context
                    final_content = doc_content
                    if parent_id:
                        try:
                            parent_docs = self.store.mget([parent_id])
                            if parent_docs and parent_docs[0]:
                                final_content = parent_docs[0].page_content
                                # Update metadata with parent metadata if needed
                                metadata.update(parent_docs[0].metadata)
                        except Exception as e:
                            logger.warning(f"Failed to retrieve parent doc {parent_id}: {e}")
                    
                    formatted_results.append(QueryResult(
                        content=final_content,
                        metadata=metadata,
                        relevance_score=results['distances'][0][i] if results.get('distances') else None
                    ))
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Query '{query[:50]}...' completed in {elapsed_ms:.2f}ms with {len(formatted_results)} results (Parent-Retriever applied).")
            
            return QueryResponse(
                status="success",
                results=formatted_results,
                query_time_ms=elapsed_ms,
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Query failed for '{query[:50]}...': {e}", exc_info=True)
            return QueryResponse(
                status="error",
                results=[],
                query_time_ms=0,
                cache_hit=False,
                error=str(e)
            )
    
    def get_stats(self, include_samples: bool = False, sample_count: int = 5):
        #============================================
        # Method: get_stats
        # Purpose: Get database statistics and health status.
        # Args:
        #   include_samples: Whether to include sample docs
        #   sample_count: Number of sample documents to return
        # Returns: StatsResponse with detailed database metrics
        #============================================
        try:
            # Get child chunks stats
            child_count = 0
            try:
                collection = self.chroma_client.get_collection(name=self.child_collection_name)
                child_count = collection.count()
                logger.info(f"Child collection '{self.child_collection_name}' count: {child_count}")
            except Exception as e:
                logger.warning(f"Child collection '{self.child_collection_name}' not found or error accessing: {e}")
                pass  # Collection doesn't exist yet
            
            # Get parent documents stats
            parent_count = 0
            if Path(self.store.root_path).exists():
                try:
                    parent_count = sum(1 for _ in self.store.yield_keys())
                    logger.info(f"Parent document store '{self.parent_collection_name}' count: {parent_count}")
                except Exception as e:
                    logger.warning(f"Error accessing parent document store at '{self.store.root_path}': {e}")
                    pass  # Silently ignore errors for MCP compatibility
            else:
                logger.info(f"Parent document store path '{self.store.root_path}' does not exist.")
            
            # Build collections dict
            collections = {
                "child_chunks": CollectionStats(count=child_count, name=self.child_collection_name),
                "parent_documents": CollectionStats(count=parent_count, name=self.parent_collection_name)
            }
            
            # Determine health status
            if child_count > 0 and parent_count > 0:
                health_status = "healthy"
            elif child_count > 0 or parent_count > 0:
                health_status = "degraded"
            else:
                health_status = "error"
            logger.info(f"RAG Cortex health status: {health_status}")
            
            # Retrieve sample documents if requested
            samples = None
            if include_samples and child_count > 0:
                try:
                    collection = self.chroma_client.get_collection(name=self.child_collection_name)
                    # Get sample documents with metadata and content
                    retrieved_docs = collection.get(limit=sample_count, include=["metadatas", "documents"])
                    
                    samples = []
                    for i in range(len(retrieved_docs["ids"])):
                        sample = DocumentSample(
                            id=retrieved_docs["ids"][i],
                            metadata=retrieved_docs["metadatas"][i],
                            content_preview=retrieved_docs["documents"][i][:150] + "..." if len(retrieved_docs["documents"][i]) > 150 else retrieved_docs["documents"][i]
                        )
                        samples.append(sample)
                    logger.info(f"Retrieved {len(samples)} sample documents.")
                except Exception as e:
                    logger.warning(f"Error retrieving sample documents: {e}")
                    # Silently ignore sample retrieval errors
                    pass
            
            return StatsResponse(
                total_documents=parent_count,
                total_chunks=child_count,
                collections=collections,
                health_status=health_status,
                samples=samples
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}", exc_info=True)
            return StatsResponse(
                total_documents=0,
                total_chunks=0,
                collections={},
                health_status="error",
                error=str(e)
            )
    
    def ingest_incremental(
        self,
        file_paths: List[str],
        metadata: Dict[str, Any] = None,
        skip_duplicates: bool = True
    ) -> IngestIncrementalResponse:
        #============================================
        # Method: ingest_incremental
        # Purpose: Incrementally ingest documents without full rebuild.
        # Args:
        #   file_paths: List of file paths to ingest
        #   metadata: Optional metadata to attach
        #   skip_duplicates: Deduplication flag
        # Returns: IngestIncrementalResponse with statistics
        #============================================
        try:
            start_time = time.time()
            
            # Validate files
            valid_files = []
            
            # Known host path prefixes that should be stripped for container compatibility
            # This handles cases where absolute host paths are passed to the containerized service
            HOST_PATH_MARKERS = [
                "/Users/",      # macOS
                "/home/",       # Linux
                "/root/",       # Linux root
                "C:\\Users\\",  # Windows
                "C:/Users/",    # Windows forward slash
            ]
            
            for fp in file_paths:
                path = Path(fp)
                
                # Handle absolute host paths by converting to relative paths
                # This enables proper resolution when running in containers
                if path.is_absolute():
                    fp_str = str(fp)
                    # Check if this looks like a host absolute path (not container /app path)
                    is_host_path = any(fp_str.startswith(marker) for marker in HOST_PATH_MARKERS)
                    
                    if is_host_path:
                        # Try to extract the relative path after common project markers
                        # Look for 'Project_Sanctuary/' or similar project root markers in the path
                        project_markers = ["Project_Sanctuary/", "project_sanctuary/", "/app/"]
                        for marker in project_markers:
                            if marker in fp_str:
                                # Extract the relative path after the project marker
                                relative_part = fp_str.split(marker, 1)[1]
                                path = self.project_root / relative_part
                                logger.info(f"Translated host path to container path: {fp} -> {path}")
                                break
                        else:
                            # No marker found, log warning and try the path as-is
                            logger.warning(f"Could not translate host path: {fp}")
                    # If it starts with /app, it's already a container path - use as-is
                    elif fp_str.startswith("/app"):
                        pass  # path is already correct
                else:
                    # Relative path - prepend project root
                    path = self.project_root / path
                
                if path.exists() and path.is_file():
                    if path.suffix == '.md':
                        valid_files.append(str(path.resolve()))
                    elif path.suffix in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                        valid_files.append(str(path.resolve()))
                else:
                    logger.warning(f"Skipping invalid file path: {fp}")
            
            if not valid_files:
                logger.warning("No valid files to ingest incrementally.")
                return IngestIncrementalResponse(
                    documents_added=0,
                    chunks_created=0,
                    skipped_duplicates=0,
                    ingestion_time_ms=(time.time() - start_time) * 1000,
                    status="success",
                    error="No valid files to ingest"
                )
            
            added_documents_count = 0
            total_child_chunks_created = 0
            skipped_duplicates_count = 0
            
            all_child_docs_to_add = []
            
            # Use ContentProcessor to load valid files
            # Note: ContentProcessor handles code-to-markdown transformation in memory
            # It expects a list of paths (valid_files are already resolved strings)
            try:
                docs_from_processor = list(self.processor.load_for_rag(valid_files))
                
                for doc in docs_from_processor:
                    if metadata:
                        doc.metadata.update(metadata)
                        
                    # Split into parent chunks
                    parent_chunks = self.parent_splitter.split_documents([doc])
                    
                    for parent_chunk in parent_chunks:
                        # Generate parent ID
                        parent_id = str(uuid4())
                        
                        # Store parent document
                        self.store.mset([(parent_id, parent_chunk)])
                        
                        # Split parent into child chunks
                        sub_docs = self.child_splitter.split_documents([parent_chunk])
                        
                        # Add parent_id to child metadata
                        for sub_doc in sub_docs:
                            sub_doc.metadata["parent_id"] = parent_id
                            all_child_docs_to_add.append(sub_doc)
                            total_child_chunks_created += 1
                
                added_documents_count = len(docs_from_processor)
                    
            except Exception as e:
                logger.error(f"Error during incremental ingest processing: {e}")
            
            # Add child chunks to vectorstore
            if all_child_docs_to_add:
                logger.info(f"Adding {len(all_child_docs_to_add)} child chunks to vectorstore...")
                batch_size = 5000
                for i in range(0, len(all_child_docs_to_add), batch_size):
                    batch = all_child_docs_to_add[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return IngestIncrementalResponse(
                documents_added=added_documents_count,
                chunks_created=total_child_chunks_created,
                skipped_duplicates=0,
                ingestion_time_ms=elapsed_ms,
                status="success"
            )
            
        except Exception as e:
            return IngestIncrementalResponse(
                documents_added=0,
                chunks_created=0,
                skipped_duplicates=0,
                ingestion_time_ms=0,
                status="error",
                error=str(e)
            )

    # [DISABLED] Synaptic Phase (Dreaming) - See ADR 091 (Rejected for now)
    # def dream(self):
    #     #============================================
    #     # Method: dream
    #     # Purpose: Execute the Synaptic Phase (Dreaming).
    #     #          Consolidate memories and update Opinion Network.
    #     # Reference: ADR 091
    #     #============================================
    #     from .dreaming import Dreamer
    #     
    #     try:
    #         logger.info("Initializing Synaptic Phase (Dreaming)...")
    #         dreamer = Dreamer(self.project_root)
    #         dreamer.dream()
    #         return {"status": "success", "message": "Synaptic Phase complete."}
    #     except Exception as e:
    #         logger.error(f"Dreaming failed: {e}", exc_info=True)
    #         return {"status": "error", "error": str(e)}

    # ========================================================================
    # Cache Operations (Protocol 114 - Guardian Wakeup)
    # ========================================================================

    def cache_get(self, query: str):
        #============================================
        # Method: cache_get
        # Purpose: Retrieve answer from semantic cache.
        # Args:
        #   query: Search query string
        # Returns: CacheGetResponse with hit status and answer
        #============================================
        from .cache import get_cache
        from .models import CacheGetResponse
        import time
        
        try:
            start = time.time()
            cache = get_cache()
            
            # Generate cache key
            structured_query = {"semantic": query, "filters": {}}
            cache_key = cache.generate_key(structured_query)
            
            # Attempt retrieval
            result = cache.get(cache_key)
            query_time_ms = (time.time() - start) * 1000
            
            if result:
                return CacheGetResponse(
                    cache_hit=True,
                    answer=result.get("answer"),
                    query_time_ms=query_time_ms,
                    status="success"
                )
            else:
                return CacheGetResponse(
                    cache_hit=False,
                    answer=None,
                    query_time_ms=query_time_ms,
                    status="success"
                )
        except Exception as e:
            return CacheGetResponse(
                cache_hit=False,
                answer=None,
                query_time_ms=0,
                status="error",
                error=str(e)
            )

    def cache_set(self, query: str, answer: str):
        #============================================
        # Method: cache_set
        # Purpose: Store answer in semantic cache.
        # Args:
        #   query: Cache key string
        #   answer: Response to cache
        # Returns: CacheSetResponse confirmation
        #============================================
        from .cache import get_cache
        from .models import CacheSetResponse
        
        try:
            cache = get_cache()
            structured_query = {"semantic": query, "filters": {}}
            cache_key = cache.generate_key(structured_query)
            
            cache.set(cache_key, {"answer": answer})
            
            return CacheSetResponse(
                cache_key=cache_key,
                stored=True,
                status="success"
            )
        except Exception as e:
            return CacheSetResponse(
                cache_key="",
                stored=False,
                status="error",
                error=str(e)
            )

    def cache_warmup(self, genesis_queries: List[str] = None):
        #============================================
        # Method: cache_warmup
        # Purpose: Pre-populate cache with genesis queries.
        # Args:
        #   genesis_queries: Optional list of queries to cache
        # Returns: CacheWarmupResponse with counts
        #============================================
        from .models import CacheWarmupResponse
        import time
        
        try:
            # Import genesis queries if not provided
            if genesis_queries is None:
                from .genesis_queries import GENESIS_QUERIES
                genesis_queries = GENESIS_QUERIES
            
            start = time.time()
            cache_hits = 0
            cache_misses = 0
            
            for query in genesis_queries:
                # Check if already cached
                cache_response = self.cache_get(query)
                
                if cache_response.cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    # Generate answer and cache it
                    query_response = self.query(query, max_results=3, use_cache=False)
                    if query_response.results:
                        answer = query_response.results[0].content[:1000]
                        self.cache_set(query, answer)
            
            total_time_ms = (time.time() - start) * 1000
            
            return CacheWarmupResponse(
                queries_cached=len(genesis_queries),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                total_time_ms=total_time_ms,
                status="success"
            )
        except Exception as e:
            return CacheWarmupResponse(
                queries_cached=0,
                cache_hits=0,
                cache_misses=0,
                total_time_ms=0,
                status="error",
                error=str(e)
            )

    # ========================================================================
    # Helper: Recency Delta (High-Signal Filter) is implemented below
    # ================================================================================================================================================
    # Helper: Recency Delta (High-Signal Filter)
    # ========================================================================



    #============================================
    # Protocol 130: Manifest Deduplication (ADR 089)
    # Prevents including files already embedded in generated outputs
    #============================================
    
    def _load_manifest_registry(self) -> Dict[str, Any]:
        """
        Load the manifest registry that maps manifests to their generated outputs.
        Location: .agent/learning/manifest_registry.json
        """
        registry_path = self.project_root / ".agent" / "learning" / "manifest_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Protocol 130: Failed to load manifest registry: {e}")
        return {"manifests": {}}
    
    def _get_output_to_manifest_map(self, registry: Dict[str, Any]) -> Dict[str, str]:
        """
        Invert the registry: output_file → source_manifest_path
        """
        output_map = {}
        for manifest_path, info in registry.get("manifests", {}).items():
            output = info.get("output")
            if output:
                output_map[output] = manifest_path
        return output_map
    
    def _dedupe_manifest(self, manifest: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Protocol 130: Remove files from manifest that are already embedded in included outputs.
        
        Args:
            manifest: List of file paths
            
        Returns:
            Tuple of (deduped_manifest, duplicates_found)
            duplicates_found is dict of {file: reason}
        """
        registry = self._load_manifest_registry()
        output_map = self._get_output_to_manifest_map(registry)
        duplicates = {}
        
        # For each file in manifest, check if it's an output of another manifest
        for file in manifest:
            if file in output_map:
                # This file is a generated output. Check if its source files are also included.
                source_manifest_path = self.project_root / output_map[file]
                
                if source_manifest_path.exists():
                    try:
                        with open(source_manifest_path, "r") as f:
                            source_files = json.load(f)
                        
                        # Check each source file - if it's in the manifest, it's a duplicate
                        for source_file in source_files:
                            if source_file in manifest and source_file != file:
                                duplicates[source_file] = f"Already embedded in {file}"
                    except Exception as e:
                        logger.warning(f"Protocol 130: Failed to load source manifest {source_manifest_path}: {e}")
        
        if duplicates:
            logger.info(f"Protocol 130: Found {len(duplicates)} embedded duplicates, removing from manifest")
            for dup, reason in duplicates.items():
                logger.debug(f"  - {dup}: {reason}")
        
        # Remove duplicates
        deduped = [f for f in manifest if f not in duplicates]
        return deduped, duplicates

    #============================================
    # Diagram Rendering (Task #154)
    # Automatically renders .mmd to .png during snapshot
    #============================================

    def _check_mermaid_cli(self) -> bool:
        """Check if mermaid-cli is available (via npx)."""
        try:
            # Check if npx is in path
            subprocess.run(["npx", "--version"], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _render_single_diagram(self, mmd_path: Path) -> bool:
        """Render a single .mmd file to png if outdated."""
        output_path = mmd_path.with_suffix(".png")
        try:
            # Check timestamps
            if output_path.exists() and mmd_path.stat().st_mtime <= output_path.stat().st_mtime:
                return True # Up to date

            logger.info(f"Rendering outdated diagram: {mmd_path.name}")
            result = subprocess.run(
                [
                    "npx", "-y", "@mermaid-js/mermaid-cli",
                    "-i", str(mmd_path),
                    "-o", str(output_path),
                    "-b", "transparent", "-t", "default"
                ],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                logger.warning(f"Failed to render {mmd_path.name}: {result.stderr[:200]}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Error rendering {mmd_path.name}: {e}")
            return False

    def _ensure_diagrams_rendered(self):
        """Scan docs/architecture_diagrams and render any outdated .mmd files."""
        try:
            diagrams_dir = self.project_root / "docs" / "architecture_diagrams"
            if not diagrams_dir.exists():
                return
                
            if not self._check_mermaid_cli():
                logger.warning("mermaid-cli not found (npx missing or failed). Skipping diagram rendering.")
                return

            mmd_files = sorted(diagrams_dir.rglob("*.mmd"))
            logger.info(f"Verifying {len(mmd_files)} architecture diagrams...")
            
            rendered_count = 0
            for mmd_path in mmd_files:
                # We only render if outdated, logic is in _render_single_diagram
                if self._render_single_diagram(mmd_path): 
                   pass 
        except Exception as e:
            logger.warning(f"Diagram rendering process failed: {e}")



    def get_cache_stats(self):
        #============================================
        # Method: get_cache_stats
        # Purpose: Get semantic cache statistics.
        # Returns: Dict with hit/miss counts and entry total
        #============================================
        from .cache import get_cache
        try:
            cache = get_cache()
            return cache.get_stats()
        except Exception as e:
            return {"error": str(e)}
    def query_structured(
        self,
        query_string: str,
        request_id: str = None
    ) -> Dict[str, Any]:
        #============================================
        # Method: query_structured
        # Purpose: Execute Protocol 87 structured query.
        # Args:
        #   query_string: Standardized inquiry format
        #   request_id: Unique request identifier
        # Returns: API response with matches and routing info
        #============================================
        from .structured_query import parse_query_string
        from .mcp_client import MCPClient
        import uuid
        import json
        from datetime import datetime, timezone
        
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            # Parse Protocol 87 query
            query_data = parse_query_string(query_string)
            
            # Extract components
            scope = query_data.get("scope", "cortex:index")
            intent = query_data.get("intent", "RETRIEVE")
            constraints = query_data.get("constraints", "")
            granularity = query_data.get("granularity", "ATOM")
            
            # Route to appropriate MCP
            client = MCPClient(self.project_root)
            results = client.route_query(
                scope=scope,
                intent=intent,
                constraints=constraints,
                query_data=query_data
            )
            
            # Build Protocol 87 response
            response = {
                "request_id": request_id,
                "steward_id": "CORTEX-MCP-01",
                "timestamp_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "query": json.dumps(query_data, separators=(',', ':')),
                "granularity": granularity,
                "matches": [],
                "checksum_chain": [],
                "signature": "cortex.mcp.v1",
                "notes": ""
            }
            
            # Process results from MCP routing
            for result in results:
                if "error" in result:
                    response["notes"] = f"Error from {result.get('source', 'unknown')}: {result['error']}"
                    continue
                
                match = {
                    "source_path": result.get("source_path", "unknown"),
                    "source_mcp": result.get("source", "unknown"),
                    "mcp_tool": result.get("mcp_tool", "unknown"),
                    "content": result.get("content", {}),
                    "sha256": "placeholder_hash"  # TODO: Implement actual hash
                }
                response["matches"].append(match)
            
            # Add routing metadata
            response["routing"] = {
                "scope": scope,
                "routed_to": self._get_mcp_name(scope),
                "orchestrator": "CORTEX-MCP-01",
                "intent": intent
            }
            
            response["notes"] = f"Found {len(response['matches'])} matches. Routed to {response['routing']['routed_to']}."
            
            return response
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "query": query_string
            }
    


    def _get_mcp_name(self, mcp_class_str: str) -> str:
        #============================================
        # Method: _get_mcp_name
        # Purpose: Map scope to corresponding MCP name.
        # Args:
        #   scope: Logical scope from query
        # Returns: MCP identifier string
        #============================================
        mapping = {
            "Protocols": "Protocol MCP",
            "Living_Chronicle": "Chronicle MCP",
            "tasks": "Task MCP",
            "Code": "Code MCP",
            "ADRs": "ADR MCP"
        }
        return mapping.get(scope, "Cortex MCP (Vector DB)")
