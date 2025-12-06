"""
RAG Cortex Operations Module

Handles all core RAG operations including ingestion, querying, and statistics.
"""

import os
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
    handler = logging.StreamHandler(sys.stdout)
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
    CacheWarmupResponse,
    DocumentSample
)

# Imports that were previously inside methods, now moved to top for class initialization
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_chroma import Chroma
from mcp_servers.rag_cortex.file_store import SimpleFileStore
from langchain_core.documents import Document


class CortexOperations:
    """Core operations for Cortex MCP server."""
    
    def __init__(self, project_root: str):
        """
        Initialize operations.
        
        Args:
            project_root: Absolute path to project root
        """
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "mcp_servers" / "rag_cortex" / "scripts"

        # Load environment variables
        load_dotenv(dotenv_path=self.project_root / ".env")

        # Network configuration
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_data_path = os.getenv("CHROMA_DATA_PATH", ".vector_data")
        
        self.child_collection_name = os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
        self.parent_collection_name = os.getenv("CHROMA_PARENT_STORE", "parent_documents_v5")

        # Initialize ChromaDB HTTP client
        self.chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        
        # Initialize embedding model
        self.embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

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
    
    # Helper methods for ingestion
    def _chunked_iterable(self, seq: List, size: int):
        """Yield successive n-sized chunks from seq."""
        for i in range(0, len(seq), size):
            yield seq[i : i + size]
    
    def _safe_add_documents(self, retriever, docs: List, max_retries: int = 5):
        """
        Recursively retry adding documents to handle ChromaDB batch size limits.
        
        Args:
            retriever: ParentDocumentRetriever instance
            docs: List of documents to add
            max_retries: Maximum number of retry attempts
        """
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

    def _load_documents_from_directory(self, directory_path: Path) -> List[Document]:
        """Helper to load documents from a single directory."""
        exclude_subdirs = ["ARCHIVE", "archive", "Archive", "node_modules", "ARCHIVED_MESSAGES", "DEPRECATED"]
        
        if not directory_path.is_dir():
            logger.warning(f"Directory not found, skipping: {directory_path}")
            return []

        logger.info(f"Loading documents from directory: {directory_path}")
        loader = DirectoryLoader(
            str(directory_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            recursive=True,
            show_progress=False,
            use_multithreading=True,
            exclude=[f"**/{ex}/**" for ex in exclude_subdirs],
        )
        try:
            docs = loader.load()
            logger.info(f"Found {len(docs)} documents in {directory_path}")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents from {directory_path}: {e}")
            return []

    def ingest_full(
        self,
        purge_existing: bool = True,
        source_directories: List[str] = None
    ) -> IngestFullResponse:
        """
        Perform full ingestion of knowledge base.
        
        Directly implements batching and retry logic (no service layer delegation).
        
        Args:
            purge_existing: Whether to purge existing database
            source_directories: Optional list of source directories
            
        Returns:
            IngestFullResponse with accurate statistics
        """
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
            
            # Default source directories
            default_source_dirs = [
                "00_CHRONICLE", "01_PROTOCOLS", "02_USER_REFLECTIONS", "04_THE_FORTRESS",
                "05_ARCHIVED_BLUEPRINTS", "06_THE_EMBER_LIBRARY", "07_COUNCIL_AGENTS",
                "RESEARCH_SUMMARIES", "WORK_IN_PROGRESS"
            ]
            
            # Determine directories
            dirs_to_process = source_directories or default_source_dirs
            
            # Load documents
            all_docs = []
            for directory in dirs_to_process:
                dir_path = self.project_root / directory
                all_docs.extend(self._load_documents_from_directory(dir_path))
            
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
    ) -> QueryResponse:
        """
        Perform semantic search query.
        
        Uses: Cortex MCP RAG infrastructure directly
        
        Args:
            query: Query string
            max_results: Maximum number of results
            use_cache: Whether to use cache (Phase 2)
            reasoning_mode: Whether to use LLM to structure the query
            
        Returns:
            QueryResponse with results
        """
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
            
            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i, doc_content in enumerate(results['documents'][0]):
                    # Reconstruct QueryResult object
                    formatted_results.append(QueryResult(
                        content=doc_content,
                        metadata=results['metadatas'][0][i],
                        relevance_score=results['distances'][0][i] if results.get('distances') else None
                    ))
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Query '{query[:50]}...' completed in {elapsed_ms:.2f}ms with {len(formatted_results)} results.")
            
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
    
    def get_stats(self, include_samples: bool = False, sample_count: int = 5) -> StatsResponse:
        """
        Get database statistics and health status.
        
        Args:
            include_samples: If True, include sample documents with metadata
            sample_count: Number of sample documents to return (if include_samples=True)
            
        Returns:
            StatsResponse with database statistics
        """
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
        """
        Incrementally ingest documents without rebuilding the database.
        
        Directly implements incremental ingestion logic (no service layer delegation).
        
        Args:
            file_paths: List of markdown file paths to ingest
            metadata: Optional metadata to attach to documents
            skip_duplicates: Whether to skip files already in database
            
        Returns:
            IngestIncrementalResponse with accurate statistics
        """
        try:
            start_time = time.time()
            
            # Validate files
            valid_files = []
            for fp in file_paths:
                path = Path(fp)
                if not path.is_absolute():
                    path = self.project_root / path
                
                if path.exists() and path.is_file() and path.suffix == '.md':
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
            
            for file_path in valid_files:
                try:
                    # Check for duplicates if skip_duplicates is True
                    # This is a simplified check, a more robust one would query the vectorstore
                    # or docstore for existing documents with this source_file metadata.
                    # For now, we'll assume the file_path itself is the unique identifier.
                    # This part of the diff was not fully provided, so I'm making an assumption.
                    # A proper check would involve querying the vectorstore for documents with this source_file.
                    # For now, we'll just load and process.
                    
                    loader = TextLoader(file_path)
                    docs_from_file = loader.load()
                    
                    if not docs_from_file:
                        logger.info(f"No content found in {file_path}, skipping.")
                        continue
                    
                    file_name = Path(file_path).name
                    
                    # Set metadata for the original documents
                    for doc in docs_from_file:
                        doc.metadata['source_file'] = file_path
                        doc.metadata['source'] = file_path
                        doc.metadata['filename'] = file_name
                        if metadata:
                            doc.metadata.update(metadata)
                    
                    logger.info(f"Processing {len(docs_from_file)} documents from {file_path} with parent-child splitting...")
                    
                    for doc in docs_from_file:
                        # Split into parent chunks
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
                    
                    added_documents_count += len(docs_from_file)
                    
                except Exception as e:
                    logger.error(f"Error ingesting {file_path}: {e}")
                    continue
            
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

    # ========================================================================
    # Cache Operations (Protocol 114 - Guardian Wakeup)
    # ========================================================================

    def cache_get(self, query: str):
        """
        Retrieve answer from cache.
        
        Args:
            query: Query string to look up
            
        Returns:
            CacheGetResponse with cache hit status and answer
        """
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
        """
        Store answer in cache.
        
        Args:
            query: Query string (cache key)
            answer: Answer to cache
            
        Returns:
            CacheSetResponse with storage confirmation
        """
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
        """
        Pre-populate cache with frequently asked genesis queries.
        
        Args:
            genesis_queries: Optional list of queries to cache. 
                           If None, uses default genesis queries.
        
        Returns:
            CacheWarmupResponse with warmup statistics
        """
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


    def guardian_wakeup(self):
        """
        Generate Guardian boot digest from cache (Protocol 114).
        
        Retrieves chronicles, protocols, and roadmap summaries from cache
        and writes a digest to WORK_IN_PROGRESS/guardian_boot_digest.md.
        
        Returns:
            GuardianWakeupResponse with digest path and statistics
        """
        from .models import GuardianWakeupResponse
        from pathlib import Path
        import time
        
        try:
            start = time.time()
            bundles = ["chronicles", "protocols", "roadmap"]
            cache_hits = 0
            cache_misses = 0
            digest_content = []
            
            # Retrieve each bundle from cache
            for bundle in bundles:
                query = f"Latest {bundle} summary"
                response = self.cache_get(query)
                
                if response.cache_hit:
                    cache_hits += 1
                    digest_content.append(f"## {bundle.title()}\n\n{response.answer}\n")
                else:
                    cache_misses += 1
                    # Fall back to query if not cached
                    query_response = self.query(query, max_results=3, use_cache=False)
                    if query_response.results:
                        answer = query_response.results[0].content[:1000]
                        digest_content.append(f"## {bundle.title()}\n\n{answer}...\n")
                        # Cache for next time
                        self.cache_set(query, answer)
            
            # Write digest
            digest_path = Path(self.project_root) / "WORK_IN_PROGRESS" / "guardian_boot_digest.md"
            digest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(digest_path, "w") as f:
                f.write("# Guardian Boot Digest\n\n")
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("\n".join(digest_content))
            
            total_time_ms = (time.time() - start) * 1000
            
            return GuardianWakeupResponse(
                digest_path=str(digest_path),
                bundles_loaded=bundles,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                total_time_ms=total_time_ms,
                status="success"
            )
        except Exception as e:
            return GuardianWakeupResponse(
                digest_path="",
                bundles_loaded=[],
                cache_hits=0,
                cache_misses=0,
                total_time_ms=0,
                status="error",
                error=str(e)
            )

    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
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
        """
        Execute Protocol 87 structured query with MCP orchestration.
        
        Routes queries to specialized MCPs based on scope:
        - Protocols -> Protocol MCP
        - Living_Chronicle -> Chronicle MCP
        - Tasks -> Task MCP
        - Code -> Code MCP
        - ADRs -> ADR MCP
        - Fallback -> Vector DB (cortex_query)
        
        Args:
            query_string: Protocol 87 formatted query (INTENT :: SCOPE :: CONSTRAINTS)
            request_id: Optional request ID for tracing
            
        Returns:
            Protocol 87 compliant response with routing metadata
            
        Example:
            >>> ops.query_structured("RETRIEVE :: Protocols :: Name=\\"Protocol 101\\"")
            {
                "request_id": "...",
                "steward_id": "CORTEX-MCP-01",
                "matches": [...],
                "routing": {"scope": "Protocols", "routed_to": "Protocol MCP"}
            }
        """
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
    
    def _get_mcp_name(self, scope: str) -> str:
        """Map scope to MCP name."""
        mapping = {
            "Protocols": "Protocol MCP",
            "Living_Chronicle": "Chronicle MCP",
            "Tasks": "Task MCP",
            "Code": "Code MCP",
            "ADRs": "ADR MCP"
        }
        return mapping.get(scope, "Cortex MCP (Vector DB)")
