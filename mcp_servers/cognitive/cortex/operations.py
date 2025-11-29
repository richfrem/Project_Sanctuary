"""
Cortex MCP Server - Core Operations

Wraps existing Mnemonic Cortex scripts as MCP operations.
"""
import os
import sys
import time
import subprocess
import contextlib
import io
from pathlib import Path
from typing import Dict, Any, List

from .models import (
    IngestFullResponse,
    QueryResponse,
    QueryResult,
    StatsResponse,
    CollectionStats,
    IngestIncrementalResponse,
    to_dict
)


class CortexOperations:
    """Core operations for Cortex MCP server."""
    
    def __init__(self, project_root: str):
        """
        Initialize operations.
        
        Args:
            project_root: Absolute path to project root
        """
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "mnemonic_cortex" / "scripts"
    
    def ingest_full(
        self,
        purge_existing: bool = True,
        source_directories: List[str] = None
    ) -> IngestFullResponse:
        """
        Perform full ingestion of knowledge base.
        
        Wraps: mnemonic_cortex/scripts/ingest.py
        
        Args:
            purge_existing: Whether to purge existing database
            source_directories: Optional list of source directories
            
        Returns:
            IngestFullResponse with statistics
        """
        try:
            # Import and use IngestionService
            sys.path.insert(0, str(self.project_root))
            from mnemonic_cortex.app.services.ingestion_service import IngestionService
            
            service = IngestionService(str(self.project_root))
            result = service.ingest_full(
                purge_existing=purge_existing,
                source_directories=source_directories
            )
            
            if result.get("status") == "error":
                return IngestFullResponse(
                    documents_processed=0,
                    chunks_created=0,
                    ingestion_time_ms=result.get("ingestion_time_ms", 0),
                    vectorstore_path="",
                    status="error",
                    error=result.get("message", "Unknown error")
                )
            
            return IngestFullResponse(
                documents_processed=result.get("documents_processed", 0),
                chunks_created=result.get("chunks_created", 0),
                ingestion_time_ms=result.get("ingestion_time_ms", 0),
                vectorstore_path=result.get("vectorstore_path", ""),
                status="success"
            )
            
        except Exception as e:
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
        
        Uses: mnemonic_cortex RAG infrastructure directly
        
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
            
            # Import RAG services
            sys.path.insert(0, str(self.project_root))
            
            # Cache Check (Phase 3)
            if use_cache:
                try:
                    from mnemonic_cortex.core.cache import get_cache
                    cache = get_cache()
                    # Generate key based on query and parameters
                    cache_key_data = {
                        "query": query,
                        "max_results": max_results,
                        "reasoning_mode": reasoning_mode
                    }
                    cache_key = cache.generate_key(cache_key_data)
                    
                    cached_data = cache.get(cache_key)
                    if cached_data:
                        # Cache Hit
                        elapsed_ms = (time.time() - start_time) * 1000
                        # Reconstruct QueryResult objects from cached data
                        results = []
                        for item in cached_data.get("results", []):
                            results.append(QueryResult(
                                content=item["content"],
                                metadata=item["metadata"],
                                relevance_score=item.get("relevance_score")
                            ))
                            
                        return QueryResponse(
                            results=results,
                            query_time_ms=elapsed_ms,
                            cache_hit=True,
                            status="success"
                        )
                except Exception as e:
                    # Log error but continue with retrieval
                    print(f"[Cortex] Cache read error: {e}")

            # Suppress all stdout/stderr from VectorDBService initialization
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from mnemonic_cortex.app.services.vector_db_service import VectorDBService
                
                # Initialize service
                db_service = VectorDBService()
                retriever = db_service.get_retriever()
            
            # Handle Reasoning Mode
            final_query = query
            reasoning_metadata = {}
            
            if reasoning_mode:
                try:
                    from mnemonic_cortex.app.services.llm_service import LLMService
                    llm_service = LLMService(str(self.project_root))
                    structured = llm_service.generate_structured_query(query)
                    
                    final_query = structured.get("semantic_query", query)
                    reasoning_metadata = {
                        "original_query": query,
                        "structured_query": structured,
                        "reasoning": structured.get("reasoning")
                    }
                    # TODO: Apply filters if VectorDBService supports them in invoke()
                except Exception as e:
                    # Fallback to raw query on LLM error
                    reasoning_metadata = {"error": f"LLM reasoning failed: {str(e)}"}
            
            # Execute query
            docs = retriever.invoke(final_query)
            
            # Limit results
            docs = docs[:max_results]
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Convert to QueryResult objects
            results = []
            results_for_cache = []
            
            for doc in docs:
                # Merge existing metadata with reasoning metadata if present
                meta = doc.metadata.copy()
                if reasoning_metadata:
                    meta["_reasoning"] = str(reasoning_metadata)
                    
                result = QueryResult(
                    content=doc.page_content,
                    metadata=meta,
                    relevance_score=None  # LangChain doesn't provide scores by default
                )
                results.append(result)
                
                # Prepare for cache
                results_for_cache.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "relevance_score": result.relevance_score
                })
            
            # Cache Set (Phase 3)
            if use_cache and results:
                try:
                    cache.set(cache_key, {"results": results_for_cache})
                except Exception as e:
                    print(f"[Cortex] Cache write error: {e}")
            
            return QueryResponse(
                results=results,
                query_time_ms=elapsed_ms,
                cache_hit=False,  # Phase 2 feature
                status="success"
            )
            
        except Exception as e:
            return QueryResponse(
                results=[],
                query_time_ms=0,
                cache_hit=False,
                status="error",
                error=str(e)
            )
    
    def get_stats(self) -> StatsResponse:
        """
        Get database statistics and health status.
        
        Uses: ChromaDB collections directly
        
        Returns:
            StatsResponse with statistics
        """
        try:
            # Import required modules
            sys.path.insert(0, str(self.project_root))
            from langchain_community.vectorstores import Chroma
            from langchain_nomic import NomicEmbeddings
            from dotenv import load_dotenv
            
            # Load environment
            load_dotenv(dotenv_path=self.project_root / ".env")
            
            # Get database paths
            db_path = os.getenv("DB_PATH", "chroma_db")
            chroma_root_env = os.getenv("CHROMA_ROOT", "").strip()
            
            if chroma_root_env:
                chroma_root = Path(chroma_root_env) if Path(chroma_root_env).is_absolute() else (self.project_root / chroma_root_env)
            else:
                chroma_root = self.project_root / "mnemonic_cortex" / db_path
            
            chroma_root = chroma_root.resolve()
            
            child_collection = os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
            parent_collection = os.getenv("CHROMA_PARENT_STORE", "parent_documents_v5")
            
            # Check if database exists
            if not chroma_root.exists():
                return StatsResponse(
                    total_documents=0,
                    total_chunks=0,
                    collections={},
                    health_status="error",
                    error="Database not found"
                )
            
            # Initialize embedding model
            embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
            
            # Get child chunks stats
            child_path = chroma_root / child_collection
            child_count = 0
            if child_path.exists():
                try:
                    child_db = Chroma(
                        persist_directory=str(child_path),
                        embedding_function=embedding_model,
                        collection_name=child_collection
                    )
                    child_count = child_db._collection.count()
                except Exception as e:
                    pass  # Silently ignore errors for MCP compatibility
            
            # Get parent documents stats
            parent_path = chroma_root / parent_collection
            parent_count = 0
            if parent_path.exists():
                try:
                    # Parent documents are stored in LocalFileStore
                    from langchain_classic.storage import LocalFileStore
                    fs_store = LocalFileStore(root_path=str(parent_path))
                    parent_count = sum(1 for _ in fs_store.yield_keys())
                except Exception as e:
                    pass  # Silently ignore errors for MCP compatibility
            
            # Build collections dict
            collections = {
                "child_chunks": CollectionStats(count=child_count, name=child_collection),
                "parent_documents": CollectionStats(count=parent_count, name=parent_collection)
            }
            
            # Determine health status
            if child_count > 0 and parent_count > 0:
                health_status = "healthy"
            elif child_count > 0 or parent_count > 0:
                health_status = "degraded"
            else:
                health_status = "error"
            
            return StatsResponse(
                total_documents=parent_count,
                total_chunks=child_count,
                collections=collections,
                health_status=health_status
            )
            
        except Exception as e:
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
        Incrementally ingest files.
        
        Wraps: mnemonic_cortex/scripts/ingest_incremental.py
        
        Args:
            file_paths: List of file paths to ingest
            metadata: Optional metadata to attach
            skip_duplicates: Whether to skip duplicate files
            
        Returns:
            IngestIncrementalResponse with statistics
        """
        try:
            # Import and use IngestionService
            sys.path.insert(0, str(self.project_root))
            from mnemonic_cortex.app.services.ingestion_service import IngestionService
            
            service = IngestionService(str(self.project_root))
            result = service.ingest_incremental(
                file_paths=file_paths,
                skip_duplicates=skip_duplicates
            )
            
            if result.get("error"):
                return IngestIncrementalResponse(
                    documents_added=0,
                    chunks_created=0,
                    skipped_duplicates=0,
                    status="error",
                    error=result.get("error")
                )
            
            return IngestIncrementalResponse(
                documents_added=result.get("added", 0),
                chunks_created=result.get("total_chunks", 0),
                skipped_duplicates=result.get("skipped", 0),
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
        from mnemonic_cortex.core.cache import get_cache
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
        from mnemonic_cortex.core.cache import get_cache
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
        Pre-populate cache with genesis queries.
        
        Args:
            genesis_queries: List of queries to cache. If None, uses default set.
            
        Returns:
            CacheWarmupResponse with warmup statistics
        """
        from .models import CacheWarmupResponse
        import time
        
        try:
            if genesis_queries is None:
                # Default genesis queries for Guardian
                genesis_queries = [
                    "What is the Anvil Protocol?",
                    "What are the core doctrines of Project Sanctuary?",
                    "How does the Mnemonic Cortex work?",
                    "What is Protocol 87?",
                    "What is Protocol 101?",
                    "What is Protocol 113?",
                    "What is Protocol 114?",
                    "Latest chronicles summary",
                    "Latest protocols summary",
                    "Latest roadmap summary"
                ]
            
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
                    # Note: We use the internal query method, ensuring we don't recurse infinitely
                    # We disable cache usage for the generation step
                    query_response = self.query(query, max_results=3, use_cache=False)
                    if query_response.results:
                        answer = query_response.results[0].content[:1000] # Store reasonable amount
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
        from mnemonic_cortex.core.cache import get_cache
        try:
            cache = get_cache()
            return cache.get_stats()
        except Exception as e:
            return {"error": str(e)}
