#!/usr/bin/env python3
"""
ingest.py (CLI)
=====================================

Purpose:
    Vector Ingestion: Chunks code/docs and generates embeddings via ChromaDB.

Layer: Curate / Vector

Usage Examples:
    python tools/codify/vector/ingest.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    --full          : Full rebuild (purge + ingest all)
    --folder        : Ingest specific folder
    --file          : Ingest specific file
    --since         : Ingest files changed in last N hours (e.g., --since 24)
    --query         : Test query against the database
    --stats         : Show database statistics
    --purge         : Purge database only
    --cleanup       : Remove stale entries for deleted/renamed files
    --no-cleanup    : Skip auto-cleanup on incremental ingests

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - get_chroma_path(): No description.
    - load_manifest(): Load ingestion configuration from manifest file.
    - load_rlm_cache(): Load the RLM summary cache for Super-RAG context injection.
    - should_skip(): Check if file should be skipped.
    - collect_files(): Collect markdown and code files from target directories/files.
    - create_document_with_context(): Create a LangChain Document with code conversion and Super-RAG context.
    - run_cleanup(): Remove stale entries for files that no longer exist.
    - main(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

# ChromaDB and LangChain
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# Helper to get the Chroma DB path (using Native Linux Path to avoid WSL I/O errors)
def get_chroma_path():
    # 1. Check env var first
    env_path = os.getenv("VECTOR_DB_PATH")
    if env_path:
        # Expand user (~) if present
        return Path(os.path.expanduser(env_path)).resolve()
    
    # 2. Fallback to default (~/.agent/learning/chroma_db)
    home = Path(os.path.expanduser("~"))
    db_path = home / ".agent" / "learning" / "chroma_db"
    return db_path

# Project paths
SCRIPT_DIR = Path(__file__).parent
project_root_fallback = SCRIPT_DIR.parent.parent.parent
if str(project_root_fallback) not in sys.path:
    sys.path.append(str(project_root_fallback))

# Use the fallback path (based on __file__) which works correctly in WSL
# The PathResolver may return Windows paths that don't work in WSL
PROJECT_ROOT = project_root_fallback

VECTOR_DB_PATH = get_chroma_path()

# RLM Configuration (Manifest Factory)
try:
    from tools.codify.rlm.rlm_config import RLMConfig
    rlm_config = RLMConfig(run_type="legacy")
    RLM_CACHE_PATH = rlm_config.cache_path
except ImportError:
    # Fallback if module structure differs
    sys.path.append(str(PROJECT_ROOT))
    from tools.codify.rlm.rlm_config import RLMConfig
    rlm_config = RLMConfig(run_type="sanctuary")
    RLM_CACHE_PATH = rlm_config.cache_path

# Manifest Path
MANIFEST_PATH = PROJECT_ROOT / "tools" / "standalone" / "vector-db" / "ingest_manifest.json"

def load_manifest():
    """Load ingestion configuration from manifest file."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error reading manifest: {e}")
    return None

# Load Config
manifest = load_manifest()

if manifest:
    DEFAULT_DIRS = manifest.get("include", ["ADRs"])
    EXCLUDE_PATTERNS = manifest.get("exclude", [])
    print(f"📋 Loaded configuration from manifest ({len(DEFAULT_DIRS)} paths)")
else:
    # Fallback Defaults
    print("⚠️  Manifest not found, using fallback defaults.")
    DEFAULT_DIRS = ["ADRs", "docs", ".agent", "LEARNING"]
    EXCLUDE_PATTERNS = ["/archive/", "/.git/", "/node_modules/"]

# Allow env var override
# RLM_TARGET_DIRS logic removed to enforce manifest-driven configuration.
# Use --folder, --file, or update the manifest.


# Import Code Shim
try:
    sys.path.append(str(Path(__file__).parent))
    from ingest_code_shim import convert_code_file
except ImportError:
    print("⚠️  Could not import ingest_code_shim. Code files will be treated as plain text.")
    def convert_code_file(p): return p.read_text(errors='ignore')

def load_rlm_cache() -> Dict[str, Any]:
    """Load the RLM summary cache for Super-RAG context injection."""
    if RLM_CACHE_PATH.exists():
        try:
            with open(RLM_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load RLM cache: {e}")
    return {}


def should_skip(path: Path) -> bool:
    """Check if file should be skipped."""
    path_str = str(path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def collect_files(targets: List[str], since_hours: Optional[int] = None) -> List[Path]:
    """Collect markdown and code files from target directories/files."""
    files = []
    cutoff_time = None
    if since_hours:
        cutoff_time = datetime.now().timestamp() - (since_hours * 3600)
    
    # Supported extensions
    # XML included for Reports; Forms XML is shielded by EXCLUDE_PATTERNS
    CODE_EXTS = {".xml", ".sql", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".pll", ".fmb"} 
    ALL_EXTS = {".md", ".txt"} | CODE_EXTS

    for target in targets:
        path = PROJECT_ROOT / target
        if not path.exists():
            print(f"⚠️  Path not found: {target}")
            continue
            
        if path.is_file():
            if path.suffix.lower() in ALL_EXTS:
                if cutoff_time and path.stat().st_mtime < cutoff_time:
                    continue  # Skip old files
                files.append(path)
        else:
            # Directory: recursive glob
            for root, _, filenames in os.walk(path):
                for name in filenames:
                    f_path = Path(root) / name
                    if f_path.suffix.lower() in ALL_EXTS and not should_skip(f_path):
                        if cutoff_time and f_path.stat().st_mtime < cutoff_time:
                            continue
                        files.append(f_path)
    
    return list(set(files))  # Dedupe


def create_document_with_context(file_path: Path, rlm_cache: Dict[str, Any]) -> Optional[Document]:
    """Create a LangChain Document with code conversion and Super-RAG context."""
    try:
        # 1. Convert content (Standard MD or Code->MD)
        if file_path.suffix.lower() == ".md":
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        elif file_path.suffix.lower() == ".txt":
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            # Use Shim for code
            content = convert_code_file(file_path)

        if not content or not content.strip(): 
            return None

        rel_path = str(file_path.relative_to(PROJECT_ROOT))
        
        # 2. RLM Context Injection
        rlm_entry = rlm_cache.get(rel_path, {})
        summary = rlm_entry.get("summary", "")
        
        if summary:
            # Prepend context for better semantic matching
            augmented_content = f"[CONTEXT: {summary}]\n\n{content}"
        else:
            augmented_content = content
        
        return Document(
            page_content=augmented_content,
            metadata={
                "source": rel_path,
                "filename": file_path.name,
                "has_rlm_context": bool(summary),
                "file_type": file_path.suffix
            }
        )
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return None


class SimpleFileStore:
    """Simple JSON-based file store for Parent Documents."""
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.root_path.mkdir(parents=True, exist_ok=True)
    
    def mset(self, key_value_pairs: List[tuple]) -> None:
        for key, doc in key_value_pairs:
            file_path = self.root_path / f"{key}.json"
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    def mget(self, keys: List[str]) -> List[Document]:
        results = []
        for key in keys:
            file_path = self.root_path / f"{key}.json"
            if not file_path.exists():
                results.append(None)
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results.append(Document(page_content=data["page_content"], metadata=data.get("metadata", {})))
            except Exception:
                results.append(None)
        return results

    def mdelete(self, keys: List[str]) -> None:
        for key in keys:
            file_path = self.root_path / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
                
    def yield_keys(self):
        for f in self.root_path.glob("*.json"):
            yield f.stem


class VectorDBManager:
    """Parity-Compliant Vector DB Manager (Split-Store Topology)."""
    
    def __init__(self):
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Load Config from RLMConfig (Manifest)
        self.config = getattr(rlm_config, 'vector_config', {})
        
        # 1. Parent Store Config
        parent_cfg = self.config.get("parent_store", {})
        self.parent_path = PROJECT_ROOT / parent_cfg.get("path", ".vector_data/parent_documents_v5")
        self.parent_store = SimpleFileStore(self.parent_path)
        self.parent_chunk_size = parent_cfg.get("chunk_size", 2000)
        self.parent_chunk_overlap = parent_cfg.get("chunk_overlap", 200)

        # 2. Child Store Config
        child_cfg = self.config.get("child_store", {})
        self.child_collection_name = child_cfg.get("collection_name", "child_chunks_v5")
        self.child_chunk_size = child_cfg.get("chunk_size", 400)
        self.child_chunk_overlap = child_cfg.get("chunk_overlap", 50)
        
        print(f"🔧 Config: Parent({self.parent_chunk_size}/{self.parent_chunk_overlap}) -> Child({self.child_chunk_size}/{self.child_chunk_overlap})")
        print(f"📁 Stores: {self.parent_path} (File) -> {self.child_collection_name} (Vector)")
        
        # Chroma Client
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector Store (Child)
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.child_collection_name,
            embedding_function=self.embeddings
        )
    
    def purge(self):
        """Purge both Parent and Child stores."""
        # Purge Child (Vector)
        try:
            self.client.delete_collection(name=self.child_collection_name)
            print(f"🗑️  Purged Vector Collection: {self.child_collection_name}")
        except Exception:
            pass
            
        # Recreate Vector
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.child_collection_name,
            embedding_function=self.embeddings
        )
        
        # Purge Parent (File)
        for f in self.parent_path.glob("*.json"):
            f.unlink()
        print(f"🗑️  Purged Parent Store: {self.parent_path}")

    def ingest_documents(self, documents: List[Document]) -> int:
        """Ingest documents using Split-Store-Split topology."""
        if not documents:
            return 0
            
        total_child_chunks = 0
        child_docs_batch = []
        
        for doc in documents:
            # 1. Split into Parent Chunks
            parent_chunks = self.parent_splitter.split_documents([doc])
            
            for parent_chunk in parent_chunks:
                # Generate Parent ID
                parent_id = str(uuid4())
                
                # Store Parent
                self.parent_store.mset([(parent_id, parent_chunk)])
                
                # 2. Split into Child Chunks
                child_chunks = self.child_splitter.split_documents([parent_chunk])
                
                # Link Child to Parent
                for child in child_chunks:
                    child.metadata["parent_id"] = parent_id
                    child_docs_batch.append(child)
        
        # Batch Add Children to Vector Store
        if child_docs_batch:
            batch_size = 5000
            for i in range(0, len(child_docs_batch), batch_size):
                batch = child_docs_batch[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                total_child_chunks += len(batch)
                print(f"   Added batch {i//batch_size + 1}: {len(batch)} chunks")
                
        return total_child_chunks
    
    def query(self, query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search (Parent-Aware)."""
        results = self.vectorstore.similarity_search_with_score(query_text, k=max_results)
        
        # 1. Collect Parent IDs
        parent_ids = []
        child_docs = []
        scores = []
        
        for doc, score in results:
            pid = doc.metadata.get("parent_id")
            parent_ids.append(pid)
            child_docs.append(doc)
            scores.append(score)
            
        # 2. Fetch Parents
        parents = self.parent_store.mget(parent_ids)
        
        formatted = []
        for i, doc in enumerate(child_docs):
            content = doc.page_content # Default to child
            parent_doc = parents[i]
            
            # Use Parent if available
            if parent_doc:
                content = parent_doc.page_content
                
            formatted.append({
                "content": content,
                "source": doc.metadata.get("source", "unknown"),
                "score": scores[i],
                "has_context": doc.metadata.get("has_rlm_context", False),
                "is_parent": bool(parent_doc)
            })
        
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            count = collection.count()
            return {"collection": self.collection_name, "chunks": count, "status": "healthy"}
        except Exception as e:
            return {"collection": self.collection_name, "chunks": 0, "status": "error", "error": str(e)}


def run_cleanup(manager: VectorDBManager) -> int:
    """Remove stale entries for files that no longer exist."""
    print("🧹 Running cleanup for stale entries...")
    
    try:
        collection = manager.client.get_collection(name=manager.collection_name)
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return 0
    
    total_chunks = collection.count()
    if total_chunks == 0:
        print("   Collection is empty. Nothing to clean.")
        return 0
    
    # Get all documents and check sources
    all_data = collection.get(include=["metadatas"])
    id_to_source = {}
    
    for i, meta in enumerate(all_data['metadatas']):
        source = meta.get('source', '')
        if source:
            doc_id = all_data['ids'][i]
            if source not in id_to_source:
                id_to_source[source] = []
            id_to_source[source].append(doc_id)
    
    # Find stale sources
    stale_ids = []
    stale_parent_ids = set()
    stale_count = 0
    
    for rel_path, ids in id_to_source.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            stale_ids.extend(ids)
            stale_count += 1
            
            # Fetch Metadata to get Parent IDs (Optimized: Get only stale ones)
            # Chroma doesn't support get(ids=...) returning metadata easily mixed with existing query
            # So we fetch specifically for these IDs to get parent_ids
            try:
                stale_data = collection.get(ids=ids, include=["metadatas"])
                for m in stale_data["metadatas"]:
                    if "parent_id" in m:
                        stale_parent_ids.add(m["parent_id"])
            except:
                pass

    if not stale_ids:
        print("   ✅ No stale entries found.")
        return 0
    
    print(f"   Found {stale_count} missing files ({len(stale_ids)} chunks)")
    
    # Delete Vectors (Child)
    batch_size = 5000
    for i in range(0, len(stale_ids), batch_size):
        batch = stale_ids[i:i + batch_size]
        collection.delete(ids=batch)
    
    # Delete Parents (File)
    if stale_parent_ids:
        manager.parent_store.mdelete(list(stale_parent_ids))
        print(f"   🗑️  Removed {len(stale_parent_ids)} parent documents")

    print(f"   ✅ Removed {len(stale_ids)} stale chunks")
    return len(stale_ids)

def main():
    parser = argparse.ArgumentParser(description="Vector DB Ingestion")
    parser.add_argument("--full", action="store_true", help="Full rebuild (purge + ingest all)")
    parser.add_argument("--folder", type=str, help="Ingest specific folder")
    parser.add_argument("--file", type=str, help="Ingest specific file")
    parser.add_argument("--since", type=int, metavar="HOURS", help="Ingest files changed in last N hours (e.g., --since 24)")
    parser.add_argument("--query", type=str, help="Test query against the database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--purge", action="store_true", help="Purge database only")
    parser.add_argument("--cleanup", action="store_true", help="Remove stale entries for deleted/renamed files")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip auto-cleanup on incremental ingests")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = VectorDBManager()
    
    # Handle cleanup command
    if args.cleanup:
        run_cleanup(manager)
        return
    
    # Handle commands
    if args.stats:
        stats = manager.get_stats()
        print(f"\n📊 Vector DB Stats")
        print(f"   Collection: {stats['collection']}")
        print(f"   Chunks: {stats['chunks']}")
        print(f"   Status: {stats['status']}")
        return
    
    if args.purge:
        manager.purge()
        print("✅ Database purged")
        return
    
    if args.query:
        print(f"\n🔍 Querying: {args.query}")
        results = manager.query(args.query)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
            print(f"Source: {r['source']}")
            print(f"Has RLM Context: {r['has_context']}")
            print(r['content'])
        return
    
    # Ingestion modes
    if args.full:
        print("🚀 Full Vector DB Rebuild")
        manager.purge()
        targets = DEFAULT_DIRS
    elif args.folder:
        print(f"📂 Ingesting folder: {args.folder}")
        targets = [args.folder]
    elif args.file:
        print(f"📄 Ingesting file: {args.file}")
        targets = [args.file]
    elif args.since:
        print(f"⏰ Ingesting files changed in last {args.since} hours")
        # Auto-cleanup for incremental ingests (unless --no-cleanup)
        if not args.no_cleanup:
            run_cleanup(manager)
        targets = DEFAULT_DIRS
    else:
        parser.print_help()
        return
    
    # Load RLM cache for Super-RAG
    print("📖 Loading RLM cache for Super-RAG context injection...")
    rlm_cache = load_rlm_cache()
    print(f"   Loaded {len(rlm_cache)} cached summaries")
    
    # Collect files
    print("📁 Collecting files...")
    files = collect_files(targets, since_hours=args.since)
    print(f"   Found {len(files)} files to ingest")
    
    if not files:
        print("⚠️  No files found to ingest")
        return
    
    # BATCH PROCESSING
    BATCH_SIZE = 100
    documents = []
    total_docs = 0
    total_chunks = 0
    start_time = time.time()
    
    print(f"⚡ Ingesting in batches of {BATCH_SIZE}...")
    
    for i, f in enumerate(files, 1):
        doc = create_document_with_context(f, rlm_cache)
        if doc:
            documents.append(doc)
            
        # Check if we should process this batch
        if len(documents) >= BATCH_SIZE or i == len(files):
            added = manager.ingest_documents(documents)
            total_chunks += added
            total_docs += len(documents)
            
            # Progress Log
            msg = f"   [{i}/{len(files)}] Processed {len(documents)} docs -> {added} chunks"
            # Overwrite line if possible, or just print
            sys.stdout.write(f"\r{msg}")
            sys.stdout.flush()
            if i == len(files): print() # Newline at end
            
            documents = [] # Clear memory
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Ingestion Complete!")
    print(f"   Documents: {total_docs}")
    print(f"   Chunks: {total_chunks}")
    print(f"   Time: {elapsed:.2f}s")
    
    # Show stats
    stats = manager.get_stats()
    print(f"\n📊 Final Stats: {stats['chunks']} total chunks in database")


if __name__ == "__main__":
    main()
