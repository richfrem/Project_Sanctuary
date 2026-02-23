#!/usr/bin/env python3
"""
ingest.py (CLI)
=====================================

Purpose:
<<<<<<< HEAD
    Command-line interface for the Vector DB ingestion pipeline.
    Parses the project manifest and feeds documentation/code into the Vector backend.

Workflow:
    1. Resolve Project Root.
    2. Load Config from JSON Profile (VectorConfig).
    3. Initialize VectorDBOperations.
    4. Execute Ingestion (Since time or Full reset).
"""

import sys
import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from langchain_core.documents import Document

# Project paths
# File is at: plugins/vector-db/skills/vector-db-agent/scripts/ingest.py
# Root is 6 levels up (0: scripts, 1: agent, 2: skills, 3: vector-db, 4: plugins, 5: ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vector_config import VectorConfig
from operations import VectorDBOperations

# Try to import RLM for code context injection if available
try:
    # This might be in a different plugin or legacy path
    from rlm_config import RLMConfig
    HAS_RLM = True
except ImportError:
    HAS_RLM = False

# Code shim for advanced parsing
try:
    import ingest_code_shim as code_shim
    HAS_CODE_SHIM = True
except ImportError:
    HAS_CODE_SHIM = False


def main():
    parser = argparse.ArgumentParser(description="Ingest documentation into Vector DB")
    parser.add_argument("--profile", type=str, help="Vector DB profile to use (e.g., knowledge)")
    parser.add_argument("--full", action="store_true", help="Force full re-indexing (wipes database)")
    parser.add_argument("--since", type=int, help="Only ingest files modified in last N hours")
    parser.add_argument("--file", type=str, help="Ingest a specific file relative to root")
    parser.add_argument("--folder", type=str, help="Ingest a specific folder relative to root")
    
    args = parser.parse_args()
    
    # 1. Load configuration from JSON profile (no .env)
    vec_config = VectorConfig(profile_name=args.profile, project_root=str(PROJECT_ROOT))
    manifest = vec_config.load_manifest()
    
    # 2. Initialize operations module with profile config
    cortex = VectorDBOperations(
        str(PROJECT_ROOT),
        child_collection=vec_config.child_collection,
        parent_collection=vec_config.parent_collection,
        chroma_host=vec_config.chroma_host,
        chroma_port=vec_config.chroma_port,
        chroma_data_path=vec_config.chroma_data_path
    )
    
    if args.full:
        print("💥 Wipe and Re-index requested.")
        cortex.purge()
        target_files = manifest.get_files()
    elif args.file:
        target_files = [args.file]
    elif args.folder:
        target_files = manifest.get_files_in_folder(args.folder)
    else:
        # Incremental by since hours
        if args.since:
            cutoff = datetime.now() - timedelta(hours=args.since)
            target_files = manifest.get_files_modified_since(cutoff)
            print(f"🕒 Incremental ingest: Checking files modified since {cutoff.strftime('%Y-%m-%d %H:%M')}")
        else:
            # Default to checking everything but only updating if file hash changed
            target_files = manifest.get_files()
            print("🔄 Smart Sync: Checking all files for changes...")

    if not target_files:
        print("✅ No files found to ingest.")
        return

    print(f"🚀 Processing {len(target_files)} files...")
    
    stats = {"success": 0, "failed": 0, "skipped": 0, "chunks": 0}
    
    for i, rel_path in enumerate(target_files, 1):
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            stats["skipped"] += 1
            continue
            
        try:
            # Try to use code shim for structured parsing if applicable
            if HAS_CODE_SHIM and full_path.suffix.lower() in ['.py', '.js', '.ts', '.tsx', '.xml', '.sql']:
                content = code_shim.convert_code_file(full_path)
                if not content:
                    content = full_path.read_text(encoding='utf-8', errors='replace')
            else:
                content = full_path.read_text(encoding='utf-8', errors='replace')
            
            # Simple metadata
            metadata = {
                "source": rel_path,
                "type": full_path.suffix.lstrip('.'),
                "last_modified": os.path.getmtime(full_path)
            }
            
            # Ingest via Core
            doc = Document(page_content=content, metadata=metadata)
            res = cortex.ingest_documents([doc])
            
            stats["success"] += 1
            stats["chunks"] += res.get("chunks", 0)
            
            if i % 50 == 0:
                print(f"   ... Progress: {i}/{len(target_files)} (Chunks: {stats['chunks']})")
                
        except Exception as e:
            print(f"❌ Error ingesting {rel_path}: {e}")
            stats["failed"] += 1

    print(f"\n✨ Ingestion Finished:")
    print(f"   - Success: {stats['success']}")
    print(f"   - Failed:  {stats['failed']}")
    print(f"   - Skipped: {stats['skipped']}")
    print(f"   - Chunks:  {stats['chunks']}")
=======
    Vector Ingestion: Chunks code/docs and generates embeddings via ChromaDB.

Layer: Curate / Vector

Usage Examples:
    python plugins/vector-db/scripts/ingest.py --help

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
    rlm_config = RLMConfig(run_type="legacy")
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
    DEFAULT_DIRS = manifest.get("include", ["legacy-system"])
    EXCLUDE_PATTERNS = manifest.get("exclude", [])
    print(f"📋 Loaded configuration from manifest ({len(DEFAULT_DIRS)} paths)")
else:
    # Fallback Defaults
    print("⚠️  Manifest not found, using fallback defaults.")
    DEFAULT_DIRS = ["legacy-system"]
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


class VectorDBManager:
    """Simplified Vector DB manager for Project project."""
    
    def __init__(self):
        """Initialize ChromaDB with persistent local storage."""
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Use persistent client (local file-based)
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = os.getenv("VECTOR_DB_COLLECTION", "project_forms_v1")
        
        # Initialize embeddings (HuggingFace for local, no API needed)
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Fast, good quality
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter for chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    
    def purge(self):
        """Purge existing collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"🗑️  Purged collection: {self.collection_name}")
        except Exception:
            pass  # Collection doesn't exist
        
        # Recreate
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    
    def ingest_documents(self, documents: List[Document]) -> int:
        """Ingest documents into vector store."""
        if not documents:
            return 0
        
        # Split into chunks
        chunks = self.splitter.split_documents(documents)
        
        # Add to vectorstore in batches
        batch_size = 5000
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            total_added += len(batch)
            print(f"   Added batch {i//batch_size + 1}: {len(batch)} chunks")
        
        return total_added
    
    def query(self, query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        results = self.vectorstore.similarity_search_with_score(query_text, k=max_results)
        
        formatted = []
        for doc, score in results:
            formatted.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": score,
                "has_context": doc.metadata.get("has_rlm_context", False)
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
    stale_count = 0
    for rel_path, ids in id_to_source.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            stale_ids.extend(ids)
            stale_count += 1
    
    if not stale_ids:
        print("   ✅ No stale entries found.")
        return 0
    
    print(f"   Found {stale_count} missing files ({len(stale_ids)} chunks)")
    
    # Delete in batches
    batch_size = 5000
    for i in range(0, len(stale_ids), batch_size):
        batch = stale_ids[i:i + batch_size]
        collection.delete(ids=batch)
    
    print(f"   ✅ Removed {len(stale_ids)} stale chunks")
    return len(stale_ids)

def main():
    parser = argparse.ArgumentParser(description="Project Vector DB Ingestion")
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
>>>>>>> origin/main


if __name__ == "__main__":
    main()
