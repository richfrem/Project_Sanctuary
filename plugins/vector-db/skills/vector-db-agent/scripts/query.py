#!/usr/bin/env python3
"""
query.py (CLI)
=====================================

Purpose:
<<<<<<< HEAD
    Command-line interface for semantic search over the Vector DB index.
    Outputs results retrieved from the underlying Parent Store via the Child search match.
"""

import sys
import argparse
from pathlib import Path

# Project paths
# File is at: plugins/vector-db/skills/vector-db-agent/scripts/query.py
PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vector_config import VectorConfig
from operations import VectorDBOperations


def main():
    parser = argparse.ArgumentParser(description="Query the Vector DB")
    parser.add_argument("query", type=str, help="The semantic search query string")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of parent documents to return")
    parser.add_argument("--profile", type=str, help="Vector DB profile to use (e.g., knowledge)")
    
    args = parser.parse_args()
    
    # Load configuration from JSON profile (no .env)
    vec_config = VectorConfig(profile_name=args.profile, project_root=str(PROJECT_ROOT))
    
    cortex = VectorDBOperations(
        str(PROJECT_ROOT),
        child_collection=vec_config.child_collection,
        parent_collection=vec_config.parent_collection,
        chroma_host=vec_config.chroma_host,
        chroma_port=vec_config.chroma_port,
        chroma_data_path=vec_config.chroma_data_path
    )
    
    print(f"\n🔍 Searching Vector Index for: '{args.query}'\n")
    results = cortex.query(args.query, max_results=args.limit)
    
    if not results:
        print("⚠️ No matching context found.")
        return
        
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        source = r.get("source", "unknown")
        parent_id = r.get("parent_id_matched", "none")
        content = r.get("content", "")
        
        print(f"\n{'='*60}")
        print(f"🏆 Result {i} (Score: {score:.4f})")
        print(f"📄 Source: {source}")
        print(f"🧩 Parent Chunk: {parent_id}")
        if r.get("has_rlm_context"):
            print(f"🧠 RLM Summary Super-RAG Applied")
        print(f"{'-'*60}")
        
        # Display an excerpt to prevent terminal flooding
        if len(content) > 1000:
            print(content[:1000] + "\n... [TRUNCATED] ...")
        else:
            print(content)
=======
    Vector Search: Semantic search interface for the ChromaDB collection.

Layer: Curate / Vector

Usage Examples:
    python plugins/vector-db/scripts/query.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    query           : Search query text
    --results       : Number of results (default: 5)
    --stats         : Show database statistics
    --json          : Output results as JSON

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - get_chroma_path(): No description.
    - main(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import argparse
import sys
import os
import json
from pathlib import Path

# ChromaDB and LangChain
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VECTOR_DB_PATH = get_chroma_path()
COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION", "project_forms_v1")


class VectorDBQuery:
    """Query interface for Project Vector DB."""
    
    def __init__(self):
        """Initialize ChromaDB connection."""
        if not VECTOR_DB_PATH.exists():
            print(f"❌ Vector DB not found at: {VECTOR_DB_PATH}")
            print("   Run: python plugins/vector-db/scripts/ingest.py --full")
            sys.exit(1)
        
        # Use persistent client
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings
        )
    
    def query(self, query_text: str, n_results: int = 5, output_json: bool = False, silent: bool = False):
        """Search the vector database."""
        results = self.vectorstore.similarity_search_with_score(query_text, k=n_results)
        
        if silent:
            # Return raw results for internal library usage
            return results

        if output_json:
            json_results = []
            for doc, score in results:
                json_results.append({
                    "source": doc.metadata.get('source', 'Unknown'),
                    "content": doc.page_content,
                    "score": float(score), # Convert numpy float if needed
                    "has_rlm": doc.metadata.get('has_rlm_context', False)
                })
            # Print ONLY the JSON string to stdout
            print(json.dumps(json_results))
            return results

        print(f"\n🔍 Query: '{query_text}'")
        print(f"   Results: {len(results)}\n")
        
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            has_context = doc.metadata.get('has_rlm_context', False)
            context_badge = "🧠" if has_context else "📄"
            
            print(f"--- Result {i} [Score: {score:.4f}] {context_badge} ---")
            print(f"Source: {source}")
            
            # Show content preview
            content = doc.page_content
            if content.startswith("[CONTEXT:"):
                # Separate RLM context from content
                parts = content.split("]\n\n", 1)
                if len(parts) == 2:
                    context = parts[0] + "]"
                    body = parts[1][:300] + "..." if len(parts[1]) > 300 else parts[1]
                    print(f"Context: {context}")
                    print(f"Content: {body}")
                else:
                    print(f"Content: {content[:400]}...")
            else:
                print(f"Content: {content[:400]}...")
            print()
        
        return results
    
    def stats(self):
        """Get collection statistics."""
        try:
            collection = self.client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            
            print(f"\n📊 Vector Memory Stats")
            print(f"   Collection: {COLLECTION_NAME}")
            print(f"   Total Chunks: {count}")
            print(f"   Location: {VECTOR_DB_PATH}")
            print(f"   Status: {'✅ Healthy' if count > 0 else '⚠️ Empty'}")
            
            # Sample a few docs to show sources
            if count > 0:
                sample = collection.peek(limit=5)
                if sample and sample.get('metadatas'):
                    print(f"\n   Sample Sources:")
                    for meta in sample['metadatas'][:5]:
                        source = meta.get('source', 'unknown')
                        print(f"      - {source}")
            
            return {"collection": COLLECTION_NAME, "chunks": count, "status": "healthy"}
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {"collection": COLLECTION_NAME, "chunks": 0, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Project Vector DB Query")
    parser.add_argument("query", nargs="?", help="Search query text")
    parser.add_argument("--results", "-n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--json", "-j", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Initialize query interface
    db = VectorDBQuery()
    
    if args.stats:
        db.stats()
    elif args.query:
        db.query(args.query, n_results=args.results, output_json=args.json)
    else:
        parser.print_help()

>>>>>>> origin/main

if __name__ == "__main__":
    main()
