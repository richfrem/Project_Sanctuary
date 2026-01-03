#============================================
# scripts/cortex_cli.py
# Purpose: CLI Orchestrator for the Mnemonic Cortex RAG server.
# Role: Single Source of Truth for Terminal Operations.
# Reference: Protocol 128 (Hardened Learning Loop)
#
# INGESTION EXAMPLES:
#   python3 scripts/cortex_cli.py ingest                    # Full purge & rebuild (Default behavior)
#   python3 scripts/cortex_cli.py ingest --no-purge         # Append to existing Vector DB
#   python3 scripts/cortex_cli.py ingest --dirs "LEARNING"  # Target specific directory ingestion
#   python3 scripts/cortex_cli.py ingest --type incremental --files "path/to/file.md"  # Targeted update
#
# SNAPSHOT EXAMPLES (Protocol 128 Workflow):
#   python3 scripts/cortex_cli.py snapshot --type audit --manifest .agent/learning/red_team/red_team_manifest.json
#   python3 scripts/cortex_cli.py snapshot --type learning_audit --manifest .agent/learning/learning_audit/learning_audit_manifest.json
#   python3 scripts/cortex_cli.py snapshot --type seal --manifest .agent/learning/learning_manifest.json
#   python3 scripts/cortex_cli.py snapshot --type learning_audit --context "Egyptian Labyrinth research"
#
# GUARDIAN WAKEUP (Protocol 128 Bootloader):
#   python3 scripts/cortex_cli.py guardian                     # Standard wakeup
#   python3 scripts/cortex_cli.py guardian --mode TELEMETRY    # Telemetry-focused wakeup
#   python3 scripts/cortex_cli.py guardian --show              # Display digest content after generation
#
# BOOTSTRAP DEBRIEF (Fresh Repo Onboarding):
#   python3 scripts/cortex_cli.py bootstrap-debrief            # Generate onboarding context packet
#
# DIAGNOSTICS & RETRIEVAL:
#   python3 scripts/cortex_cli.py stats                     # View child/parent counts & health
#   python3 scripts/cortex_cli.py query "Protocol 128"      # Semantic search across Mnemonic Cortex
#   python3 scripts/cortex_cli.py debrief --hours 48        # Session diff & recency scan
#   python3 scripts/cortex_cli.py cache-stats               # Check semantic cache (CAG) efficiency
#   python3 scripts/cortex_cli.py cache-warmup              # Pre-populate CAG with genesis queries
#
# SOUL PERSISTENCE (ADR 079 / 081):
#   Incremental (append 1 seal to JSONL + upload MD to lineage/):
#     python3 scripts/cortex_cli.py persist-soul
#     python3 scripts/cortex_cli.py persist-soul --valence 0.8 --snapshot .agent/learning/learning_package_snapshot.md
#
#   Full Sync (regenerate entire JSONL from all files + deploy data/):
#     python3 scripts/cortex_cli.py persist-soul-full
#============================================
import argparse
import sys
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations

def main():
    parser = argparse.ArgumentParser(description="Mnemonic Cortex CLI")
    parser.add_argument("--root", default=".", help="Project root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available operations")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Perform full ingestion")
    ingest_parser.add_argument("--no-purge", action="store_false", dest="purge", help="Skip purging DB")
    ingest_parser.add_argument("--dirs", nargs="+", help="Specific directories to ingest")
    ingest_parser.add_argument("--incremental", action="store_true", help="Incremental ingestion mode")
    ingest_parser.add_argument("--hours", type=int, default=24, help="Hours to look back (for incremental mode)")

    # Command: snapshot
    snapshot_parser = subparsers.add_parser("snapshot", help="Capture a Protocol 128 snapshot")
    snapshot_parser.add_argument("--type", choices=["audit", "learning_audit", "seal"], required=True)
    snapshot_parser.add_argument("--manifest", help="Path to manifest JSON file")
    snapshot_parser.add_argument("--context", help="Strategic context for the snapshot")

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Get RAG health and statistics")
    stats_parser.add_argument("--samples", action="store_true", help="Include sample documents")
    stats_parser.add_argument("--sample-count", type=int, default=5, help="Number of samples to include")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Perform semantic search query")
    query_parser.add_argument("query_text", help="Search query string")
    query_parser.add_argument("--max-results", type=int, default=5, help="Maximum results to return")
    query_parser.add_argument("--use-cache", action="store_true", help="Use semantic cache")

    # Command: debrief
    debrief_parser = subparsers.add_parser("debrief", help="Run learning debrief (Protocol 128)")
    debrief_parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours")
    debrief_parser.add_argument("--output", help="Output file path (default: .agent/learning/learning_debrief.md)")

    # Command: guardian (Protocol 128 Bootloader)
    guardian_parser = subparsers.add_parser("guardian", help="Generate Guardian Boot Digest (Protocol 128)")
    guardian_parser.add_argument("--mode", default="HOLISTIC", choices=["HOLISTIC", "TELEMETRY"], help="Wakeup mode")
    guardian_parser.add_argument("--show", action="store_true", help="Display digest content after generation")
    guardian_parser.add_argument("--manifest", default=".agent/learning/guardian_manifest.json", help="Path to guardian manifest")

    # Command: bootstrap-debrief (Fresh Repo Onboarding)
    bootstrap_parser = subparsers.add_parser("bootstrap-debrief", help="Generate onboarding context packet for fresh repo setup")
    bootstrap_parser.add_argument("--manifest", default=".agent/learning/bootstrap_manifest.json", help="Path to bootstrap manifest")
    bootstrap_parser.add_argument("--output", default=".agent/learning/bootstrap_packet.md", help="Output path for the packet")

    # Command: cache-stats
    subparsers.add_parser("cache-stats", help="Get cache statistics")

    # Command: cache-warmup
    warmup_parser = subparsers.add_parser("cache-warmup", help="Pre-populate cache with genesis queries")
    warmup_parser.add_argument("--queries", nargs="+", help="Custom queries to cache")

    # Command: persist-soul (ADR 079)
    soul_parser = subparsers.add_parser("persist-soul", help="Broadcast snapshot to HF AI Commons")
    soul_parser.add_argument("--snapshot", default=".agent/learning/learning_package_snapshot.md", help="Path to snapshot")
    soul_parser.add_argument("--valence", type=float, default=0.0, help="Moral/emotional charge")
    soul_parser.add_argument("--uncertainty", type=float, default=0.0, help="Logic confidence")
    soul_parser.add_argument("--full-sync", action="store_true", help="Sync entire learning directory")

    # Command: persist-soul-full (ADR 081)
    subparsers.add_parser("persist-soul-full", help="Regenerate full JSONL and deploy to HF (ADR 081)")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize Operations
    ops = CortexOperations(project_root=args.root)

    if args.command == "ingest":
        if args.incremental:
            print(f"🔄 Starting INCREMENTAL ingestion (Last {args.hours}h)...")
            # Find files modified in the last N hours
            import time
            from datetime import timedelta
            
            cutoff_time = time.time() - (args.hours * 3600)
            modified_files = []
            
            # Walk project root to find modified files
            # Exclude known heavy/irrelevant dirs
            exclude_dirs = {'.git', '.vector_data', '__pycache__', 'node_modules', 'venv', 'env', 
                            'dataset_package', 'docs/site', 'training_logs'}
            
            for path in ops.project_root.rglob('*'):
                if path.is_file():
                    # Check exclusions
                    if any(part in exclude_dirs for part in path.parts):
                        continue
                        
                    # Check extension
                    if path.suffix not in ['.md', '.py', '.js', '.ts', '.txt', '.json']:
                        continue
                        
                    # Check mtime
                    if path.stat().st_mtime > cutoff_time:
                        modified_files.append(str(path))
            
            if not modified_files:
                print(f"⚠️ No files modified in the last {args.hours} hours. Skipping ingestion.")
                sys.exit(0)
                
            print(f"📄 Found {len(modified_files)} modified files.")
            res = ops.ingest_incremental(file_paths=modified_files)
            
            if res.status == "success":
                print(f"✅ Success: {res.documents_added} added, {res.chunks_created} chunks in {res.ingestion_time_ms/1000:.2f}s")
            else:
                print(f"❌ Error: {res.error}")
                sys.exit(1)
        
        else:
            # Full Ingestion
            print(f"🔄 Starting full ingestion (Purge: {args.purge})...")
            res = ops.ingest_full(purge_existing=args.purge, source_directories=args.dirs)
            if res.status == "success":
                print(f"✅ Success: {res.documents_processed} docs, {res.chunks_created} chunks in {res.ingestion_time_ms/1000:.2f}s")
            else:
                print(f"❌ Error: {res.error}")
                sys.exit(1)

    elif args.command == "snapshot":
        manifest = []
        if args.manifest:
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                print(f"❌ Manifest file not found: {args.manifest}")
                sys.exit(1)
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📋 Loaded manifest with {len(manifest)} files")
        
        print(f"📸 Capturing {args.type} snapshot...")
        res = ops.capture_snapshot(
            manifest_files=manifest, 
            snapshot_type=args.type,
            strategic_context=args.context
        )
        
        if res.status == "success":
            print(f"✅ Snapshot created at: {res.snapshot_path}")
            print(f"📊 Files: {res.total_files} | Bytes: {res.total_bytes}")
            print(f"🔍 Manifest verified: {res.manifest_verified}")
            print(f"📝 Git context: {res.git_diff_context}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "stats":
        stats = ops.get_stats(include_samples=args.samples, sample_count=args.sample_count)
        print(f"🏥 Health: {stats.health_status}")
        print(f"📚 Documents: {stats.total_documents}")
        print(f"🧩 Chunks: {stats.total_chunks}")
        
        if stats.collections:
            print("\n📊 Collections:")
            for name, coll in stats.collections.items():
                print(f"  - {coll.name}: {coll.count} items")
        
        if stats.samples:
            print(f"\n🔍 Sample Documents:")
            for i, sample in enumerate(stats.samples, 1):
                print(f"\n  {i}. ID: {sample.id}")
                print(f"     Preview: {sample.content_preview[:100]}...")
                if sample.metadata:
                    print(f"     Metadata: {sample.metadata}")
        
        if stats.error:
            print(f"\n❌ Error: {stats.error}")

    elif args.command == "query":
        print(f"🔍 Querying: {args.query_text}")
        res = ops.query(
            query=args.query_text,
            max_results=args.max_results,
            use_cache=args.use_cache
        )
        
        if res.status == "success":
            print(f"✅ Found {len(res.results)} results in {res.query_time_ms:.2f}ms")
            print(f"💾 Cache hit: {res.cache_hit}")
            
            for i, result in enumerate(res.results, 1):
                print(f"\n--- Result {i} (Score: {result.relevance_score:.4f}) ---")
                print(f"Content: {result.content[:300]}...")
                if result.metadata:
                    source = result.metadata.get('source', 'Unknown')
                    print(f"Source: {source}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "debrief":
        print(f"📋 Running learning debrief (lookback: {args.hours}h)...")
        debrief_content = ops.learning_debrief(hours=args.hours)
        
        # Default output path
        output_path = args.output or ".agent/learning/learning_debrief.md"
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(debrief_content)
        
        print(f"✅ Debrief written to: {output_file}")
        print(f"📊 Content length: {len(debrief_content)} characters")

    elif args.command == "guardian":
        print(f"🛡️ Generating Guardian Boot Digest (mode: {args.mode})...")
        
        # Load manifest if exists
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📋 Loaded guardian manifest: {len(manifest)} files")
        else:
            print(f"⚠️  Guardian manifest not found at {args.manifest}. Using defaults.")
        
        response = ops.guardian_wakeup(mode=args.mode)
        
        print(f"   Status: {response.status}")
        print(f"   Digest: {response.digest_path}")
        print(f"   Time: {response.total_time_ms:.2f}ms")
        
        if response.error:
            print(f"❌ Error: {response.error}")
            sys.exit(1)
        
        if args.show and response.digest_path:
            print("\n" + "="*60)
            with open(response.digest_path, 'r') as f:
                print(f.read())
        
        print(f"✅ Guardian Boot Digest generated.")

    elif args.command == "bootstrap-debrief":
        print(f"🏗️  Generating Bootstrap Context Packet...")
        
        # Load manifest
        manifest_path = Path(args.manifest)
        manifest = []
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📋 Loaded bootstrap manifest: {len(manifest)} files")
        else:
            print(f"⚠️  Bootstrap manifest not found at {args.manifest}. Using defaults.")
        
        # Generate snapshot using the manifest
        res = ops.capture_snapshot(
            manifest_files=manifest,
            snapshot_type="seal",
            strategic_context="Fresh repository onboarding context"
        )
        
        if res.status == "success":
            # Copy to output path
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy(res.snapshot_path, output_path)
            
            print(f"✅ Bootstrap packet generated: {output_path}")
            print(f"📊 Files: {res.total_files} | Bytes: {res.total_bytes}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "cache-stats":
        stats = ops.get_cache_stats()
        print(f"💾 Cache Statistics:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {stats}")

    elif args.command == "cache-warmup":
        queries = args.queries or None
        print(f"🔥 Warming up cache...")
        res = ops.cache_warmup(genesis_queries=queries)
        
        if res.status == "success":
            print(f"✅ Cached {res.queries_cached} queries")
            print(f"💾 Cache hits: {res.cache_hits}")
            print(f"❌ Cache misses: {res.cache_misses}")
            print(f"⏱️  Total time: {res.total_time_ms/1000:.2f}s")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "persist-soul":
        from mcp_servers.rag_cortex.models import PersistSoulRequest
        print(f"🌱 Broadcasting soul to Hugging Face AI Commons...")
        print(f"   Snapshot: {args.snapshot}")
        print(f"   Valence: {args.valence} | Uncertainty: {args.uncertainty}")
        print(f"   Full sync: {args.full_sync}")
        
        request = PersistSoulRequest(
            snapshot_path=args.snapshot,
            valence=args.valence,
            uncertainty=args.uncertainty,
            is_full_sync=args.full_sync
        )
        res = ops.persist_soul(request)
        
        if res.status == "success":
            print(f"✅ Soul planted successfully!")
            print(f"🔗 Repository: {res.repo_url}")
            print(f"📄 Snapshot: {res.snapshot_name}")
        elif res.status == "quarantined":
            print(f"🚫 Quarantined: {res.error}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    elif args.command == "persist-soul-full":
        print(f"🧬 Regenerating full Soul JSONL and deploying to HuggingFace...")
        res = ops.persist_soul_full()
        
        if res.status == "success":
            print(f"✅ Full sync complete!")
            print(f"🔗 Repository: {res.repo_url}")
            print(f"📄 Output: {res.snapshot_name}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()

