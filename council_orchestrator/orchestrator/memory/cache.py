# council_orchestrator/orchestrator/memory/cache.py
import os
import json
from pathlib import Path
from datetime import datetime
# VectorDBService import is done lazily inside the method so the orchestrator can
# start even if the mnemonic_cortex package is not available in this environment.

class CacheManager:
    @staticmethod
    def prefill_guardian_start_pack(project_root, logger):
        """Pre-fills the Guardian Start Pack cache from the Mnemonic Cortex."""
        bundles = {
            "chronicles": "00_CHRONICLE/ENTRIES/",
            "protocols": "01_PROTOCOLS/",
        }
        project_root = Path(project_root)
        for bundle_name, prefix in bundles.items():
            logger.info(f"Fetching latest 15 documents from path prefix: {prefix}")
            try:
                # --- CORRECTED LOGIC: Use semantic query instead of invalid metadata filter ---
                # This is more robust and aligns with the purpose of a vector DB.
                query_text = f"Retrieve the most recent and relevant documents from the directory {prefix}"
                # Lazy import so orchestrator can start even if mnemonic_cortex isn't installed here
                try:
                    from ...mnemonic_cortex.app.services.vector_db_service import VectorDBService
                except Exception:
                    try:
                        from mnemonic_cortex.app.services.vector_db_service import VectorDBService
                    except Exception as e:
                        logger.error(f"[CACHE] VectorDBService import failed: {e}")
                        # Skip caching for this bundle when the DB service isn't available
                        continue

                # Use the DB service semantic query interface
                db_service = VectorDBService()
                retrieved_docs = db_service.query(query_text)

                # Filter by source prefix (if metadata provides 'source') and limit to 15
                docs_to_cache = [
                    {"page_content": getattr(doc, 'page_content', ''), "metadata": getattr(doc, 'metadata', {})}
                    for doc in retrieved_docs
                    if isinstance(getattr(doc, 'metadata', {}), dict) and
                       str(getattr(doc, 'metadata', {}).get('source', '')).startswith(str(project_root / prefix))
                ][:15]

                cache_dir = project_root / "mnemonic_cortex" / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                bundle_file = cache_dir / f"{bundle_name}_bundle.json"

                with open(bundle_file, 'w', encoding='utf-8') as f:
                    json.dump(docs_to_cache, f, indent=2)

                logger.info(f"[CACHE] Prefilled {len(docs_to_cache)} {bundle_name} entries.")

            except Exception as e:
                logger.error(f"Failed to get latest documents for {prefix}: {e}")

        # Handle roadmap file separately as it's a single file
        roadmap_path = project_root / "ROADMAP" / "PHASED_EVOLUTION_PLAN_Phase2-Phase3-Protocol113.md"
        if roadmap_path.exists():
            # Cache this logic in a similar fashion if needed
            logger.info("[CACHE] Roadmap file found; skipping detailed cache behavior for roadmap.")
        else:
            logger.warning("[CACHE] Roadmap file not found, skipping.")

    @staticmethod
    def prefill_guardian_delta(files_to_add):
        """Placeholder for refreshing cache with specific files after a git commit."""
        print(f"[CACHE DELTA] Received {len(files_to_add)} files to refresh cache (logic not yet implemented).")
# council_orchestrator/orchestrator/memory/cache.py
# Cache as Learning (CAG) functionality with Guardian Start Pack prefill

import hashlib
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .cortex import CortexManager

# Global cache store (Phase 3: replace with SQLite backend)
CACHE: Dict[str, Dict[str, Any]] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class CacheItem:
    key: str
    value: Any
    ttl_seconds: int
    created_at: float = time.time()
    ema_score: float = 0.0  # EMA tracking for Phase 3 promotion

class CacheManager:
    """Phase 3 Cache Manager with Guardian Start Pack prefill."""

    def __init__(self, project_root: Path = None, logger = None):
        self.project_root = project_root or PROJECT_ROOT
        self.logger = logger

    def set(self, item: CacheItem) -> None:
        """Store item in cache with TTL."""
        CACHE[item.key] = {
            "value": item.value,
            "expires_at": item.created_at + item.ttl_seconds,
            "ema_score": item.ema_score,
            "created_at": item.created_at,
        }

    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache, respecting TTL."""
        rec = CACHE.get(key)
        if not rec:
            return None
        if time.time() > rec["expires_at"]:
            CACHE.pop(key, None)
            return None
        return rec["value"]

    def prefill_guardian_start_pack(self, cortex_manager: "CortexManager") -> None:
        """
        Prefills the cache by querying the Mnemonic Cortex, adhering to Protocol 85.
        """
        if self.logger:
            self.logger.info("[CACHE] Initiating Guardian Start Pack pre-fill from Mnemonic Cortex...")

        # 1. Chronicles (latest N from RAG DB)
        chronicles = cortex_manager.get_latest_documents_by_path(path_prefix="00_CHRONICLE/ENTRIES/", n_results=15)
        self.set(CacheItem("guardian:dashboard:chronicles:latest", chronicles, ttl_seconds=86400))
        self.logger.info(f"[CACHE] Prefilled {len(chronicles)} chronicle entries.")

        # 2. Protocols (latest N from RAG DB)
        protocols = cortex_manager.get_latest_documents_by_path(path_prefix="01_PROTOCOLS/", n_results=15)
        self.set(CacheItem("guardian:dashboard:protocols:latest", protocols, ttl_seconds=86400))
        self.logger.info(f"[CACHE] Prefilled {len(protocols)} protocol entries.")

        # 3. Roadmap (static file, as before)
        roadmap_path_str = "ROADMAP/PHASED_EVOLUTION_PLAN_Phase2-Phase3-Protocol113.md"
        roadmap_path = self.project_root / roadmap_path_str
        if roadmap_path.exists():
            roadmap_content = roadmap_path.read_text(encoding="utf-8")
            roadmap_item = [{
                "title": "Phased Evolution Plan",
                "path": roadmap_path_str,
                "updated_at": time.strftime("%Y-%m-%d", time.localtime(roadmap_path.stat().st_mtime))
            }]
            self.set(CacheItem("guardian:dashboard:roadmap", roadmap_item, ttl_seconds=86400))
            self.logger.info("[CACHE] Prefilled roadmap.")
        else:
            self.logger.warning("[CACHE] Roadmap file not found, skipping.")
            self.set(CacheItem("guardian:dashboard:roadmap", [], ttl_seconds=86400))

        if self.logger:
            self.logger.info("[CACHE] Pre-fill complete. Cache is warm.")

    def prefill_guardian_delta(self, updated_files: List[str]) -> None:
        """Refresh cache for updated files during ingest/git-ops."""
        watched = {
            "00_CHRONICLE/ENTRIES": "guardian:dashboard:chronicles:latest",
            "01_PROTOCOLS": "guardian:dashboard:protocols:latest",
            "ROADMAP": "guardian:dashboard:roadmap",
            "council_orchestrator/README.md": "guardian:docs:orchestrator_readme",
            "council_orchestrator/command_schema.md": "guardian:docs:command_schema",
            "council_orchestrator/howto-commit-command.md": "guardian:docs:howto_commit",
            "council_orchestrator/schemas/council-round-packet-v1.0.0.json": "guardian:packets:schema",
            "council_orchestrator/OPERATION_OPTICAL_ANVIL_BLUEPRINT.md": "guardian:blueprint:optical_anvil",
            "council_orchestrator/schemas/engine_config.json": "guardian:ops:engine_config",
        }

        # Refresh keys for updated paths
        for path in updated_files:
            for watch, key in watched.items():
                if path == watch or path.startswith(f"{watch}/"):
                    if key == "guardian:dashboard:chronicles:latest":
                        chronicles = self._collect_latest("00_CHRONICLE/ENTRIES", (".md",), 8)
                        self.set(CacheItem(key, chronicles, 86400))
                    elif key == "guardian:dashboard:protocols:latest":
                        protocols = self._collect_latest("01_PROTOCOLS", (".md",), 8)
                        self.set(CacheItem(key, protocols, 86400))
                    elif key == "guardian:dashboard:roadmap":
                        roadmap = self._read_concat(["ROADMAP/PHASED_EVOLUTION_PLAN_Phase2-Phase3-Protocol113.md"])
                        self.set(CacheItem(key, roadmap, 86400))
                    elif key == "guardian:ops:engine_config":
                        self._set_text_file(key, "council_orchestrator/schemas/engine_config.json", 21600)
                    else:
                        # docs/blueprints/schemas
                        self._set_text_file(key, watch, 86400)

    # ---------- helpers ----------
    def _collect_latest(self, rel_dir: str, exts: Tuple[str, ...], limit: int) -> List[Dict[str, Any]]:
        """Collect latest N files from directory."""
        base = self.project_root / rel_dir
        items = []
        if not base.exists():
            return items
        for p in sorted(base.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.suffix.lower() in exts:
                items.append({
                    "path": str(p.relative_to(self.project_root)),
                    "name": p.name,
                    "mtime": p.stat().st_mtime
                })
                if len(items) >= limit:
                    break
        return items

    def _read_concat(self, paths: List[str]) -> str:
        """Concatenate multiple files with separators."""
        chunks = []
        for rel in paths:
            p = self.project_root / rel
            if p.exists():
                chunks.append(p.read_text(encoding="utf-8"))
        return "\n\n---\n\n".join(chunks)

    def _set_text_file(self, key: str, rel: str, ttl: int) -> None:
        """Cache a text file."""
        p = self.project_root / rel
        if p.exists():
            self.set(CacheItem(key, p.read_text(encoding="utf-8"), ttl))

    def _set_tail(self, key: str, rel: str, lines: int, ttl: int) -> None:
        """Cache tail of a text file."""
        p = self.project_root / rel
        if p.exists():
            text = p.read_text(encoding="utf-8").splitlines()[-lines:]
            self.set(CacheItem(key, "\n".join(text), ttl))

    def _latest_jsonl(self, rel_root: str) -> Optional[Dict[str, str]]:
        """Find most recent JSONL file in rounds directory."""
        root = self.project_root / rel_root
        if not root.exists():
            return None

        latest = None
        for p in root.glob("**/round_*.jsonl"):
            if latest is None or p.stat().st_mtime > latest.stat().st_mtime:
                latest = p

        if latest:
            return {"path": str(latest.relative_to(self.project_root))}
        return None

    def get_bundle_items(self, bundle_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return a list of dict items for the given bundle.
        Each item: {title, path, updated_at, source, size}
        This function reads from cache backend.
        """
        bundle_key_map = {
            "chronicles": "guardian:dashboard:chronicles:latest",
            "protocols": "guardian:dashboard:protocols:latest",
            "roadmap": "guardian:dashboard:roadmap"
        }
        
        key = bundle_key_map.get(bundle_name)
        if not key:
            return []
        
        data = self.get(key)
        if not data:
            return []
        
        # data is a list of dicts like [{"path": "...", "name": "...", "mtime": ...}]
        items = []
        for item in data[:limit]:
            items.append({
                "title": item.get("name", "").replace(".md", "").replace("_", " "),
                "path": item.get("path", ""),
                "updated_at": time.strftime("%Y-%m-%d", time.localtime(item.get("mtime", 0))),
                "source": "cache",
                "size": "N/A"  # Could calculate if needed
            })
        return items

    def get_keys(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Get cache entries for specific keys with metadata."""
        entries = []
        current_time = time.time()
        
        for key in keys:
            rec = CACHE.get(key)
            if rec:
                expires_at = rec["expires_at"]
                if current_time > expires_at:
                    # Expired
                    CACHE.pop(key, None)
                    entries.append({
                        "key": key,
                        "missing": False,
                        "expired": True,
                        "refreshed": False,
                        "ttl_remaining": "expired",
                        "size": "N/A",
                        "sha256_prefix": "N/A",
                        "source": "cache",
                        "last_updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(rec["created_at"]))
                    })
                else:
                    # Valid entry
                    ttl_remaining_seconds = int(expires_at - current_time)
                    ttl_display = f"{ttl_remaining_seconds // 3600}h{(ttl_remaining_seconds % 3600) // 60}m"
                    
                    value = rec["value"]
                    size_bytes = len(str(value).encode('utf-8'))
                    size_display = f"{size_bytes / 1024:.1f} KB" if size_bytes > 1024 else f"{size_bytes} B"
                    
                    sha256 = hashlib.sha256(str(value).encode('utf-8')).hexdigest()
                    
                    entries.append({
                        "key": key,
                        "missing": False,
                        "expired": False,
                        "refreshed": False,
                        "ttl_remaining": ttl_display,
                        "size": size_display,
                        "sha256_prefix": sha256,
                        "source": "cache",
                        "last_updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(rec["created_at"]))
                    })
            else:
                # Missing
                entries.append({
                    "key": key,
                    "missing": True,
                    "expired": False,
                    "refreshed": False,
                    "ttl_remaining": "N/A",
                    "size": "N/A",
                    "sha256_prefix": "N/A",
                    "source": "N/A",
                    "last_updated": "N/A"
                })
        
        return entries

    def fetch_guardian_start_pack(self, bundles: List[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Fetch Guardian Start Pack bundles from cache for boot digest."""
        if bundles is None:
            bundles = ["chronicles", "protocols", "roadmap"]

        result = {"bundles": {}}

        for bundle_name in bundles:
            if bundle_name == "chronicles":
                # Get chronicles from cache
                cache_key = "guardian:dashboard:chronicles:latest"
                cached_data = self.get(cache_key)
                if cached_data:
                    # Parse the cached data (it's a list of file info)
                    try:
                        items = json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                        result["bundles"]["chronicles"] = items[:limit]
                    except (json.JSONDecodeError, TypeError):
                        result["bundles"]["chronicles"] = []
                else:
                    result["bundles"]["chronicles"] = []

            elif bundle_name == "protocols":
                # Get protocols from cache
                cache_key = "guardian:dashboard:protocols:latest"
                cached_data = self.get(cache_key)
                if cached_data:
                    try:
                        items = json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                        result["bundles"]["protocols"] = items[:limit]
                    except (json.JSONDecodeError, TypeError):
                        result["bundles"]["protocols"] = []
                else:
                    result["bundles"]["protocols"] = []

            elif bundle_name == "roadmap":
                # Get roadmap from cache
                cache_key = "guardian:dashboard:roadmap"
                cached_data = self.get(cache_key)
                if cached_data:
                    # Roadmap is a single text blob, convert to single-item list
                    result["bundles"]["roadmap"] = [{
                        "title": "PHASED_EVOLUTION_PLAN_Phase2-Phase3-Protocol113",
                        "path": "ROADMAP/PHASED_EVOLUTION_PLAN_Phase2-Phase3-Protocol113.md",
                        "content": cached_data[:500] + "..." if len(cached_data) > 500 else cached_data,
                        "updated_at": "cached"
                    }]
                else:
                    result["bundles"]["roadmap"] = []

        return result


def get_cag_data(prompt: str, engine_type: str, cache_adapter = None) -> Dict[str, Any]:
    """Get CAG (Cache as Learning) data for round packet."""
    try:
        # Generate cache key from prompt and engine
        query_key = hashlib.sha256(f"{prompt}:{engine_type}".encode()).hexdigest()[:16]

        # Check cache (simplified - would use actual cache DB)
        cache_hit = False
        hit_streak = 0

        # Phase 3 readiness: EMA tracking
        ema_data = {}
        if cache_adapter:
            ema_data = cache_adapter.update_ema(query_key)

        # In real implementation, would query SQLite cache database
        # For now, return placeholder data
        return {
            "query_key": query_key,
            "cache_hit": cache_hit,
            "hit_streak": hit_streak,
            "last_hit_at": ema_data.get("last_hit_at", 0),
            "ema_7d": ema_data.get("ema_7d", 0.0)
        }
    except Exception as e:
        return {"error": str(e)}