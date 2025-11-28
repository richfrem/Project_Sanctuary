"""
Mnemonic Cache (core/cache.py)
Implements the Cached Augmented Generation (CAG) layer for the Mnemonic Cortex.

This module provides a two-tier caching system to eliminate redundant cognitive load
and ensure instant responses for common queries, aligning with the Hearth Protocol (P43).

Architecture:
- Hot Cache (In-Memory): Python dict for sub-millisecond access to recent queries
- Warm Cache (Persistent): SQLite-based storage for cross-session persistence
- Cache Key: SHA-256 hash of structured query JSON for deterministic lookups
- Cache Population: Integrated with cache_warmup.py for proactive loading

Usage:
    from mnemonic_cortex.core.cache import MnemonicCache

    cache = MnemonicCache()
    key = cache.generate_key(structured_query_json)

    # Check cache
    result = cache.get(key)
    if result:
        return result  # Cache hit

    # Cache miss - compute answer
    answer = generate_rag_answer(structured_query_json)
    cache.set(key, answer)
    return answer
"""

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, Optional


class MnemonicCache:
    """
    Two-tier caching system for Mnemonic Cortex queries.

    Hot Cache: In-memory dict for instant access
    Warm Cache: SQLite database for persistence
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the two-tier cache system.

        Args:
            db_path: Path to SQLite database for warm cache. Defaults to project cache dir.
        """
        # Hot Cache: In-memory dictionary
        self.hot_cache: Dict[str, Any] = {}
        self.hot_cache_lock = threading.Lock()

        # Warm Cache: SQLite database
        if db_path is None:
            # Check env var first
            env_path = os.getenv("MNEMONIC_CACHE_DB_PATH")
            if env_path:
                db_path = env_path
            else:
                # Default to mnemonic_cortex/cache directory
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                cache_dir = os.path.join(project_root, 'mnemonic_cortex', 'cache')
                os.makedirs(cache_dir, exist_ok=True)
                db_path = os.path.join(cache_dir, 'mnemonic_cache.db')

        self.db_path = db_path
        self._init_warm_cache()

    def _init_warm_cache(self):
        """Initialize the SQLite warm cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create index for faster lookups
            conn.execute('CREATE INDEX IF NOT EXISTS idx_key ON cache(key)')

    def generate_key(self, structured_query: Dict[str, Any]) -> str:
        """
        Generate a deterministic cache key from a structured query.

        Args:
            structured_query: JSON-serializable dict containing query and filters

        Returns:
            SHA-256 hash of the JSON representation
        """
        # Sort keys for consistent hashing
        query_json = json.dumps(structured_query, sort_keys=True)
        return hashlib.sha256(query_json.encode('utf-8')).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache (Hot cache first, then Warm cache).

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        # Check Hot Cache first
        with self.hot_cache_lock:
            if key in self.hot_cache:
                # Update access stats in background
                threading.Thread(target=self._update_access_stats, args=(key,), daemon=True).start()
                return self.hot_cache[key]

        # Check Warm Cache
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT value FROM cache WHERE key = ?',
                    (key,)
                )
                result = cursor.fetchone()

                if result:
                    value = json.loads(result[0])
                    # Promote to Hot Cache
                    with self.hot_cache_lock:
                        self.hot_cache[key] = value

                    # Update access stats
                    threading.Thread(target=self._update_access_stats, args=(key,), daemon=True).start()
                    return value

        except Exception as e:
            print(f"[CACHE] Warning: Error reading from warm cache: {e}")

        return None

    def set(self, key: str, value: Any, promote_to_hot: bool = True) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            promote_to_hot: Whether to also store in hot cache
        """
        # Store in Hot Cache
        if promote_to_hot:
            with self.hot_cache_lock:
                self.hot_cache[key] = value

        # Store in Warm Cache
        try:
            json_value = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)',
                    (key, json_value)
                )
                conn.commit()
        except Exception as e:
            print(f"[CACHE] Warning: Error writing to warm cache: {e}")

    def clear_hot_cache(self) -> None:
        """Clear the in-memory hot cache."""
        with self.hot_cache_lock:
            self.hot_cache.clear()

    def clear_warm_cache(self) -> None:
        """Clear the persistent warm cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM cache')
                conn.commit()
        except Exception as e:
            print(f"[CACHE] Warning: Error clearing warm cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'hot_cache_size': len(self.hot_cache),
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*), SUM(access_count) FROM cache')
                result = cursor.fetchone()
                stats.update({
                    'warm_cache_entries': result[0] or 0,
                    'total_accesses': result[1] or 0,
                })
        except Exception as e:
            print(f"[CACHE] Warning: Error getting warm cache stats: {e}")

        return stats

    def _update_access_stats(self, key: str) -> None:
        """Update access statistics for a cache entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE cache SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE key = ?',
                    (key,)
                )
                conn.commit()
        except Exception as e:
            print(f"[CACHE] Warning: Error updating access stats: {e}")


# Global cache instance for application-wide use
_cache_instance: Optional[MnemonicCache] = None
_cache_lock = threading.Lock()


def get_cache() -> MnemonicCache:
    """Get the global cache instance (singleton pattern)."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = MnemonicCache()
    return _cache_instance