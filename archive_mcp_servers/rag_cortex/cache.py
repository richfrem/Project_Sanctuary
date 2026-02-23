#   - _update_access_stats
#   - clear_hot_cache
#   - clear_warm_cache
#   - generate_key
#   - get
#   - get_cache
#   - get_stats
#   - set
#============================================
import hashlib
import json
import os
import sqlite3
import threading
import logging
import time
from typing import Any, Dict, Optional, List, Tuple

# Configure logging
logger = logging.getLogger("rag_cortex.cache")

from mcp_servers.lib.env_helper import get_env_variable


#============================================
# Class: MnemonicCache
# Purpose: Two-tier caching system for Mnemonic Cortex queries.
# Components:
#   Hot Cache: In-memory dict for instant access
#   Warm Cache: SQLite database for persistence
#============================================
class MnemonicCache:

    #============================================
    # Method: __init__
    # Purpose: Initialize the two-tier cache system.
    # Args:
    #   db_path: Path to SQLite database for warm cache. Defaults to project cache dir.
    #============================================
    def __init__(self, db_path: str = None):
        # Hot Cache: In-memory dictionary
        self.hot_cache: Dict[str, Any] = {}
        self.hot_cache_lock = threading.Lock()

        # Warm Cache: SQLite database
        if db_path is None:
            # Check env var first
            env_path = get_env_variable("MNEMONIC_CACHE_DB_PATH", required=False)
            if env_path:
                db_path = env_path
            else:
                # Default to mcp_servers/cognitive/cortex/data/cache directory
                # Note: Assuming this file is in mcp_servers/cognitive/cortex/
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cache_dir = os.path.join(current_dir, 'data', 'cache')
                os.makedirs(cache_dir, exist_ok=True)
                db_path = os.path.join(cache_dir, 'mnemonic_cache.db')

        self.db_path = db_path
        self._init_warm_cache()

    #============================================
    # Method: _init_warm_cache
    # Purpose: Initialize the SQLite warm cache database.
    #============================================
    def _init_warm_cache(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS warm_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create index for faster lookups
            conn.execute('CREATE INDEX IF NOT EXISTS idx_key ON warm_cache(key)')

    #============================================
    # Method: generate_key
    # Purpose: Generate a deterministic cache key from a structured query.
    # Args:
    #   structured_query: JSON-serializable dict containing query and filters
    # Returns: SHA-256 hash of the JSON representation
    #============================================
    def generate_key(self, structured_query: Dict[str, Any]) -> str:
        # Sort keys for consistent hashing
        query_json = json.dumps(structured_query, sort_keys=True)
        return hashlib.sha256(query_json.encode('utf-8')).hexdigest()

    #============================================
    # Method: get
    # Purpose: Retrieve a value from the cache (Hot cache first, then Warm cache).
    # Args:
    #   key: Cache key
    # Returns: Cached value if found, None otherwise
    #============================================
    def get(self, key: str) -> Optional[Any]:
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
                    'SELECT value, access_count, last_accessed FROM warm_cache WHERE key = ?',
                    (key,)
                )
                row = cursor.fetchone()

                if row:
                    value_json, access_count, last_accessed = row
                    # Promote to Hot Cache
                    with self.hot_cache_lock:
                        self.hot_cache[key] = json.loads(value_json)

                    # Update access stats asynchronously/lazily
                    # In this simple implementation we just do it here
                    self._update_access_stats(key)
                    
                    return json.loads(value_json)
            return None
        except Exception as e:
            logger.warning(f"[CACHE] Warning: Error reading from warm cache: {e}")
            return None

    #============================================
    # Method: set
    # Purpose: Store a value in the cache.
    # Args:
    #   key: Cache key
    #   value: Value to cache (must be JSON serializable)
    #   promote_to_hot: Whether to also store in hot cache
    #============================================
    def set(self, key: str, value: Any, promote_to_hot: bool = True) -> None:
        # Store in Hot Cache
        if promote_to_hot:
            with self.hot_cache_lock:
                self.hot_cache[key] = value

        # Store in Warm Cache
        try:
            json_value = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                "INSERT OR REPLACE INTO warm_cache (key, value, access_count, last_accessed) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value), 0, time.time())
            )
            conn.commit()
            
        except Exception as e:
            logger.warning(f"[CACHE] Warning: Error writing to warm cache: {e}")

    #============================================
    # Method: clear_hot_cache
    # Purpose: Clear the in-memory hot cache.
    #============================================
    def clear_hot_cache(self) -> None:
        with self.hot_cache_lock:
            self.hot_cache.clear()

    #============================================
    # Method: clear_warm_cache
    # Purpose: Clear the persistent warm cache.
    #============================================
    def clear_warm_cache(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM warm_cache")
                conn.commit()
        except Exception as e:
            logger.warning(f"[CACHE] Warning: Error clearing warm cache: {e}")

    #============================================
    # Method: get_stats
    # Purpose: Get cache statistics.
    #============================================
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'hot_cache_size': len(self.hot_cache),
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM warm_cache')
                count = cursor.fetchone()[0]
            return {
                "size": count,
                "db_path": self.db_path
            }
        except Exception as e:
            logger.warning(f"[CACHE] Warning: Error getting warm cache stats: {e}")
            return {"error": str(e)}

    #============================================
    # Method: _update_access_stats
    # Purpose: Update access statistics for a cache entry.
    # Args:
    #   key: Cache key
    #============================================
    def _update_access_stats(self, key: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Increment access count and update last_accessed
                cursor.execute(
                    "UPDATE warm_cache SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                    (time.time(), key)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"[CACHE] Warning: Error updating access stats: {e}")


# Global cache instance for application-wide use
_cache_instance: Optional[MnemonicCache] = None
_cache_lock = threading.Lock()


#============================================
# Function: get_cache
# Purpose: Get the global cache instance (singleton pattern).
# Returns: MnemonicCache instance
#============================================
def get_cache() -> MnemonicCache:
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = MnemonicCache()
    return _cache_instance