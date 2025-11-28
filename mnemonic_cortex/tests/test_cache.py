import pytest
import sqlite3
import json
import time
from mnemonic_cortex.core.cache import MnemonicCache, get_cache
from mnemonic_cortex.core import cache as cache_module

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_cache.db")

@pytest.fixture
def cache_instance(temp_db_path):
    """Create a cache instance with a temp DB."""
    return MnemonicCache(db_path=temp_db_path)

def test_generate_key(cache_instance):
    """Test deterministic key generation."""
    q1 = {"semantic": "test", "filters": {"a": 1}}
    q2 = {"filters": {"a": 1}, "semantic": "test"} # Different order
    
    k1 = cache_instance.generate_key(q1)
    k2 = cache_instance.generate_key(q2)
    
    assert k1 == k2
    assert len(k1) == 64 # SHA-256 hex digest

def test_set_get_hot(cache_instance):
    """Test hot cache operations."""
    key = "test_key"
    value = {"data": "test_value"}
    
    cache_instance.set(key, value)
    
    # Check hot cache directly
    assert key in cache_instance.hot_cache
    assert cache_instance.hot_cache[key] == value
    
    # Check get
    assert cache_instance.get(key) == value

def test_set_get_warm(cache_instance, temp_db_path):
    """Test warm cache persistence and promotion."""
    key = "warm_key"
    value = {"data": "warm_value"}
    
    cache_instance.set(key, value)
    
    # Clear hot cache to force warm lookup
    cache_instance.clear_hot_cache()
    assert key not in cache_instance.hot_cache
    
    # Verify it's in DB
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.execute("SELECT value FROM cache WHERE key=?", (key,))
        row = cursor.fetchone()
        assert row is not None
        assert json.loads(row[0]) == value
    
    # Get should retrieve from warm and promote to hot
    retrieved = cache_instance.get(key)
    assert retrieved == value
    assert key in cache_instance.hot_cache # Promoted

def test_clear_cache(cache_instance, temp_db_path):
    """Test clearing caches."""
    cache_instance.set("k1", "v1")
    
    cache_instance.clear_hot_cache()
    assert "k1" not in cache_instance.hot_cache
    
    # Still in warm
    with sqlite3.connect(temp_db_path) as conn:
        assert conn.execute("SELECT count(*) FROM cache").fetchone()[0] == 1
        
    cache_instance.clear_warm_cache()
    with sqlite3.connect(temp_db_path) as conn:
        assert conn.execute("SELECT count(*) FROM cache").fetchone()[0] == 0

def test_singleton_reset():
    """Test singleton getter."""
    # Reset global
    cache_module._cache_instance = None
    
    c1 = get_cache()
    c2 = get_cache()
    
    assert c1 is c2
    assert isinstance(c1, MnemonicCache)
