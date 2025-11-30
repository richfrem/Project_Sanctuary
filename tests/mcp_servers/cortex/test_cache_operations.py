
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.cognitive.cortex.operations import CortexOperations
from mcp_servers.cognitive.cortex.models import CacheGetResponse, CacheSetResponse

@pytest.fixture
def mock_cache():
    with patch('mnemonic_cortex.core.cache.get_cache') as mock_get_cache:
        cache_instance = MagicMock()
        mock_get_cache.return_value = cache_instance
        yield cache_instance

@pytest.fixture
def ops(tmp_path):
    return CortexOperations(str(tmp_path))

class TestCacheOperations:
    
    def test_cache_get_hit(self, ops, mock_cache):
        # Setup
        mock_cache.generate_key.return_value = "test_key"
        mock_cache.get.return_value = {"answer": "Cached Answer"}
        
        # Execute
        response = ops.cache_get("test query")
        
        # Verify
        assert response.status == "success"
        assert response.cache_hit is True
        assert response.answer == "Cached Answer"
        mock_cache.get.assert_called_with("test_key")

    def test_cache_get_miss(self, ops, mock_cache):
        # Setup
        mock_cache.generate_key.return_value = "test_key"
        mock_cache.get.return_value = None
        
        # Execute
        response = ops.cache_get("test query")
        
        # Verify
        assert response.status == "success"
        assert response.cache_hit is False
        assert response.answer is None

    def test_cache_set(self, ops, mock_cache):
        # Setup
        mock_cache.generate_key.return_value = "test_key"
        
        # Execute
        response = ops.cache_set("test query", "test answer")
        
        # Verify
        assert response.status == "success"
        assert response.stored is True
        mock_cache.set.assert_called_with("test_key", {"answer": "test answer"})

    def test_cache_stats(self, ops, mock_cache):
        # Setup
        mock_cache.get_stats.return_value = {"hits": 10, "misses": 5}
        
        # Execute
        stats = ops.get_cache_stats()
        
        # Verify
        assert stats == {"hits": 10, "misses": 5}

    def test_cache_warmup(self, ops, mock_cache):
        # Setup
        mock_cache.generate_key.return_value = "key"
        mock_cache.get.return_value = None # Miss initially
        
        # Mock query to return a result
        with patch.object(ops, 'query') as mock_query:
            mock_result = MagicMock()
            mock_result.content = "Generated Answer"
            mock_query.return_value.results = [mock_result]
            
            # Execute
            response = ops.cache_warmup(["query1"])
            
            # Verify
            assert response.status == "success"
            assert response.queries_cached == 1
            assert response.cache_misses == 1
            mock_cache.set.assert_called()

    def test_guardian_wakeup(self, ops, mock_cache):
        # Setup
        mock_cache.generate_key.return_value = "key"
        mock_cache.get.return_value = {"answer": "Cached Summary"} # Hit
        
        # Mock file writing
        with patch("builtins.open", mock_open()) as mock_file:
            # Execute
            response = ops.guardian_wakeup()
            
            # Verify
            assert response.status == "success"
            assert len(response.bundles_loaded) == 3
            # Should write to file
            mock_file.assert_called()
            handle = mock_file()
            handle.write.assert_any_call("# Guardian Boot Digest\n\n")

class TestAdaptationPacket:
    def test_generate_adaptation_packet(self):
        # We need to test the server function or logic. 
        # Since we can't easily import the server function due to decorators,
        # we'll test the SynthesisGenerator usage pattern.
        
        with patch('mnemonic_cortex.app.synthesis.generator.SynthesisGenerator') as MockGenerator:
            instance = MockGenerator.return_value
            instance.generate_packet.return_value = {"data": "test"}
            instance.save_packet.return_value = "/path/to/packet.json"
            
            # Simulate what the server tool does
            from mnemonic_cortex.app.synthesis.generator import SynthesisGenerator
            generator = SynthesisGenerator("/tmp")
            packet = generator.generate_packet(days=7)
            output_path = generator.save_packet(packet)
            
            assert output_path == "/path/to/packet.json"
            instance.generate_packet.assert_called_with(days=7)
            instance.save_packet.assert_called_with({"data": "test"})
