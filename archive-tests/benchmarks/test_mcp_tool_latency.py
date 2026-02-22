import pytest
import time
from unittest.mock import MagicMock

# Import operations to benchmark
from mcp_servers.chronicle.operations import ChronicleOperations
from mcp_servers.task.operations import TaskOperations
from mcp_servers.protocol.operations import ProtocolOperations

@pytest.mark.benchmark
def test_chronicle_list_latency(benchmark, tmp_path):
    """Benchmark Chronicle list_entries latency."""
    # Setup
    chronicle_dir = tmp_path / "00_CHRONICLE"
    chronicle_dir.mkdir()
    # Create a few dummy entries
    for i in range(10):
        (chronicle_dir / f"{i:03d}_entry.md").write_text(f"# Entry {i}")
        
    ops = ChronicleOperations(str(tmp_path))
    
    # Benchmark
    result = benchmark(ops.list_entries)
    assert len(result) == 10

@pytest.mark.benchmark
def test_task_list_latency(benchmark, tmp_path):
    """Benchmark Task list_tasks latency."""
    # Setup
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "backlog").mkdir()
    (task_dir / "todo").mkdir()
    (task_dir / "in-progress").mkdir()
    (task_dir / "done").mkdir()
    
    # Create dummy tasks
    for i in range(10):
        (task_dir / "backlog" / f"{i:03d}_task.md").write_text(f"# Task {i}\nStatus: Backlog")
        
    ops = TaskOperations(str(tmp_path))
    
    # Benchmark
    result = benchmark(ops.list_tasks)
    assert len(result) == 10

@pytest.mark.benchmark
def test_protocol_get_latency(benchmark, tmp_path):
    """Benchmark Protocol get_protocol latency."""
    # Setup
    proto_dir = tmp_path / "01_PROTOCOLS"
    proto_dir.mkdir()
    (proto_dir / "101_test_protocol.md").write_text("# Protocol 101\nTitle: Test")
    
    ops = ProtocolOperations(str(proto_dir))
    
    # Benchmark
    result = benchmark(ops.get_protocol, 101)
    assert result["number"] == 101
