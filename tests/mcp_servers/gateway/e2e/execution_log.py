"""
Gateway E2E Test Suite - Execution Framework

This module provides utilities for comprehensive E2E testing of all 86 Gateway MCP operations
with detailed execution logging per Task 148 requirements.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TestExecution:
    """Record of a single test execution."""
    tool_name: str
    timestamp: str
    input_args: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    success: bool = False


class ExecutionLogger:
    """
    Execution logger that records detailed test execution data.
    
    Per Task 148: Every test must show timestamp, tool call, input, output, duration.
    Anti-shortcut validation: proves each test actually executed.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.executions: List[TestExecution] = []
        self.log_path = log_path or Path(__file__).parent / "execution_log.json"
        self.start_time = datetime.utcnow()
    
    def log_execution(
        self,
        tool_name: str,
        input_args: Dict[str, Any],
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0
    ) -> TestExecution:
        """Log a single tool execution with all required metadata."""
        execution = TestExecution(
            tool_name=tool_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            input_args=input_args,
            output=output,
            error=error,
            duration_ms=duration_ms,
            success=error is None and output is not None
        )
        self.executions.append(execution)
        return execution
    
    def save(self) -> Path:
        """Save execution log to JSON file."""
        report = {
            "test_run": {
                "started": self.start_time.isoformat() + "Z",
                "completed": datetime.utcnow().isoformat() + "Z",
                "total_executions": len(self.executions),
                "passed": sum(1 for e in self.executions if e.success),
                "failed": sum(1 for e in self.executions if not e.success),
            },
            "executions": [asdict(e) for e in self.executions]
        }
        self.log_path.write_text(json.dumps(report, indent=2))
        return self.log_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        return {
            "total": len(self.executions),
            "passed": sum(1 for e in self.executions if e.success),
            "failed": sum(1 for e in self.executions if not e.success),
            "pass_rate": (
                sum(1 for e in self.executions if e.success) / len(self.executions) * 100
                if self.executions else 0
            )
        }


# Global logger instance for pytest fixtures
_execution_logger: Optional[ExecutionLogger] = None


def get_execution_logger() -> ExecutionLogger:
    """Get or create the global execution logger."""
    global _execution_logger
    if _execution_logger is None:
        _execution_logger = ExecutionLogger()
    return _execution_logger


def reset_execution_logger() -> None:
    """Reset the global execution logger (for test isolation)."""
    global _execution_logger
    _execution_logger = None
