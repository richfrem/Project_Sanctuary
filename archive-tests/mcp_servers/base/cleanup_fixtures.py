"""
Cleanup Utilities for E2E and Integration Tests
================================================

Provides pytest fixtures for managing test artifacts that get created
in the real project directories (ADRs, Chronicle entries, etc.).

Per Task 149 - Self-Cleaning Test Logic:
- MUST only delete generated outputs, never test scripts
- Use targeted patterns, not broad wildcards
- Implement in teardown/finally blocks

Usage:
------
@pytest.fixture
def my_test_artifact(e2e_cleanup):
    # Create artifact
    path = Path(PROJECT_ROOT) / "ADRs" / "999_e2e_test_artifact.md"
    path.write_text("test content")
    e2e_cleanup.register(path)  # Auto-cleaned after test
    yield path
"""
import pytest
import os
import re
from pathlib import Path
from typing import List, Set, Optional
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class CleanupRegistry:
    """
    Registry for tracking test-generated files for cleanup.
    
    Files registered here will be deleted after the test completes,
    even if the test fails.
    """
    _files: Set[Path] = field(default_factory=set)
    _patterns: List[re.Pattern] = field(default_factory=list)
    _directories: Set[Path] = field(default_factory=set)
    
    def register(self, path: Path) -> None:
        """Register a specific file for cleanup."""
        if isinstance(path, str):
            path = Path(path)
        self._files.add(path.resolve())
    
    def register_pattern(self, directory: Path, pattern: str) -> None:
        """
        Register a glob pattern for cleanup within a directory.
        
        Args:
            directory: Directory to search in
            pattern: Glob pattern (e.g., "*_e2e_test*.md")
        """
        self._patterns.append((Path(directory).resolve(), pattern))
    
    def register_directory(self, path: Path) -> None:
        """Register a directory for cleanup (will be deleted if empty after file cleanup)."""
        if isinstance(path, str):
            path = Path(path)
        self._directories.add(path.resolve())
    
    def cleanup(self) -> dict:
        """
        Execute cleanup of all registered files and patterns.
        
        Returns:
            dict with cleanup statistics
        """
        stats = {"files_deleted": 0, "files_failed": 0, "dirs_deleted": 0}
        
        # 1. Clean specific files
        for path in self._files:
            if path.exists() and path.is_file():
                try:
                    os.remove(path)
                    stats["files_deleted"] += 1
                    print(f"🧹 Cleaned: {path.name}")
                except Exception as e:
                    stats["files_failed"] += 1
                    print(f"⚠️  Failed to clean {path}: {e}")
        
        # 2. Clean pattern matches
        for directory, pattern in self._patterns:
            if directory.exists():
                for match in directory.glob(pattern):
                    if match.is_file():
                        try:
                            os.remove(match)
                            stats["files_deleted"] += 1
                            print(f"🧹 Cleaned (pattern): {match.name}")
                        except Exception as e:
                            stats["files_failed"] += 1
                            print(f"⚠️  Failed to clean {match}: {e}")
        
        # 3. Clean empty directories
        for directory in sorted(self._directories, reverse=True):  # depth-first
            if directory.exists() and directory.is_dir():
                try:
                    if not any(directory.iterdir()):  # Only if empty
                        directory.rmdir()
                        stats["dirs_deleted"] += 1
                except Exception:
                    pass  # Non-empty dirs are expected, don't report
        
        # Reset for next test
        self._files.clear()
        self._patterns.clear()
        self._directories.clear()
        
        return stats


@pytest.fixture
def e2e_cleanup():
    """
    Fixture that provides a cleanup registry for E2E tests.
    
    Usage:
        def test_something(e2e_cleanup):
            path = create_test_file()
            e2e_cleanup.register(path)
            # ... test logic ...
            # Cleanup happens automatically after test
    """
    registry = CleanupRegistry()
    yield registry
    registry.cleanup()


@pytest.fixture
def safe_adr_path(e2e_cleanup):
    """
    Fixture that provides a safe ADR path for testing.
    Returns a path that will be auto-cleaned after the test.
    
    Usage:
        def test_adr_creation(safe_adr_path):
            adr_path = safe_adr_path(999, "e2e_test_artifact")
            # Creates: ADRs/999_e2e_test_artifact.md
    """
    def _create_path(number: int, suffix: str = "e2e_test") -> Path:
        filename = f"{number:03d}_{suffix}.md"
        path = PROJECT_ROOT / "ADRs" / filename
        e2e_cleanup.register(path)
        return path
    
    return _create_path


@pytest.fixture
def safe_chronicle_path(e2e_cleanup):
    """
    Fixture that provides a safe Chronicle entry path for testing.
    Returns a path that will be auto-cleaned after the test.
    """
    def _create_path(number: int, suffix: str = "e2e_test") -> Path:
        filename = f"{number:03d}_{suffix}.md"
        path = PROJECT_ROOT / "00_CHRONICLE" / "ENTRIES" / filename
        e2e_cleanup.register(path)
        return path
    
    return _create_path


@pytest.fixture
def safe_test_file(e2e_cleanup):
    """
    Fixture that provides a safe file path in project root for testing.
    Returns a path that will be auto-cleaned after the test.
    """
    def _create_path(filename: str) -> Path:
        path = PROJECT_ROOT / filename
        e2e_cleanup.register(path)
        return path
    
    return _create_path


# Known E2E test patterns for global cleanup (emergency use only)
E2E_ARTIFACT_PATTERNS = {
    "ADRs": ["*e2e_test*.md", "*_999_*.md", "*E2E_TEST*.md"],
    "00_CHRONICLE/ENTRIES": ["*e2e_test*.md", "*E2E_Test*.md"],
}


def cleanup_stale_e2e_artifacts() -> dict:
    """
    Emergency cleanup function for stale E2E artifacts.
    
    This is NOT run automatically - only call manually if tests 
    crashed without cleanup.
    
    Usage:
        python -c "from tests.mcp_servers.base.cleanup_fixtures import cleanup_stale_e2e_artifacts; cleanup_stale_e2e_artifacts()"
    """
    print("🚨 Emergency E2E artifact cleanup...")
    stats = {"files_deleted": 0}
    
    for rel_dir, patterns in E2E_ARTIFACT_PATTERNS.items():
        directory = PROJECT_ROOT / rel_dir
        if not directory.exists():
            continue
        
        for pattern in patterns:
            for match in directory.glob(pattern):
                if match.is_file():
                    try:
                        print(f"   Removing: {match}")
                        os.remove(match)
                        stats["files_deleted"] += 1
                    except Exception as e:
                        print(f"   Failed: {e}")
    
    print(f"✅ Cleaned {stats['files_deleted']} stale artifacts")
    return stats
