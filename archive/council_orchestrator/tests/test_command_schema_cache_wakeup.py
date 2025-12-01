# council_orchestrator/tests/test_command_schema_cache_wakeup.py
# Tests for cache_wakeup command schema validation

import pytest
import json
from council_orchestrator.orchestrator.commands import determine_command_type, validate_command


class TestCacheWakeupCommandSchema:
    """Test cache_wakeup command schema validation."""

    def test_cache_wakeup_command_type_detection(self):
        """Test that cache_wakeup commands are correctly identified."""
        command = {
            "task_type": "cache_wakeup",
            "task_description": "Test description",
            "output_artifact_path": "test_output.md"
        }

        command_type = determine_command_type(command)
        assert command_type == "CACHE_WAKEUP"

    def test_cache_wakeup_command_validation_passes(self):
        """Test that valid cache_wakeup commands pass validation."""
        command = {
            "task_type": "cache_wakeup",
            "task_description": "Guardian boot digest from cache",
            "output_artifact_path": "WORK_IN_PROGRESS/guardian_boot_digest.md",
            "config": {
                "bundle_names": ["chronicles", "protocols", "roadmap"],
                "max_items_per_bundle": 15
            }
        }

        is_valid, error_msg = validate_command(command)
        assert is_valid is True
        assert error_msg == "Command is valid"

    def test_cache_wakeup_command_validation_missing_task_type(self):
        """Test that cache_wakeup commands fail validation without task_type."""
        command = {
            "task_description": "Test description",
            "output_artifact_path": "test_output.md"
        }

        is_valid, error_msg = validate_command(command)
        assert is_valid is False
        assert "Missing required field 'task_type'" in error_msg

    def test_cache_wakeup_command_validation_wrong_task_type(self):
        """Test that commands with wrong task_type fail validation."""
        command = {
            "task_type": "wrong_type",
            "task_description": "Test description",
            "output_artifact_path": "test_output.md"
        }

        is_valid, error_msg = validate_command(command)
        assert is_valid is False
        assert "task_type must be 'cache_wakeup'" in error_msg

    def test_cache_wakeup_command_validation_missing_description(self):
        """Test that cache_wakeup commands fail without task_description."""
        command = {
            "task_type": "cache_wakeup",
            "output_artifact_path": "test_output.md"
        }

        is_valid, error_msg = validate_command(command)
        assert is_valid is False
        assert "Missing required field 'task_description'" in error_msg

    def test_cache_wakeup_command_validation_missing_output_path(self):
        """Test that cache_wakeup commands fail without output_artifact_path."""
        command = {
            "task_type": "cache_wakeup",
            "task_description": "Test description"
        }

        is_valid, error_msg = validate_command(command)
        assert is_valid is False
        assert "Missing required field 'output_artifact_path'" in error_msg