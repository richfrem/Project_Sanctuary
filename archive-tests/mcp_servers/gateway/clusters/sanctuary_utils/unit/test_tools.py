"""
Unit tests for Sanctuary Utils Tools (Business Logic).
Tests the functional logic of Calculator, Time, UUID, and String tools.
"""
import pytest
from unittest.mock import patch, MagicMock
from mcp_servers.gateway.clusters.sanctuary_utils.tools import (
    calculator_tool,
    time_tool,
    uuid_tool,
    string_tool
)

class TestCalculatorTool:
    def test_calculate_safe(self):
        res = calculator_tool.calculate("2 + 2")
        assert res["success"] is True
        assert res["result"] == 4

    def test_calculate_math_funcs(self):
        res = calculator_tool.calculate("sqrt(16)")
        assert res["success"] is True
        assert res["result"] == 4.0

    def test_calculate_unsafe(self):
        res = calculator_tool.calculate("__import__('os').system('ls')")
        assert res["success"] is False
        assert "error" in res

    def test_basic_ops(self):
        assert calculator_tool.add(1, 2)["result"] == 3
        assert calculator_tool.subtract(5, 3)["result"] == 2
        assert calculator_tool.multiply(3, 4)["result"] == 12
        assert calculator_tool.divide(10, 2)["result"] == 5.0
        assert calculator_tool.divide(1, 0)["success"] is False


class TestTimeTool:
    def test_get_current_time(self):
        res = time_tool.get_current_time()
        assert res["success"] is True
        assert res["timezone"] == "UTC"
        assert "isoformat" in str(res["time"]) or "T" in res["time"]

    def test_get_timezone_info(self):
        res = time_tool.get_timezone_info()
        assert res["success"] is True
        assert "UTC" in res["available_timezones"]


class TestUUIDTool:
    def test_generate_uuid4(self):
        res = uuid_tool.generate_uuid4()
        assert res["success"] is True
        assert len(res["uuid"]) == 36
        assert res["version"] == 4

    def test_validate_uuid(self):
        # Valid
        u = uuid_tool.generate_uuid4()["uuid"]
        res = uuid_tool.validate_uuid(u)
        assert res["valid"] is True
        
        # Invalid
        res = uuid_tool.validate_uuid("not-a-uuid")
        assert res["valid"] is False


class TestStringTool:
    def test_transformations(self):
        assert string_tool.to_upper("abc")["result"] == "ABC"
        assert string_tool.to_lower("ABC")["result"] == "abc"
        assert string_tool.trim("  abc  ")["result"] == "abc"
        assert string_tool.reverse("abc")["result"] == "cba"

    def test_word_count(self):
        res = string_tool.word_count("hello world")
        assert res["word_count"] == 2
        assert res["char_count"] == 11
        assert res["char_count_no_spaces"] == 10

    def test_replace(self):
        res = string_tool.replace("hello world", "world", "friend")
        assert res["result"] == "hello friend"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
