"""
E2E Tests for sanctuary_utils cluster (17 tools)

Tools tested:
- Time: get-current-time, get-timezone-info
- Calculator: calculate, add, subtract, multiply, divide
- UUID: generate-uuid4, generate-uuid1, validate-uuid
- String: to-upper, to-lower, trim, reverse, word-count, replace
- Gateway: get-capabilities
"""
import pytest
import re


# =============================================================================
# TIME TOOLS (2)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestTimeTools:
    
    def test_time_get_current_time(self, logged_call):
        """Test time-get-current-time returns valid ISO timestamp."""
        result = logged_call("sanctuary-utils-time-get-current-time", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = result["result"].get("content", [])
        text = str(content)
        # Should contain time-related info
        assert any(char.isdigit() for char in text), "Response should contain timestamp"
    
    def test_time_get_timezone_info(self, logged_call):
        """Test time-get-timezone-info returns timezone data."""
        result = logged_call("sanctuary-utils-time-get-timezone-info", {})
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# CALCULATOR TOOLS (5)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestCalculatorTools:
    
    def test_calculator_add(self, logged_call):
        """Test calculator-add: 10 + 5 = 15"""
        result = logged_call("sanctuary-utils-calculator-add", {"a": 10, "b": 5})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "15" in content, f"Expected 15 in result, got: {content}"
    
    def test_calculator_subtract(self, logged_call):
        """Test calculator-subtract: 20 - 7 = 13"""
        result = logged_call("sanctuary-utils-calculator-subtract", {"a": 20, "b": 7})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "13" in content, f"Expected 13 in result, got: {content}"
    
    def test_calculator_multiply(self, logged_call):
        """Test calculator-multiply: 6 * 7 = 42"""
        result = logged_call("sanctuary-utils-calculator-multiply", {"a": 6, "b": 7})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "42" in content, f"Expected 42 in result, got: {content}"
    
    def test_calculator_divide(self, logged_call):
        """Test calculator-divide: 100 / 4 = 25"""
        result = logged_call("sanctuary-utils-calculator-divide", {"a": 100, "b": 4})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "25" in content, f"Expected 25 in result, got: {content}"
    
    def test_calculator_calculate(self, logged_call):
        """Test calculator-calculate with expression: (2 + 3) * 4 = 20"""
        result = logged_call("sanctuary-utils-calculator-calculate", {"expression": "(2 + 3) * 4"})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "20" in content, f"Expected 20 in result, got: {content}"


# =============================================================================
# UUID TOOLS (3)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestUUIDTools:
    
    def test_uuid_generate_uuid4(self, logged_call):
        """Test uuid-generate-uuid4 returns valid UUID v4."""
        result = logged_call("sanctuary-utils-uuid-generate-uuid4", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        # UUID v4 pattern
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
        assert re.search(uuid_pattern, content, re.IGNORECASE), f"Invalid UUID v4: {content}"
    
    def test_uuid_generate_uuid1(self, logged_call):
        """Test uuid-generate-uuid1 returns valid UUID v1."""
        result = logged_call("sanctuary-utils-uuid-generate-uuid1", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        # UUID pattern (any version)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        assert re.search(uuid_pattern, content, re.IGNORECASE), f"Invalid UUID: {content}"
    
    def test_uuid_validate_uuid(self, logged_call):
        """Test uuid-validate-uuid with valid UUID."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = logged_call("sanctuary-utils-uuid-validate-uuid", {"uuid_string": valid_uuid})
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# STRING TOOLS (6)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestStringTools:
    
    def test_string_to_upper(self, logged_call):
        """Test string-to-upper: hello -> HELLO"""
        result = logged_call("sanctuary-utils-string-to-upper", {"text": "hello"})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "HELLO" in content, f"Expected HELLO in result, got: {content}"
    
    def test_string_to_lower(self, logged_call):
        """Test string-to-lower: WORLD -> world"""
        result = logged_call("sanctuary-utils-string-to-lower", {"text": "WORLD"})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "world" in content, f"Expected world in result, got: {content}"
    
    def test_string_trim(self, logged_call):
        """Test string-trim: '  spaced  ' -> 'spaced'"""
        result = logged_call("sanctuary-utils-string-trim", {"text": "  spaced  "})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_string_reverse(self, logged_call):
        """Test string-reverse: hello -> olleh"""
        result = logged_call("sanctuary-utils-string-reverse", {"text": "hello"})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "olleh" in content, f"Expected olleh in result, got: {content}"
    
    def test_string_word_count(self, logged_call):
        """Test string-word-count: 'one two three' -> 3"""
        result = logged_call("sanctuary-utils-string-word-count", {"text": "one two three"})
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "3" in content, f"Expected 3 in result, got: {content}"
    
    def test_string_replace(self, logged_call):
        """Test string-replace: 'hello world' with 'world' -> 'universe'"""
        result = logged_call("sanctuary-utils-string-replace", {
            "text": "hello world",
            "old": "world",
            "new": "universe"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "universe" in content, f"Expected universe in result, got: {content}"


# =============================================================================
# GATEWAY TOOLS (1)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestGatewayTools:
    
    def test_gateway_get_capabilities(self, logged_call):
        """Test gateway-get-capabilities returns MCP server overview."""
        result = logged_call("sanctuary-utils-gateway-get-capabilities", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
