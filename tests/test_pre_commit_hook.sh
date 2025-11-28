#!/bin/bash
# Tests for Pre-Commit Hook Migration (Task #028)

echo "=== Testing Pre-Commit Hook Migration ==="

# Setup
TEST_FILE="test_mcp_migration.txt"
echo "test content" > "$TEST_FILE"
git add "$TEST_FILE"

# Test 1: Legacy Commit WITHOUT Manifest (Should FAIL)
echo -n "Test 1: Legacy Commit (No Manifest)... "
if git commit -m "legacy: test commit" > /dev/null 2>&1; then
    echo "FAILED (Should have been rejected)"
    exit 1
else
    echo "PASSED (Rejected as expected)"
fi

# Test 2: MCP Commit WITHOUT Env Var (Should FAIL)
echo -n "Test 2: MCP Commit (No Env Var)... "
if git commit -m "mcp(test): should fail" > /dev/null 2>&1; then
    echo "FAILED (Should have been rejected)"
    exit 1
else
    echo "PASSED (Rejected as expected)"
fi

# Test 3: MCP Commit WITH Env Var (Should PASS)
echo -n "Test 3: MCP Commit (With IS_MCP_AGENT=1)... "
if IS_MCP_AGENT=1 git commit -m "mcp(test): verification commit" > /dev/null 2>&1; then
    echo "PASSED"
else
    echo "FAILED (Should have been accepted)"
    exit 1
fi

# Cleanup
git reset --soft HEAD~1
rm "$TEST_FILE"
git reset HEAD "$TEST_FILE"

echo "=== All Tests Passed ==="
exit 0
