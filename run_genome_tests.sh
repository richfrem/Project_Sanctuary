#!/bin/bash

# run_genome_tests.sh
# Automated tests for genome updates - verifies both query types work after ingestion

echo "[TEST] Starting genome functionality tests..."

# Test 1: Natural Language Query (main.py)
echo "[TEST 1/2] Testing natural language query..."
if python3 mnemonic_cortex/app/main.py "What is the core principle of the Anvil Protocol?" > /dev/null 2>&1; then
    echo "[PASS] Natural language query test passed"
else
    echo "[FAIL] Natural language query test failed"
    exit 1
fi

# Test 2: Structured JSON Query (protocol_87_query.py)
echo "[TEST 2/2] Testing structured JSON query..."
TEST_QUERY='{
  "intent": "RETRIEVE",
  "scope": "Protocols",
  "constraints": "Name=\"P27: The Doctrine of Flawed, Winning Grace\"",
  "granularity": "ATOM",
  "requestor": "genome-test",
  "request_id": "genome-test-001"
}'

if echo "$TEST_QUERY" | python3 mnemonic_cortex/scripts/protocol_87_query.py /dev/stdin > /dev/null 2>&1; then
    echo "[PASS] Structured JSON query test passed"
else
    echo "[FAIL] Structured JSON query test failed"
    exit 1
fi

echo "[SUCCESS] All genome tests passed - system is functional"
exit 0