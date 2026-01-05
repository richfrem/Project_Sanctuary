#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure we run from project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
USE_REAL_LLM=false
for arg in "$@"
do
    if [ "$arg" == "-r" ] || [ "$arg" == "--real" ]; then
        USE_REAL_LLM=true
    fi
done

PYTEST_ARGS="-m integration -v $@"
if [ "$USE_REAL_LLM" = true ]; then
    echo -e "${GREEN}=== Running Integration Tests with REAL LLM ===${NC}"
else
    echo -e "${GREEN}=== Running Integration Tests with MOCKED LLM ===${NC}"
fi

# Re-construct args for pytest
FINAL_ARGS=""
for arg in "$@"
do
    if [ "$arg" == "-r" ] || [ "$arg" == "--real" ]; then
        FINAL_ARGS="$FINAL_ARGS --real-llm"
    else
        FINAL_ARGS="$FINAL_ARGS $arg"
    fi
done

echo "Running: pytest -m integration -v $FINAL_ARGS"
pytest -m integration -v $FINAL_ARGS
INTEGRATION_EXIT_CODE=$?

echo -e "\n${GREEN}=== Running Performance Benchmarks ===${NC}"
pytest -m benchmark --benchmark-only
BENCHMARK_EXIT_CODE=$?

echo -e "\n${GREEN}=== Summary ===${NC}"
if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo -e "Integration Tests: ${GREEN}PASSED${NC}"
else
    echo -e "Integration Tests: ${RED}FAILED${NC}"
fi

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo -e "Benchmarks: ${GREEN}PASSED${NC}"
else
    echo -e "Benchmarks: ${RED}FAILED${NC}"
fi

# Exit with failure if either failed
if [ $INTEGRATION_EXIT_CODE -ne 0 ] || [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0
