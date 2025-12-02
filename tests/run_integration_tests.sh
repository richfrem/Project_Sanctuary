#!/bin/bash

# Run Integration Tests
echo "Running Integration Tests..."
python3 -m pytest tests/integration -v -m integration

# Run Benchmarks (if pytest-benchmark is installed)
if pip show pytest-benchmark > /dev/null; then
    echo "\nRunning Performance Benchmarks..."
    python3 -m pytest tests/benchmarks -v -m benchmark --benchmark-only
else
    echo "\nSkipping benchmarks (pytest-benchmark not installed)"
    echo "Install with: pip install pytest-benchmark"
fi
