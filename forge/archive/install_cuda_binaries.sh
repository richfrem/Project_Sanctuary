#!/bin/bash

# CUDA Binaries Installation Script for Project Sanctuary
# This script automates the installation of CUDA-specific binaries for the ML environment
# Run this after activating the ml_env virtual environment

LOG_FILE="cuda_binaries_install_$(date +%Y%m%d_%H%M%S).log"
echo "Starting CUDA binaries installation at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Function to run command and log output
run_and_log() {
    echo "Running: $*" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Command failed with exit code $exit_code" | tee -a "$LOG_FILE"
        return $exit_code
    fi
    echo "Command completed successfully" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

echo "Step 1: Uninstalling existing CUDA binaries..." | tee -a "$LOG_FILE"
run_and_log pip uninstall -y bitsandbytes triton xformers

echo "Step 2: Installing triton==3.5.0..." | tee -a "$LOG_FILE"
run_and_log pip install triton==3.5.0

echo "Step 3: Installing bitsandbytes-cuda126==0.43.1 with CUDA support..." | tee -a "$LOG_FILE"
run_and_log pip install --no-cache-dir bitsandbytes-cuda126==0.43.1 --no-deps

echo "Step 4: Installing xformers..." | tee -a "$LOG_FILE"
run_and_log pip install xformers

echo "Step 5: Fixing fsspec version conflict..." | tee -a "$LOG_FILE"
run_and_log pip install "fsspec<=2024.3.1"

echo "CUDA binaries installation completed at $(date)" | tee -a "$LOG_FILE"
echo "Check the log file for details: $LOG_FILE" | tee -a "$LOG_FILE"