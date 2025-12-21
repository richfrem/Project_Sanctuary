#!/bin/bash
# scripts/wait_for_pulse.sh
# Checks for 'pulse' (health) of key fleet services before orchestration proceeds.
# Refer: ADR 065 v1.3

# Config
MAX_RETRIES=15
BACKOFF=3

# Helper function for retrying curls
wait_for_url() {
    local url=$1
    local name=$2
    local attempt=1

    echo -n "   - Checking $name ($url)... "
    
    while [ $attempt -le $MAX_RETRIES ]; do
        if curl -s -f -o /dev/null "$url"; then
            echo "✅ OK"
            return 0
        fi
        
        # Simple progress indicator
        echo -n "."
        sleep $BACKOFF
        ((attempt++))
    done
    
    echo " ❌ TIMEOUT after $((MAX_RETRIES * BACKOFF))s"
    return 1
}

# 1. Critical Backend: Vector DB (Port 8110)
wait_for_url "http://localhost:8110/api/v2/heartbeat" "Vector DB" || exit 1

# 2. Critical Backend: Ollama (Port 11434)
# Note: This might take longer if pulling models
wait_for_url "http://localhost:11434/api/tags" "Ollama" || exit 1

echo "   ✨ Fleet Pulse Detected."
exit 0
