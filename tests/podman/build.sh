#!/bin/bash
# Build and run Podman test container
# This verifies Podman is working before implementing Task MCP

set -e

echo "🚀 Building Podman Test Container..."
cd "$(dirname "$0")"

# Build the image
podman build -t sanctuary-podman-test:latest .

echo "✅ Image built successfully!"
echo ""
echo "📋 To run the container in Podman Desktop:"
echo ""
echo "1. Open Podman Desktop"
echo "2. Go to 'Images' tab"
echo "3. Find 'sanctuary-podman-test:latest'"
echo "4. Click the ▶️ play button"
echo "5. Configure:"
echo "   - Port mapping: 5001:5001 (or use any available port like 5003:5001)"
echo "   - Name: sanctuary-test"
echo "6. Click 'Start Container'"
echo "7. Open browser: http://localhost:5001 (or your chosen port)"
echo ""
echo "Or run from command line:"
echo "  podman run -d -p 5001:5001 --name sanctuary-test sanctuary-podman-test:latest"
echo "  # Or use a different host port:"
echo "  podman run -d -p 5003:5001 --name sanctuary-test sanctuary-podman-test:latest"
echo ""
echo "To view in browser: http://localhost:5001 (or http://localhost:5003 if you used that port)"
echo "To check health: http://localhost:5001/health"
