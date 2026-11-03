# Podman Test Container

This is a simple Flask web app to verify Podman is working correctly before implementing the Task MCP server.

## Files

- `app.py` - Simple Flask hello world web app
- `Dockerfile` - Container definition
- `build.sh` - Build script with instructions

## Quick Start

### Build the Image

```bash
cd tests/podman
./build.sh
```

### Run in Podman Desktop (Visual)

1. Open **Podman Desktop**
2. Go to **Images** tab
3. Find `sanctuary-podman-test:latest`
4. Click the **▶️ play button**
5. Configure:
   - **Port mapping:** `5001:5001`
   - **Container name:** `sanctuary-test`
6. Click **Start Container**
7. Go to **Containers** tab
8. Click on `sanctuary-test`
9. Click **Open Browser** or visit: http://localhost:5001

### Run from Command Line

```bash
# Run container
podman run -d -p 5001:5001 --name sanctuary-test sanctuary-podman-test:latest

# View logs
podman logs sanctuary-test

# Stop container
podman stop sanctuary-test

# Remove container
podman rm sanctuary-test
```

## What You Should See

- **Browser:** A purple gradient page with "Podman Test Successful!" 🚀
- **Health endpoint:** http://localhost:5001/health returns JSON

## Verification Checklist

- [x] Podman installed (v5.7.0)
- [x] Podman machine running
- [ ] Image builds successfully
- [ ] Container runs in Podman Desktop
- [ ] Web page loads in browser
- [ ] Health endpoint responds

Once all checks pass, Podman is ready for Task MCP deployment! ✅
