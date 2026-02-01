# System Requirements

This tool requires specific system binaries and global Python packages.

## 1. Core Binaries
- **Python**: 3.8+
    - Check: `python --version`
- **C++ Build Tools**: (Recommended)
    - Required by **ChromaDB** dependencies (wrapt, mmh3, etc.) on systems where pre-built wheels are unavailable.
    - Check documentation for your OS (e.g., `build-essential` on Linux, Visual Studio Build Tools on Windows).

## 2. Global Python Tools
Install these in your environment *before* installing the tool's libraries.

- **pip**:
    - Upgrade: `python -m pip install --upgrade pip`
- **pip-tools**:
    - **CRITICAL**: Required for reproducible dependency resolution (compiling `requirements.in`).
    - Install: `pip install pip-tools`

## 3. Storage
- **Disk Space**:
    - The Vector DB (Chroma) creates persistent files. Ensure at least 1GB free space for the database directory.
