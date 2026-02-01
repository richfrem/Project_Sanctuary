# System Requirements

This tool requires specific system binaries and global Python packages to function.

## 1. Core Binaries
- **Python**: 3.8+
    - Check: `python --version`
- **Ollama**: AI Inference Server
    - Required for: `distiller.py`
    - Install: https://ollama.com/
    - Check: `ollama --version`

## 2. Global Python Tools
Install these in your environment *before* installing the tool's libraries.

- **pip**: Package Installer
    - Upgrade: `python -m pip install --upgrade pip`
- **pip-tools**: Dependency Management
    - Required for: Compiling `requirements.in` if you modify dependencies.
    - Install: `pip install pip-tools`

## 3. AI Models (Ollama)
The tool is optimized for specific models. You must pull them before running.

- **granite3.2:8b**: (Recommended) High-performance summarization.
    - Pull: `ollama pull granite3.2:8b`
    - Run: `ollama serve` (Keep this running in a separate terminal)
