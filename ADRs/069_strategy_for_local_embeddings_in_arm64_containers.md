# ADR 069: Strategy for Local Embeddings in ARM64 Containers

**Status:** Accepted
**Date:** 2025-12-22
**Author:** System Architect
**Deciders:** Red Team Consensus (User Architect, Grok, Claude, Gemini, Principal SRE)

---

## Context

The project has standardized on `NomicEmbeddings` with `inference_mode="local"` for RAG operations to remove the Ollama runtime dependency for embeddings. 

However, verifying this architecture in the Dockerized Fleet (Linux ARM64 container on MacOS host) revealed a critical issue:
- `gpt4all` (the backend for Nomic local) does not provide pre-built wheels for Linux ARM64.
- `pip install` fails to build the C++ backend from source correctly within standard Python containers, even with build tools present.
- This results in a missing `libllama.so` shared object, causing the container to crash on `cortex_query`.

**The Failure of "Build Anywhere":**
We are currently in a state where our Development environment (MacOS Host) and our Deployment environment (Docker Container) are dangerously divergent. The library works on the Host via a pre-built Darwin binary that simply does not exist for the Container's architecture. This necessitates a strategic change to ensure our containers are reproducible and robust, rather than relying on a fragile specific setup that only works on the developer's metal. 

- The Host environment (MacOS) works fine because it uses Darwin wheels.

## Research Verification (Source: Claude 4.5 / Red Team)
**Issue Identified:** Upstream architectural limitation in `gpt4all`.

**Root Cause Analysis:**
- **The "Magic Box" Problem:** `gpt4all` builds and publishes wheels for **Linux x86_64** and **MacOS Universal** (ARM64/x64). It does **NOT** publish wheels for **Linux ARM64**.
- **Evidence:** GitHub Issues #101 (Mar 2023), #748, #1872, and #2426 confirm requests for Linux ARM64 support have been unaddressed for >18 months. An August 2025 issue questioned if the project is maintained.

**Architectural Deep Dive (Podman on M1):**
- **Host:** MacOS M1 (ARM64). Uses Darwin universal2 wheel. Works.
- **Virtual Machine:** Podman on M1 runs a Fedora CoreOS VM. This VM is **Linux ARM64**.
- **Container:** Runs inside the VM as **Linux ARM64**.
- **The Failure:** When `pip install gpt4all` runs in this container:
    1. It looks for a Linux ARM64 wheel on PyPI. None exists.
    2. It falls back to Source Distribution (sdist).
    3. It attempts to build the C++ backend (`llama.cpp`) from source.
    4. This build fails silently or produces an incomplete artifact because standard containers lack the complex toolchain (cmake, vulkan SDK, specific flags) required.
    5. Runtime Result: `OSError: libllama.so not found`.

**Why Multi-Arch Emulation Doesn't Help:**
Using `qemu-user-static` allows running x86_64 containers on ARM, but we are running *native* ARM64 containers. The issue isn't the CPU architecture itself; it's the missing pre-compiled binary for that architecture in the PyPI package.

We need to decide how to support local embeddings in the containerized fleet preventing a "works on my machine" divergence.

## Decision

**We accept Path B: Strategic Pivot to HuggingFace.**

We will replace the `gpt4all`-bound `NomicEmbeddings` implementation with `HuggingFaceEmbeddings` (via `sentence-transformers`) using the same Nomic model weights (`nomic-ai/nomic-embed-text-v1.5`).

This decision prioritizes **Standardization** and **Reproducibility** over persistence of the specific `gpt4all` library.

**Red Team Consensus:**
- **Grok (Principal SRE):** "Path B is the clear winner... minimizes future toil... anti-pattern to maintain heroic custom builds."
- **Claude (Security Architect):** "Strongly Agree... 18-month unaddressed issues... this dependency is a liability... Path B restores 'Build Anywhere'."
- **Gemini (Agent):** "Robust architectural choice... eliminates the need for fragile manual builds."

## Consequences

- **Immediate Action:** Refactor `operations.py` to use `HuggingFaceEmbeddings`; update `requirements.txt`.
- **Migration Required:** The Vector Database (Chroma) must be re-ingested/miagrated as floating-point variances between backends will likely invalidate existing search indices.
- **Benefit:** Restores full "Build Anywhere" capability; containers will build reliably on ARM64 Linux without custom C++ toolchains.
- **Trade-off:** One-time migration cost (Development Time + Compute).

## Detailed Analysis & Red Team Findings (Updated)

### 1. The Discrepancy (Host vs. Container)
- **Host (MacOS ARM64):** Works seamlessly. `pip install gpt4all` fetches a universal Darwin wheel that includes the necessary libraries. `inspect_chroma.py` passes immediately.
- **Container (Linux ARM64):** Fails on execution.
  - Error: `OSError: .../libllama.so` missing.
- **Hypothesis Confirmed:** External intelligence and updated searches (e.g., PyPI files, GitHub issues) confirm that `gpt4all` v2.8.x **does not publish Linux ARM64 wheels** to PyPI. No 2025 releases address this; latest is Aug 2024 with only x86-64 and macOS wheels. Issues persist without official fixes.

### 2. Failed Fix Attempts
We attempted to force the build inside the container by providing the necessary compiler stack:
- **Experiment A:** `pip install` on `python:3.11-slim` (Failed: missing shared libs).
- **Experiment B:** `pip install` on `python:3.11` (Full Debian) + `apt-get install build-essential cmake git`.
  - **Result:** Build completed, but `exec ls` confirmed `libllama.so` was **still missing**.
  - **Conclusion:** The standard `pip install` process for the source tarball (sdist) either fails silently or creates a broken package structure where the loader cannot find the compiled binary. This aligns with known issues where ARM64 builds require specific flags or Vulkan for full functionality, but even basic CPU builds skip `libllama.so` generation in containers.

### 3. Proposed Resolution Paths (For Team Review, with Updates)

#### Path A: The "Red Team" Build (High Effort)
We must bypass `pip`'s implicit build and perform an explicit manual compilation in the Dockerfile:
```dockerfile
# Proposed Dockerfile Logic (Validated as Workaround)
FROM python:3.11

# Install build dependencies (updated: add libopenblas-dev for potential optimizations)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build gpt4all from source (includes llama.cpp) – use latest commit for 2025 fixes if any
RUN git clone --recurse-submodules https://github.com/nomic-ai/gpt4all.git \
    && cd gpt4all/gpt4all-backend \
    && mkdir build && cd build \
    && cmake .. \
    && cmake --build . --parallel \
    && cd ../../gpt4all-bindings/python \
    && pip install -e .

# Install nomic with local mode
RUN pip install nomic[local]

# Rest of your Dockerfile...
```
*Pros:* Keeps `nomic` ecosystem; works in testing per community reports.
*Cons:* Extremely fragile build process; increases image build time significantly; requires maintenance. No 2025 updates simplify this—still manual.

#### Path B: Strategic Pivot to HuggingFace (Medium Effort, Recommended)
Switch the embedding provider from `NomicEmbeddings` to `HuggingFaceEmbeddings` (using `sentence-transformers`).
- Use models like `nomic-ai/nomic-embed-text-v1.5` directly via HuggingFace, which has ARM64 support and pre-built wheels.
*Pros:* First-class support for ARM64 containers; standardized ecosystem; widely cached; no custom builds needed.
*Cons:* Requires code changes in `operations.py`; requires re-generating all vector embeddings (Database Migration). However, migration scripts can automate this.

#### Path C: The "Universal" Fix (Cross-Platform, Deprecated)
Investigate if we can copy the working `.dylib` from the Host (renamed) or cross-compile on the host, but this violates "build anywhere" principles and risks binary incompatibilities between macOS and Linux.

**Current Recommendation (Updated):** Evaluate Path B (Standardization) vs Path A (Persistence). With no ARM64 wheel progress in 2025, Path B minimizes future toil and aligns with SRE principles of using well-supported tools. If sticking with Nomic, commit to Path A but monitor GitHub for upstream fixes.

## Related Decisions & Impact
This decision necessitates a review or potential amendment of the following ADRs:
- **[ADR 002: Select Core Technology Stack](002_select_core_technology_stack.md)** - Re-evaluating `langchain-nomic` as a core dependency for containerized environments.
- **[ADR 006: Select Nomic Embed Text Embeddings](006_select_nomic_embed_text_embeddings.md)** - The *choice* of model is valid (Nomic), but the *implementation* (`gpt4all` binding) is the point of failure.
- **[ADR 012: Mnemonic Cortex Architecture](012_mnemonic_cortex_architecture.md)** - Cortex deployment strategy must account for ARM64 build constraints.
- **[ADR 024: RAG Database Population](024_rag_database_population_maintenance_architecture.md)** - Switching to HuggingFace (Path B) would require a migration/re-population strategy.
- **[ADR Cortex 004: Choice of Nomic Embed Text](cortex/004-choice-of-nomic-embed-text.md)** - Specific architectural binding may need to change from `nomic` lib to `sentence-transformers`.
