# Project Sanctuary Installation & Lifecycle Guide

> **Central installation, dependency resolution, and runtime onboarding guide for Project Sanctuary.**

This document explains how to install and manage skills in Project Sanctuary using [APM (Agent Package Manager)](https://github.com/microsoft/apm), covering both **project-local installation** (self-contained repository) and **global workstation installation**, alongside upstream capability dependencies from [`richfrem/agent-plugins-skills`](https://github.com/richfrem/agent-plugins-skills).

---

## 1. Architecture & Installation Scopes

Project Sanctuary provides project-specific capabilities via [`plugins/sanctuary-core`](plugins/sanctuary-core/README.md) and consumes Level 1 utility skills from `richfrem/agent-plugins-skills`.

You can install these skills under two supported models:

| Scope | Location | Command Flags | Best For |
| :--- | :--- | :--- | :--- |
| **Project-Local (Recommended)** | `.agents/`, `.claude/`, `.codex/` in project root | `apm install ...` (no `--global`) | Standalone repository work, reproducible CI/CD, isolated environments. |
| **Global Workstation** | `~/.agents/`, `~/.claude/`, `~/.gemini/` in home dir | `apm install ... --global` | Multi-repo developers wanting skills available across all workspaces. |

```mermaid
graph TD
    subgraph Upstream["Upstream: richfrem/agent-plugins-skills"]
        Memory["agent-memory"]
        Orch["agent-orchestration"]
        Dev["dev-utils"]
        Auth["agent-extension-authoring"]
    end

    subgraph SanctuaryRepo["Project Sanctuary (Local Repo)"]
        Core["plugins/sanctuary-core<br/>(chronicle, protocol, guardian, forge, traces)"]
        RootManifest["apm.yml (Project Manifest)"]
        HarnessLocal[".agents/skills/ & .claude/skills/<br/>(Project-Local APM Harness)"]
    end

    subgraph GlobalHarness["Workstation Global Profile"]
        HarnessGlobal["~/.agents/skills/ & ~/.claude/skills/"]
    end

    Upstream -->|apm install --global| HarnessGlobal
    Upstream -->|apm install (local)| HarnessLocal
    Core -->|apm install ./plugins/sanctuary-core| HarnessLocal
    Core -->|apm install $(pwd)/plugins/sanctuary-core --global| HarnessGlobal
```

---

## 2. Prerequisites: Install APM CLI

Ensure you have APM installed on your workstation (version 0.35.0 or newer is required for `agy` Antigravity target):

### On macOS / Linux:
```bash
curl -sSL https://aka.ms/apm-unix | sh -s -- @v0.35.0
```

### On Windows (PowerShell):
```powershell
irm https://aka.ms/apm-windows | iex
```

Verify installation:
```bash
apm --version
```

---

## 3. Option A: Project-Local Installation (Recommended)

Project-local installation places materialized, dereferenced copies of the skills directly into your local agent harness folders (`.agents/skills/`, `.claude/skills/`, `.codex/`), ensuring the repository is fully self-contained.

### 1. Install Sanctuary Core Skills Locally

To install all 7 sovereign skills (`managing-chronicles`, `managing-protocols`, `operating-sanctuary-guardian`, `forging-sovereign-models`, `preserving-soul-traces`, `managing-sovereign-memory`, `evaluating-agent-work`) into this project's client targets:

```bash
# From Project_Sanctuary root:
apm install ./plugins/sanctuary-core --target claude,copilot,codex,agy
```

Or run the single-command project resolver using the repository's root [`apm.yml`](apm.yml):

```bash
# Reads apm.yml and installs all declared plugin dependencies locally
apm install --target claude,copilot,codex,agy
```

### 2. Install Upstream Dependencies into Project Scope

If you do not have the global skills installed on your machine, install them directly into this repository:

```bash
# Install all upstream skills locally into .agents/ and .claude/
apm install https://github.com/richfrem/agent-plugins-skills --target claude,copilot,codex,agy
```

Or install specific required plugins into project scope:
```bash
# Memory & RLM cache
apm install richfrem/agent-plugins-skills/plugins/agent-memory --target claude,copilot,codex,agy

# Hugging Face transport & dev utilities
apm install richfrem/agent-plugins-skills/plugins/dev-utils --target claude,copilot,codex,agy

# Multi-agent orchestration
apm install richfrem/agent-plugins-skills/plugins/agent-orchestration --target claude,copilot,codex,agy
```

### 3. Verify Project-Local Installation

Verify what APM installed in this repository:
```bash
apm deps list
```

You should see `./plugins/sanctuary-core` and the installed skill directories populated in `.agents/skills/` and `.claude/skills/`.

### Keep the installed copies current

Project-local installs are **copies**, not links. After you edit `plugins/sanctuary-core/` (or pull changes to it), run `apm install` again at the repo root (`apm install --dry-run` previews it; it touches only `./plugins/sanctuary-core`). Until you do, `.claude/skills/` and `.agents/skills/` keep the old scripts, and agents that run a skill from there use the old code and the old paths. The session wakeup (`guardian_wakeup.py`) prints an `installed-skills` line that turns to `WARN` when the copies differ from the source and names this fix. The install copies each skill's whole folder, **tests included**, into `.claude/skills/`, `.agents/skills/` and `apm_modules/_local/`; the local-agent sandbox hides the held-out evaluation generator by name so those copies cannot leak it (see `references/local-agent-execution-loop.md`).

---

## 4. Option B: Global Workstation Installation

If you prefer having the skills globally available across all projects on your machine, install them with the `--global` flag. Global skills are recorded in `~/.apm/apm.yml` and placed in `~/.agents/skills/` and `~/.claude/skills/`.

### 1. Install Upstream Skills Globally

```bash
# Install all upstream plugins globally (one command)
apm install https://github.com/richfrem/agent-plugins-skills --global --target claude,copilot,codex,agy
```

### 2. Register Sanctuary Core Globally (Optional)

To make Sanctuary's core skills available globally on your workstation, pass the **absolute path** to `plugins/sanctuary-core`:

```bash
# macOS / Linux (from Project_Sanctuary root):
apm install "$(pwd)/plugins/sanctuary-core" --global --target claude,copilot,codex,agy

# Windows (PowerShell):
apm install "$((Get-Location).Path)\plugins\sanctuary-core" --global --target claude,copilot,codex,agy
```

Verify global dependencies:
```bash
apm deps list --global
```

> [!NOTE]
> **One Scope Rule**: If a skill or plugin is installed globally, do not also install it in the local project scope, or it will be registered twice in your client session. Check `apm deps list --global` before choosing your installation scope.

---

## 5. Verification & Telemetry Probes

After installing (either locally or globally), run the automated verification suite to confirm that all plugins, skills, and subsystems are functional:

```bash
# 1. Run Guardian Holistic Wakeup (verifies context, protocols, and living chronicle)
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC

# 2. Verify Living Chronicle ledger and fast search
python3 plugins/sanctuary-core/.apm/skills/managing-chronicles/scripts/chronicle_manager.py list --limit 5

# 3. Verify Protocol governance status
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py list --limit 5

# 4. Verify Soul Traces dataset workspace & Hugging Face integration
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/sync_soul_dataset.py --check-readiness

# 5. Verify Sovereign Memory progressive disclosure
python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --tier 1

# 6. Verify Cognitive work evaluation & token budget governance
python3 plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/track_budget.py status

# 7. Run Plugin Open-Standard Compliance Audits
python3 ~/.gemini/antigravity-cli/skills/authoring-plugins/scripts/audit.py --path plugins/sanctuary-core
python3 ~/.gemini/antigravity-cli/skills/packaging-with-apm/scripts/validate_apm_package.py --path plugins/sanctuary-core
```

Expected output:
- `guardian_wakeup.py`: Status `GREEN - Nominal`, boot time `< 10ms`.
- `sync_soul_dataset.py`: `Dataset Check: OK (huggingface)`, `HF Skill Integration: OK`.
- Plugin Audits: `AUDIT PASSED - Fully Open Standard Compliant` and `VALIDATION PASSED`.

---

## 6. Optional: ML Environments (Apple Silicon MLX & Phoenix Forge)

### 1. Apple Silicon (M1/M2/M3/M4) Local Sovereign Inference via Apple MLX
For native, offline execution on macOS Apple Silicon unified memory without swap thrashing:

- **Requirements**: Python >= 3.11, Apple Silicon (arm64), `mlx`, `mlx-lm`
- **Recommended Model**: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (~4.3 GB footprint; fits within 6 GB RAM with zero swap)
- **Alternative General Model**: `mlx-community/Qwen3-8B-4bit` (~4.6 GB footprint)

```bash
# 1. Test offline mlx-lm inference with benchmark metrics
uvx --from mlx-lm mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --prompt "What is the capital of France?" \
  --max-tokens 50 \
  --verbose True
```

Expected Output:
```text
==========
The capital of France is Paris.
==========
Prompt: 36 tokens, ~37 tokens-per-sec
Generation: 8 tokens, ~14 tokens-per-sec
Peak memory: ~4.43 GB
```

```bash
# 2. Clean output mode (answers directly without verbose metrics)
uvx --from mlx-lm mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --prompt "What is the capital of France?"
```
*(Note: The first run downloads the ~4.3 GB weights to `~/.cache/huggingface/hub`; all subsequent runs load instantly from local disk with 0 network calls and 0 cloud tokens).*

#### How to Check That the Local Model Is Running & Ready

**A. Check model cache & engine status via Sanctuary Guardian CLI:**
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py status
```

**B. Run offline Protocol 128 Pre-Completion Verification test:**
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py verify
```

**C. Launch persistent OpenAI-compatible local server (zero reload latency):**
```bash
uvx --from mlx-lm mlx_lm.server \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --port 8080
```

**D. Verify the running server from browser or terminal:**
- In browser: visit `http://127.0.0.1:8080/v1/models`
- In terminal:
```bash
curl -s http://127.0.0.1:8080/v1/models
```

**E. How to Ask Questions to the Running Server (3 Methods):**

*Method 1: Via Sanctuary Guardian CLI (auto-detects local server on port 8080)*:
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py query "In one sentence, what is Python?"
```

*Method 2: Via Direct HTTP curl (OpenAI-compatible `/v1/chat/completions`)*:
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
  }'
```

*Method 3: Via Interactive Multi-Turn Terminal Chat*:
```bash
uvx --from mlx-lm mlx_lm.chat \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

### 2. Windows WSL2/CUDA ML Training Environment (Phoenix Forge)
If you intend to run local model fine-tuning (`Sanctuary-Qwen2-7B-v1.0`), prepare the isolated CUDA ML environment following the surgical instructions preserved in [`forging-sovereign-models`](plugins/sanctuary-core/.apm/skills/forging-sovereign-models/references/windows-wsl-cuda-setup.md):

```bash
# Verify GPU and ML dependency stack readiness
python3 plugins/sanctuary-core/.apm/skills/forging-sovereign-models/scripts/verify_ml_environment.py

# Preview fine-tuning dataset assembly
python3 plugins/sanctuary-core/.apm/skills/forging-sovereign-models/scripts/forge_dataset_builder.py --limit 10
```

---

## 7. Troubleshooting

- **Target not supported**: Ensure APM is >= 0.35.0 for the `agy` (Google Antigravity) target.
- **`managing-huggingface-assets` not found**: If installed globally, verify `~/.gemini/antigravity-cli/skills/managing-huggingface-assets` exists. Alternatively, set `export MANAGING_HF_SKILL_PATH=/path/to/managing-huggingface-assets/scripts`.
- **Permission denied on scripts**: Ensure all script binaries have executable permissions:
  ```bash
  chmod +x plugins/sanctuary-core/.apm/skills/*/scripts/*.py
  ```
- **Hugging Face Authentication**: Run the credential validation tool:
  ```bash
  python3 ~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts/hf_init.py --validate-only
  ```
