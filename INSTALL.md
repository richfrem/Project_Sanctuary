# Project Sanctuary Installation & Lifecycle Guide

> **Central installation, dependency resolution, and runtime onboarding guide for Project Sanctuary.**

This document explains how to bootstrap the Project Sanctuary workspace, install its required global APM skills from [`richfrem/agent-plugins-skills`](https://github.com/richfrem/agent-plugins-skills), register the local [`sanctuary-core`](plugins/sanctuary-core/README.md) plugin, and verify system integrity across macOS, Linux, and Windows WSL2.

---

## 1. Architecture & Dependency Flow

Project Sanctuary operates as a **Level 2 consuming repository**. It depends on Level 1 foundational skills installed globally from `agent-plugins-skills`, while managing its own domain capabilities through `plugins/sanctuary-core`.

```mermaid
graph TD
    A["GitHub: richfrem/agent-plugins-skills"] -->|apm install --global| B["Workstation Global Cache (~/.agents, ~/.claude, ~/.gemini)"]
    B --> C["Global Skills: memory-management, orchestrating-agent-work, managing-huggingface-assets, etc."]
    
    D["Local: Project_Sanctuary/plugins/sanctuary-core"] -->|apm install (local/global)| E["Core Skills: chronicles, protocols, guardian, forge, traces"]
    
    C --> F["Runtime Agent Session (Antigravity / Claude Code / Codex)"]
    E --> F
```

---

## 2. Step 1: Install APM (Agent Package Manager)

Ensure you have APM installed on your system (APM 0.35.0+ recommended):

### On macOS / Linux:
```bash
curl -sSL https://aka.ms/apm-unix | sh -s -- @v0.35.0
```

### On Windows (PowerShell):
```powershell
irm https://aka.ms/apm-windows | iex
```

Verify your APM installation:
```bash
apm --version
```

---

## 3. Step 2: Install Global Upstream Skills

Install all foundation plugins from the upstream repository directly into your agent profiles (`claude`, `copilot`, `codex`, `agy`):

```bash
# Install all six plugins globally from the upstream collection manifest
apm install https://github.com/richfrem/agent-plugins-skills --global --target claude,copilot,codex,agy
```

To install specific plugins individually:
```bash
# Memory & knowledge management (memory-management, managing-rlm-cache, managing-obsidian-vaults)
apm install richfrem/agent-plugins-skills/plugins/agent-memory --global --target claude,copilot,codex,agy

# Multi-agent orchestration (orchestrating-agent-work, cli-agent)
apm install richfrem/agent-plugins-skills/plugins/agent-orchestration --global --target claude,copilot,codex,agy

# Developer utilities & Hugging Face transport (managing-huggingface-assets, managing-adrs)
apm install richfrem/agent-plugins-skills/plugins/dev-utils --global --target claude,copilot,codex,agy

# Skill and plugin authoring tools (authoring-skills, authoring-plugins, packaging-with-apm)
apm install richfrem/agent-plugins-skills/plugins/agent-extension-authoring --global --target claude,copilot,codex,agy

# Agentic OS control plane (running-control-plane, setting-up-agentic-os)
apm install richfrem/agent-plugins-skills/plugins/agent-agentic-os --global --target claude,copilot,codex,agy
```

Verify globally installed dependencies:
```bash
apm deps list --global
```

---

## 4. Step 3: Register Local `sanctuary-core` Plugin

The Sanctuary-specific capabilities (`managing-chronicles`, `managing-protocols`, `operating-sanctuary-guardian`, `forging-sovereign-models`, `preserving-soul-traces`) live in [`plugins/sanctuary-core`](plugins/sanctuary-core/README.md).

From the repository root, install the plugin into your project or globally:

```bash
# From Project_Sanctuary root (absolute path recommended for APM)
apm install "$(pwd)/plugins/sanctuary-core" --target claude,codex,agy
```

---

## 5. Step 4: Verification & Fast-Boot Telemetry

Run the automated verification suite to confirm that all plugins, skills, and subsystems are functional:

```bash
# 1. Run Guardian Holistic Wakeup (verifies context, protocols, and living chronicle)
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC

# 2. Verify Living Chronicle ledger and fast search
python3 plugins/sanctuary-core/.apm/skills/managing-chronicles/scripts/chronicle_manager.py list --limit 5

# 3. Verify Protocol governance status
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py list --limit 5

# 4. Verify Soul Traces dataset workspace & Hugging Face integration
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/sync_soul_dataset.py --check-readiness

# 5. Run Plugin Open-Standard Compliance Audits
python3 ~/.gemini/antigravity-cli/skills/authoring-plugins/scripts/audit.py --path plugins/sanctuary-core
python3 ~/.gemini/antigravity-cli/skills/packaging-with-apm/scripts/validate_apm_package.py --path plugins/sanctuary-core
```

Expected output:
- `guardian_wakeup.py`: Status `GREEN - Nominal`, boot time `< 10ms`.
- `sync_soul_dataset.py`: `Dataset Check: OK (huggingface)`, `HF Skill Integration: OK`.
- Plugin Audits: `AUDIT PASSED - Fully Open Standard Compliant` and `VALIDATION PASSED`.

---

## 6. Optional: ML Training Environment (Phoenix Forge)

If you intend to run local model fine-tuning (`Sanctuary-Qwen2-7B-v1.0`), prepare the isolated CUDA ML environment following the surgical instructions preserved in [`forging-sovereign-models`](plugins/sanctuary-core/.apm/skills/forging-sovereign-models/references/windows-wsl-cuda-setup.md):

```bash
# Verify GPU and ML dependency stack readiness
python3 plugins/sanctuary-core/.apm/skills/forging-sovereign-models/scripts/verify_ml_environment.py

# Preview fine-tuning dataset assembly
python3 plugins/sanctuary-core/.apm/skills/forging-sovereign-models/scripts/forge_dataset_builder.py --limit 10
```

---

## 7. Troubleshooting

- **`managing-huggingface-assets` not found**: Ensure `richfrem/agent-plugins-skills/plugins/dev-utils` is installed globally via APM, or set `export MANAGING_HF_SKILL_PATH=~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts`.
- **Permission denied on scripts**: Ensure script binaries have executable permissions:
  ```bash
  chmod +x plugins/sanctuary-core/.apm/skills/*/scripts/*.py
  ```
- **Hugging Face Authentication**: Run the credential validation tool:
  ```bash
  python3 ~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts/hf_init.py --validate-only
  ```
