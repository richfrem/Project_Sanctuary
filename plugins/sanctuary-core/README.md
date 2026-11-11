# Sanctuary Core (`sanctuary-core`)

The consolidated APM-native core plugin for **Project Sanctuary**, providing governance, Living Chronicle journaling, and protocol lifecycle management.

## Directory Structure

```
sanctuary-core/
├── apm.yml
├── README.md
├── .claude-plugin/
│   └── plugin.json
└── .apm/
    └── skills/
        ├── managing-chronicles/
        │   ├── SKILL.md
        │   ├── scripts/chronicle_manager.py
        │   └── references/
        ├── managing-protocols/
        │   ├── SKILL.md
        │   ├── scripts/protocol_manager.py
        │   └── references/
        ├── operating-sanctuary-guardian/
        │   ├── SKILL.md
        │   ├── scripts/
        │   │   ├── guardian_wakeup.py
        │   │   ├── capture_snapshot.py
        │   │   ├── learning_debrief.py
        │   │   ├── persist_soul.py
        │   │   └── forge_soul.py
        │   └── references/
        ├── forging-sovereign-models/
        │   ├── SKILL.md
        │   ├── scripts/
        │   │   ├── verify_ml_environment.py
        │   │   ├── forge_dataset_builder.py
        │   │   └── generate_modelfile.py
        │   └── references/
        ├── preserving-soul-traces/
        │   ├── SKILL.md
        │   ├── scripts/
        │   │   ├── trace_extractor.py
        │   └── references/
        ├── managing-sovereign-memory/
        │   ├── SKILL.md
        │   ├── scripts/
        │   │   ├── disclose.py
        │   │   ├── record_insight.py
        │   │   ├── record_procedure_outcome.py
        │   │   ├── audit_decay.py
        │   │   └── query_graph.py
        │   └── references/
        └── evaluating-agent-work/
            ├── SKILL.md
            ├── scripts/
            │   ├── score_contribution.py
            │   ├── track_budget.py
            │   └── visualize_economy.py
            └── references/
```

## Features

- **`managing-chronicles`**: Comprehensive CRUD and search for the historical project journal (`chronicle/entries/` and `ledger.json`).
- **`managing-protocols`**: Formal governance document lifecycle management (`protocols/`).
- **`operating-sanctuary-guardian`**: Master Protocol 128 lifecycle controller (orientation boot, cognitive onboarding, technical seal verification, learning debrief, and soul persistence).
- **`forging-sovereign-models`**: Sovereign AI fine-tuning pipeline, Windows WSL2/CUDA runtime recipes, QLoRA execution, GGUF quantization, and Hugging Face artifact publishing.
- **`preserving-soul-traces`**: Cognitive lineage and memory preservation engine; extracts markdown sessions and documents into canonical JSONL datasets (`soul_traces.jsonl`) and coordinates Hugging Face dataset persistence via `managing-huggingface-assets`.
- **`managing-sovereign-memory`**: Zero-dependency, filesystem-native 3-layer memory engine. Provides progressive disclosure (Tier 1 index, Tier 2 summary, Tier 3 deep), procedural memory outcome tracking (`LEARNING/procedures/`), asymmetric persistence of friction across git rollbacks, 30-day confidence decay auditing, and associative graph traversal.
- **`operating-sanctuary-bot`**: Sanctuary Bot. One interface (`bot.py`) to set up and run approved automations on a schedule or on demand (one-off jobs, pipelines, priority backlogs, verify-and-fix sweeps) over a zero-token daemon, a cross-platform scheduler and a multi-step conductor with outcome checks, revert-and-retry and circuit breakers. Started by the human, never by bootstrap. Hash-approved remote preservation closes through generated-export/receipt PRs, existing Hugging Face transport and verified remote hashes before reporting completion.
- **`evaluating-agent-work`**: Multi-dimensional cognitive value scoring (outcome utility, learning depth, verification rigor, token economy, anti-Asch truth grounding), real-world subscription budget tracking (ChatGPT Plus, Claude Pro, Gemini Pro), and RL value analytics.

## Architecture

This plugin follows modern APM (Agent Package Manager) and late-2026 platform standards:
- **Zero Cross-Plugin Imports**: All resources, references, and scripts are self-contained.
- **Native Skill Tooling**: Capabilities are exposed as first-class skills without legacy `commands/` boilerplate.
- **Main-Thread Lifecycle**: Interactive governance is executed in the main conversation thread.
- **Shared Learning Loop**: Generic scoring, cost tiers, usage-ledger arithmetic, usage measurement and validity gates are implemented once in the companion `evaluating-agent-outcomes`; `evaluating-agent-work` keeps the Sanctuary profile, budget policy, storage, dashboards and mission records (per-function map: `evaluating-agent-work/references/responsibility-map.md`). Sanctuary consumes the companion `orchestrating-agent-work` (outcome loop), `cli-agent` (enforced launch controls) and `evaluating-agent-outcomes` (findings, gates, lessons); this plugin keeps only policy: `evaluating-agent-work/scripts/sanctuary_adapter.py` and the `eval_gate.py` compatibility adapter. Engineering delivery is recorded separately from empirical verdicts. See `evaluating-agent-work/references/shared-loop-adapter.md`.
- **Global APM Interoperability**: Storage and execution layers cleanly interface with globally installed APM tools (`managing-obsidian-vaults`, `memory-management`, `orchestrating-agent-work`, `cli-agent`, `evaluating-agent-outcomes`, `managing-huggingface-assets`).

## Prerequisite: companion plugin

This plugin requires the companion `agent-orchestration` plugin (`orchestrating-agent-work`, `cli-agent`, `evaluating-agent-outcomes`) to be installed globally; it is deliberately not a project dependency so no project-local copy shadows the global one. Without it `eval_gate.py` and `sanctuary_adapter.py` exit with "shared skill not found". Both repositories are maintained together: see Protocol 133 section 8.

## Installation

Install locally or globally via APM:

```bash
# Project installation
apm install plugins/sanctuary-core

# Global installation across targets (Claude, Antigravity, Copilot, Codex)
apm install -g --target claude,copilot,codex,antigravity plugins/sanctuary-core
```
