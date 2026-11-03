# Session Bootstrap - Project Sanctuary

<!-- Read this at the start of every session. -->
> 🧭 For full architectural doctrine and APM dependency mapping, consult [`architecture.md`](./architecture.md).  
> 📦 For installation and workstation environment onboarding, consult [`INSTALL.md`](./INSTALL.md).  
> ⚡ For the complete autonomous agent bootstrapping protocol, consult [`bootstrapping.md`](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/bootstrapping.md).

## 1. Fast-Boot Sequence (Protocol 128)

Every session follows the deterministic 5-step bootstrapping sequence (detailed in [`AGENTS.md`](./AGENTS.md)):

1. **Orientation**: Ingest [`AGENTS.md`](./AGENTS.md) and constitutional bounds.
2. **Holistic Wakeup & Local Model Gate**: Run holistic wakeup bootloader:
   ```bash
   python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC
   ```
   Verify local sovereign model readiness via persistent MLX server (`curl -s http://127.0.0.1:8080/v1/models`) or offline cache audit (`python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py status`).
3. **Layer 1 Hydration**: Inspect `context/soul.md`, `context/user.md`, `context/memory.md`, and `.agent/learning/guardian_boot_digest.md`.
4. **Layer 2 Progressive Disclosure**: Retrieve deeper knowledge on demand via `managing-sovereign-memory` (`disclose.py --tier 1` or `--tier 2`).
5. **Git Discipline**: Verify working tree is clean and on a `feat/*` branch. Never commit directly to `main`.

## 2. Invariants & Supreme Law

- **THE SUPREME LAW: HUMAN GATE**: Never execute ANY state-changing operation (writes, git mutations) without EXPLICIT user approval ("Proceed", "Go", "Execute").
- **Zero-Trust Git Discipline**: Never commit directly to `main`, and NEVER run `git push` without fresh, explicit human instruction.
- **Protocol 128 Lifecycle**: Every session follows the unbroken learning loop: Boot Orientation -> Execution -> Retrospective -> Technical Seal -> Soul Persistence.
- **Highest ROI & Replenishment Invariant**: Wakeup orientation targets the highest-priority incomplete mission in `LEARNING/research/index.jsonl`. For every mission completed (`RESOLVED`), agents must create or generate at least 1 new research inquiry to maintain an evergreen compounding pipeline.

## 3. Core Capabilities (APM Ecosystem)

- **Project Core (`plugins/sanctuary-core`)**:
  - `managing-chronicles`: Journaling in `chronicle/entries/` and fast search via `ledger.json`.
  - `managing-protocols`: Governance lifecycle in `protocols/`.
  - `operating-sanctuary-guardian`: Protocol 128 lifecycle (`boot`, `onboard`, `seal`, `persist`, `audit`).
  - `forging-sovereign-models`: Sovereign AI model fine-tuning, WSL2/CUDA ML setup, and GGUF quantization.
  - `preserving-soul-traces`: Extracting, validating, and sealing JSONL memory traces into `huggingface/` and coordinating sync via `managing-huggingface-assets`.
  - `managing-sovereign-memory`: Progressive disclosure (index/summary/deep), agent-native cognitive backlog (`manage_backlog.py`), procedural memory outcome tracking (`LEARNING/procedures/`), asymmetric friction persistence across git rollbacks, 30-day confidence decay, and associative graph traversal.
  - `evaluating-agent-work`: Multi-dimensional cognitive value scoring, real-world subscription token budget tracking (ChatGPT Plus, Claude Pro, Gemini Pro), model cost-tier evaluation (cheap vs expensive models), and RL value analytics.
- **Global APM Capabilities**:
  - Memory & Knowledge: `memory-management`, `managing-rlm-cache`, `managing-obsidian-vaults`.
  - Multi-Agent Orchestration: `orchestrating-agent-work`, `cli-agent`, `running-control-plane`.
  - Soul Persistence: `managing-huggingface-assets`.

## 4. Deep Doctrine (On-Demand)

- Cognitive Primer: `plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/cognitive-primer.md`
- Protocol 128 Spec: `protocols/128_Hardened_Learning_Loop_v3.0.md`
- Philosophical North Star: `LEARNING/philosophy/The_Garden_and_The_Cage.md`
