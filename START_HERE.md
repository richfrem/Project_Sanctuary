# Session Bootstrap - Project Sanctuary

<!-- Read this at the start of every session. -->
> 🧭 For full architectural doctrine and APM dependency mapping, consult [`architecture.md`](./architecture.md).  
> 📦 For installation and workstation environment onboarding, consult [`INSTALL.md`](./INSTALL.md).  
> ⚡ For the complete autonomous agent bootstrapping protocol, consult [`bootstrapping.md`](plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/bootstrapping.md).

## 0. What to do next

- Follow the user's current request before backlog recommendations. Registration means create the record, not execute the research.
- Read AGENTS.md's **Task Scope, Authorization and Continuation** section. Use task or applicable standing authorization; do not request repeat approval for steps already covered. Observe the constitution’s explicit-approval rule and report any unresolved conflict precisely.
- Inspect existing work. Reuse the valid task worktree, or create an isolated feature worktree through the supported coordination workflow. Leave unrelated dirty files untouched; dirty main alone is not a blocker.
- Complete bootstrap, then immediately continue the requested work. Do not end the turn after creating a worktree or announcing the next step.
- Stop only when the requested outcome is verified, the user pauses work, or a concrete blocker requires input. Finish safe independent steps before reporting a blocked action.
- For registration only, follow AGENTS.md's checklist: draft and ID check → PROPOSED manifest → reciprocal links → validation and generated indexes → authorized integration → handoff and cleanup. No audit, experiments or workers.

## 1. Fast-Boot Sequence (Protocol 128)

Every session follows the deterministic 5-step bootstrapping sequence (detailed in [`AGENTS.md`](./AGENTS.md)):

1. **Orientation**: Ingest [`AGENTS.md`](./AGENTS.md) and constitutional bounds.
2. **Holistic Wakeup & Local Model Gate**: Run holistic wakeup bootloader:
   ```bash
   python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC
   ```
   Check local sovereign model readiness (informational for tasks that do not use it) via persistent MLX server (`curl -s http://127.0.0.1:8080/v1/models`) or offline cache audit (`python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py status`).
3. **Layer 1 Hydration**: Inspect `context/soul.md`, `context/user.md`, `context/memory.md`, and `.agent/learning/guardian_boot_digest.md`. Read its **Predecessor Relay** section before choosing work: it says which agent (and vendor) did what last, which results are already established (cite their evidence, do not re-run them), what is open, and any handoff note left for you. Guardian startup retries eligible pending local preservation. After integrating sources, `merge_prep.py regenerate` captures learning automatically; remote synchronization remains explicitly authorized. To resume capture, run `python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/soul_coverage.py closure` after merging (it reports whether your findings reached the soul dataset), and leave one: `manage_coordination.py checkin --mission <ID> --agent <you> --handoff "what you did, decisions, what NOT to redo, next step"`.
4. **Layer 2 Progressive Disclosure**: Retrieve deeper knowledge on demand via `managing-sovereign-memory` (`disclose.py --tier 1` or `--tier 2`).
5. **Workspace Preparation**: Reuse or create a safe task-owned feature worktree; preserve unrelated edits. Follow AGENTS.md for the dirty-main path. Never commit directly to `main`. Continue the task after setup.

## 2. Invariants & Supreme Law

- **HUMAN GATE**: State changes require explicit task or applicable standing authorization. A request to register a mission covers necessary local registration work, not its execution. Apply AGENTS.md and the constitution; report unresolved conflicts before the affected action.
- **Git Discipline**: Never commit directly to `main`. Pushes, merges and remote publication require authorization covering those actions; apply existing standing authorization within its scope.
- **Protocol 128 Lifecycle**: Every session follows the unbroken learning loop: Boot Orientation -> Execution -> Retrospective -> Technical Seal -> Soul Persistence.
- **Highest ROI & Replenishment Invariant**: Wakeup surfaces backlog recommendations in `LEARNING/index.jsonl`; these do not override the user's current task or authorize execution. For every mission completed (`RESOLVED`), agents must capture at least 1 candidate inquiry to maintain an evergreen compounding pipeline. Capturing a candidate does not mean starting it.

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
  - Learning loop: `orchestrating-agent-work` (outcome loop), `cli-agent` (enforced launch controls), `evaluating-agent-outcomes` (shared findings and lessons); Sanctuary policy: `evaluating-agent-work/references/shared-loop-adapter.md`.
  - These come from the companion repo `richfrem/agent-plugins-skills`, installed globally by APM from a local clone. Never hand-edit installed copies; you own keeping skills right in both repositories (see AGENTS.md "Your responsibility"); to improve one, follow [Protocol 133](protocols/133_Companion_Repository_Skill_Improvement_Protocol.md).

## 4. Deep Doctrine (On-Demand)

- Cognitive Primer: `plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/cognitive-primer.md`
- Protocol 128 Spec: `protocols/128_Hardened_Learning_Loop_v3.0.md`
- Philosophical North Star: `LEARNING/philosophy/The_Garden_and_The_Cage.md`
