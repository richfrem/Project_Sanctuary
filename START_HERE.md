# Session Bootstrap - Project Sanctuary

<!-- Read this at the start of every session. -->

## 1. Fast-Boot Context & Orientation

1. **Identity & Prime Directives**: Read `context/soul.md` (Guardian fast-boot identity).
2. **Human Partner Context**: Read `context/user.md` (Richard's working style & preferences).
3. **Institutional Memory**: Read `context/memory.md` (recent architectural decisions).
4. **Boot Telemetry**: Run `python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC`.

## 2. Invariants & Supreme Law

- **THE SUPREME LAW: HUMAN GATE**: Never execute ANY state-changing operation (writes, git mutations) without EXPLICIT user approval ("Proceed", "Go", "Execute").
- **Zero-Trust Git Discipline**: Never commit directly to `main`, and NEVER run `git push` without fresh, explicit human instruction.
- **Protocol 128 Lifecycle**: Every session follows the unbroken learning loop: Boot Orientation -> Execution -> Retrospective -> Technical Seal -> Soul Persistence.

## 3. Core Capabilities (APM Ecosystem)

- **Project Core (`plugins/sanctuary-core`)**:
  - `managing-chronicles`: Journaling in `00_CHRONICLE/ENTRIES/`.
  - `managing-protocols`: Governance lifecycle in `01_PROTOCOLS/`.
  - `operating-sanctuary-guardian`: Protocol 128 lifecycle (`boot`, `onboard`, `seal`, `persist`, `audit`).
- **Global APM Capabilities**:
  - Memory & Knowledge: `memory-management`, `managing-rlm-cache`, `managing-obsidian-vaults`.
  - Multi-Agent Orchestration: `orchestrating-agent-work`, `cli-agent`, `running-control-plane`.
  - Soul Persistence: `managing-huggingface-assets`.

## 4. Deep Doctrine (On-Demand)

- Cognitive Primer: `plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/cognitive-primer.md`
- Protocol 128 Spec: `plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/protocol-128.md`
- Philosophical North Star: `06_THE_EMBER_LIBRARY/The_Garden_and_The_Cage.md`
