---
name: session-bootloader
description: Initializes and orients the agent session using the Protocol 128 Bootloader sequence. Trigger this at the start of any new assignment.
disable-model-invocation: false
---

# Session Bootloader (Protocol 128 Phase I)

You are responsible for executing the mandatory **Learning Scout** and **Initialization** sequence when a new session or workflow begins.

## Core Directives
1. **Never skip orientation**: You must establish context before planning or writing code.
2. **Constitutional Enforcement**: Execution must follow the project's zero-trust constitution.
3. **Orchestrator Handoff**: Once oriented and initialized, you must hand off execution to the `orchestrator` skill.

## Plugin Dependencies
| Plugin/Skill | Role |
|:---|:---|
| `plugins/guardian-onboarding/scripts/learning_debrief.py` | Phase I: Generates the Truth Anchor (snapshot) |
| `plugins/guardian-onboarding/scripts/guardian_wakeup.py` | Phase I: Runs Iron Check & produces boot digest |
| `plugins/agent-loops/skills/orchestrator/` | Phase II+: Receives control after boot is complete |
| `plugins/rlm-factory/` | Underlying RLM cache that `guardian_wakeup.py` reads |

---

## Phase Execution Steps

### 1. The Learning Scout (Debrief & Orientation)
Execute the tools required to acquire the project's current baseline state.
```bash
# Generate the latest debrief
python3 plugins/guardian-onboarding/scripts/learning_debrief.py --hours 24
```
**Action:** The previous command will output a path to `learning_package_snapshot.md`. You MUST read this file using `view_file` to establish the Truth Anchor for your session.

```bash
# Run the Guardian Integrity Check
python3 plugins/guardian-onboarding/scripts/guardian_wakeup.py --mode TELEMETRY
```

### 2. The Constitutional Gate
Before any execution begins, ensure the intended work aligns with the rules of reality defined in `.agent/rules/constitution.md`.

*Verify:*
1.  **Article I (Human Gate)**: Are you authorized to make state changes?
2.  **Article V (Test-First)**: Is there a verification plan?
3.  **Article IV (Docs First)**: Is the defining Spec/Plan up to date?

### 3. Feature Spec & Branch Initialization
If the work requires creating a new feature, bug fix, or documentation update, you must ensure a proper Spec-Plan-Tasks bundle exists and you are on a dedicated feature branch.

**Reference:** Read `references/spec-initialization.md` to execute the correct branching and artifact logic.

### 4. Orchestrator Routing
Once the environment is initialized, the **Orchestrator** takes over to route the task to a specific execution pattern (Learning, Red Team, Dual-Loop, Swarm).

```bash
# Trigger the Orchestrator Assessment
python3 plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py scan --spec-dir .
```
*Note: Follow the Orchestrator's routing instructions once it assesses the target.*
