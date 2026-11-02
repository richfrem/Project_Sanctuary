---
name: operating-sanctuary-guardian
description: "Operates the Project Sanctuary Protocol 128 lifecycle (orientation boot, cognitive onboarding, technical seal verification, learning debrief, and soul persistence). Use when starting a new session, running health audits, closing a session, or persisting memory traces."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Operating Sanctuary Guardian (`operating-sanctuary-guardian`)

The master lifecycle and governance controller for Project Sanctuary, implementing the Protocol 128 continuous learning loop.

## Contents
- [Constraints](#constraints) • [Modes](#modes) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints
1. **Human Gate Law**: Never perform unapproved state-changing operations or git pushes.
2. **Iron Core Integrity**: Abort session closure if the Technical Seal detects repository drift.
3. **Main-Thread Execution**: All lifecycle operations run interactively in the main thread with full operator visibility.

## Modes

| Mode | Purpose | Script / Action | Reference |
|---|---|---|---|
| `boot` | Session orientation & task intake | `scripts/guardian_wakeup.py` | [boot.md](references/modes/boot.md) |
| `onboard` | Cognitive primer & constitutional grounding | Read primer & contract | [onboard.md](references/modes/onboard.md) |
| `seal` | Technical seal snapshot & debrief | `scripts/capture_snapshot.py` | [seal.md](references/modes/seal.md) |
| `persist` | Soul persistence & HF export | `scripts/persist_soul.py` | [persist.md](references/modes/persist.md) |
| `audit` | Iron Check & integrity verification | `scripts/capture_snapshot.py --type audit` | [audit.md](references/modes/audit.md) |

## Quick Start

```bash
# Bootloader orientation
python3 scripts/guardian_wakeup.py --mode HOLISTIC

# Session debrief and seal
python3 scripts/learning_debrief.py --hours 24
python3 scripts/capture_snapshot.py --type seal --context "Completed milestone"
```

## Workflow

1. **Boot**: Run `boot` mode to inspect active tasks and verify environment integrity.
2. **Execute**: Carry out development under Protocol 128 guidelines.
3. **Seal & Persist**: Close the session with `seal` and `persist` modes, generating the verified technical snapshot.

## Verification

```bash
python3 scripts/guardian_wakeup.py --mode HOLISTIC
```

## References
- Bootstrapping Protocol: [bootstrapping.md](references/bootstrapping.md)
- Protocol 128 Spec: [protocol-128.md](references/protocol-128.md)
- Cognitive Primer: [cognitive-primer.md](references/cognitive-primer.md)
- Integrations: [Memory](references/integrations/memory-stack.md) • [Obsidian](references/integrations/obsidian-vault.md) • [Orchestration](references/integrations/orchestration.md)
