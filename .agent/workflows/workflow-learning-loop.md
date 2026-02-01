---
description: Orchestrates a Protocol 128 Learning Loop within the standard Spec-First lifecycle.
---
# Workflow: Learning Loop

> **Auto-Initialization**: This workflow automatically calls `/workflow-start` to handle Git Branching, Spec Folder creation, and Context Initialization. You do NOT need to run `/workflow-start` separately.

1. **Initialize Session**:
   // turbo
   python3 tools/cli.py workflow run --name workflow-start --target {topic}

2. **Phase I: Scout**:
   Run `/workflow-scout`.

3. **Define Goals (Spec)**:
   Run `/speckit-specify` to define the research objectives in `spec.md`.

4. **Phase II: Synthesis & Execution**:
   - Perform research (using `tools/retrieve/...`).
   - Create ADRs (using `/workflow-adr`).
   - Update Chronicles (using `/workflow-chronicle`).

5. **Phase IV: Audit**:
   Run `/workflow-audit` when ready for Red Team review.

6. **Phase V: Seal**:
   Run `/workflow-seal`.

7. **Phase VI: Persist**:
   Run `/workflow-persist`.

8. **Retrospective**:
   Run `/workflow-retrospective`.

9. **Closure**:
   Run `/workflow-end`.
