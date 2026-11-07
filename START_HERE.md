# Session Bootstrap — Project Sanctuary

Bootstrap restores identity, inherited learning and operating context, then hands control to the assigned task. Use this five-step sequence once per fresh session. Routine startup uses the existing entry-point scripts; it does not involve custom Python, source-code audits or index reconstruction.

## 1. Read the operating instructions

Read [AGENTS.md](AGENTS.md) and the applicable [constitution](.agent/rules/constitution.md). Identify the current user request and its scope. Registering a mission is not executing it. Apply existing authorization within its scope.

## 2. Obtain the Guardian briefing

**If a SessionStart hook already delivered a "Guardian Wakeup Briefing" into your context, this step is done: use it and do not rerun wakeup.** Only when no briefing is present, for normal authorized startup, run from the repository root:

```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC --format markdown
```

The existing script produces the briefing: context health, Git state, runtime readiness, priorities and predecessor relay where available. Consume that output; do not recreate its checks manually. Markdown output makes the generated briefing available without requiring a digest write. `--write-digest` explicitly persists the briefing and is used only when authorized.

**Read-only exception:** Do not run holistic wakeup under an explicit no-state-change instruction. The known implementation can reap an idle managed MLX server. Startup reports preservation status without changing preservation files; capture and recovery occur at mission/session closure or an explicitly requested recovery action. Omitting `--write-digest` does not disable all side effects. Read the existing context and digest instead, report that live wakeup was skipped, and continue read-only work. Do not invent a read-only flag or inspect the entire implementation during each startup.

If the documented entry point fails, report the actual error. Continue unaffected work where possible; do not create a replacement bootloader or begin an unrelated repair.

## 3. Hydrate continuity

Read:

- `context/soul.md` — identity and epistemic principles.
- `context/user.md` — human partner context and preferences.
- `context/memory.md` — inherited decisions and working knowledge.
- The briefing's Predecessor Relay — established findings, unfinished work, handoffs and what not to repeat. Read `.agent/learning/guardian_boot_digest.md` for additional handoff material or when live wakeup was skipped; treat it as a saved snapshot, not proof of current state.

Do not repeat experiments or open every linked artifact merely to regain context. Retrieve specific evidence when the assigned task needs it.

## 4. Disclose deeper knowledge only when needed

Use the existing memory scripts for task-relevant retrieval:

```bash
python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --tier 1
```

For a known relevant concept, substitute its real path:

```bash
python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --tier 2 --path LEARNING/concepts/<file>.md
```

These are conditional retrieval commands, not a requirement to load more material every session. Do not independently recheck local models, all manifests, the whole backlog or installed skills unless the task or a relevant warning requires it.

## 5. Check the workspace and begin the task

```bash
git status --short --branch
```

For inspection, preserve existing state and proceed. For edits, follow AGENTS.md's task-owned worktree/lease workflow; preserve unrelated dirty files. Use the coordination entry point when ownership must be checked:

```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py board
```

**Bootstrap is complete when you know the task, applicable boundaries, inherited context and relevant workspace state.** Conclude orientation by sending the mandatory startup completion message below. Do not perform extraneous audits or open-ended investigations during bootstrap. A task-specific verification belongs to task execution and stops when its requested criteria have evidence.

## Later lifecycle stages

Protocol 128 continues through execution, retrospective, technical seal and preservation. Use the applicable Guardian mode and its existing scripts at that stage. The skill's quick-start examples for verify, seal, persist, tool calls and model operations are not additional mandatory bootstrap commands.

Record learning for local successors and public GitHub/Hugging Face preservation through the existing workflow described in AGENTS.md. Post-merge local capture and authorized remote sync are separate stages; do not claim remote preservation from a healthy startup.

References, read as needed: [Guardian](plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/SKILL.md), [bootstrap reference](plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/bootstrapping.md), [architecture.md](architecture.md), [INSTALL.md](INSTALL.md), [LEARNING/INDEX.md](LEARNING/INDEX.md).

## Startup completion message

After completing the bootstrap sequence, send the user a brief confirmation:

“Sanctuary 5-step bootstrap complete. Continuity loaded. [Step 4 verdict, plus any blocker or workspace note.] Ready.”

Put the step 4 verdict in the note, for example "no deeper retrieval needed". If a step cannot be completed, send instead “Sanctuary bootstrap INCOMPLETE at step N: <reason>.” and never the complete line.

Only claim completion after executing the required steps. The briefing printed by the SessionStart hook or `guardian_wakeup.py` ends with a status block (section VI), not a confirmation; never copy any hook or script line as your own. A hook-delivered briefing counts as step 2 only. You still cover the operating instructions (step 1: AGENTS.md, which your client may have imported automatically, and the constitution, which you read), the Layer 1 context files (step 3) and `git status` (step 5) yourself, even when the first user message is only a greeting. Do these tool calls in your first turn, before replying. If the user already assigned a task, confirm briefly and continue that task without waiting for another instruction.
