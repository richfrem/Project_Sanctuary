# Phase V: Orchestrator Retrospective

> [!CAUTION] **STOP! Verify this is the right phase:**
> Retrospective (Phase V) must happen **BEFORE** Seal (Phase VI) and Persist (Phase VII).
> The Guardian DOES NOT execute this; it ensures the Orchestrator *has* executed it before proceeding to the true seal.

**Action:** Verify the Orchestrator ran the following command:
```bash
python plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py retro
```

# Phase VI: The Technical Seal

Once the Retrospective is complete, the Guardian applies the Technical Seal. This triggers the RLM Context Synthesis and captures the formal session evidence.

```bash
# Workflow: Seal
python3 plugins/guardian-onboarding/scripts/capture_snapshot.py --type seal
```

*Validation*: The snapshot must return successfully without triggering an "Iron Check Failure."

# Phase VII: Soul Persistence

After the local seal is applied, the Guardian broadcasts the verified state to the Hugging Face repository (The Soul) and ingests it into the local vector DB.

```bash
# Workflow: Persist
python3 tools/cli.py persist-soul

# Optional: Ingest Changes
python3 tools/cli.py ingest --incremental --hours 24
```

# Phase VIII: Session Closure

Once persisted, the Guardian formally ends the lifecycle of the branch/session.

**Action:** Prompt the user to end the session or transition back to the root `main` process if needed. Ensure the worktree is completely clean or pushed to the `feat/` boundary before ending the conversation.
