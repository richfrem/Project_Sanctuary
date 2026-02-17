# /agent-orchestrator_review — Context Bundling for Review

**Purpose:** This command packages the relevant technical context into a single, portable Markdown document. It is optimized for sharing with human reviewers or piping into specialized "Red Team" AI agents.

---

## 📦 The Bundling Workflow

1. **Self-Contained Review**:
   ```bash
   /agent-orchestrator_review --files file1.py file2.md spec.md plan.md
   ```
   *Action:* Leverages the `context-bundler` plugin to compile source, design, and status into a single file.
   *Output:* `.agent/reviews/review_bundle_TIMESTAMP.md`

2. **Persona-Based Peer Review (Optional)**:
   You can pipe this bundle to a specialized auditor (e.g., Security, QA):
   ```bash
   /claude-cli_run --persona auditor --file .agent/reviews/review_bundle.md
   ```

3. **Human Review Gate**:
   - Provide the bundle path to the user.
   - **Wait for Approval**: Feedback must be processed before the final merge.

---

## 🛠️ Closing the Loop
- **Approved**: Proceed to `/agent-orchestrator_retro`.
- **Revision Needed**: Return to `/agent-orchestrator_plan` (strategy change) or `/agent-orchestrator_delegate` (implementation correction).

> [!NOTE]
> Bundles preserve folder hierarchy and include contextual metadata, making them the "Hologram" of the current work package state.

---

# /agent-orchestrator_retro — Retrospective & Intelligence Sync

**Purpose:** This is the **CRITICAL FINAL STEP** of every session. We analyze the process, record technical debt, and ensure the system's "Long Term Memory" (RLM/Vector DB) is updated.

---

## 🧠 The Learning Loop

1. **Generate Retrospective Artifact**:
   ```bash
   /agent-orchestrator_retro
   ```
   *Action:* Creates a session-specific post-mortem document.

2. **Analyze The Session**:
   Identify patterns in:
   - **Handoff Quality**: Were the Strategy Packets clear?
   - **Correction Frequency**: Why did verification fail?
   - **SOP Efficiency**: Should the `codify-*` workflow be updated?

3. **The Boy Scout Rule (MANDATORY)**:
   > "Always leave the codebase cleaner than you found it."
   - You **MUST** fix at least one small issue (refactor a helper, update a template, clarify a rule) before closing.

4. **Intelligence Persistence**:
   - Run the sync pipeline to ingest new knowledge derived during the session.
   - `/inventory-manager_curate-inventories`

---

## 🏁 Finalizing the Feature
Once the retrospective is complete, proceed to:
```bash
/spec-kitty.merge
```

> [!TIP]
> Retrospectives are the fuel for the "AI Modernization" engine. They transform tribal knowledge into searchable documentation.
