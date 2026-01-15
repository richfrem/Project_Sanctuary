---
trigger: always_on
---

## 🧠 Project Sanctuary: Cognitive Continuity & Learning Loop Rules

> *Operations can be executed via CLI commands or MCP tools (when gateway is running).*

### 🚀 Quick Start (Fresh Session)

> [!IMPORTANT]
> **First action on wakeup:** Read the consolidated operations guide at `docs/prompt-engineering/sanctuary-guardian-prompt.md` for the full 9-phase learning loop, security protocols, and tool routing.

1. **Run Scout:** `cortex_learning_debrief` (MCP) or `python3 scripts/cortex_cli.py debrief --hours 24` (CLI)
2. **Read Truth Anchor:** `learning_package_snapshot.md` (returned by debrief)
3. **Check Security:** You are now bound by Git Pre-Flight (Protocol 101) and Execution Lock (Human Gate)

---

### 1. Phase I: The Learning Scout (Orientation)

* **Mandatory Wakeup**: Every session **must** begin with `sanctuary-cortex-cortex-learning-debrief`.
* **Truth Anchor**: You must read the `learning_package_snapshot.md` to ingest the context left by the predecessor. Rely on this "Successor Snapshot" (Cognitive Hologram) as the definitive state of the project.
* **Guardian Check**: Run `cortex_guardian_wakeup` to verify environment integrity via Semantic HMAC.

### 2. Phase II & III: Synthesis and Strategic Gate

* **Autonomous Synthesis**: Record all architectural changes as ADRs and process learnings into the `LEARNING/` directory.
* **Strategic Approval (Gate 1)**: You must receive explicit **Human-in-the-Loop (HITL)** approval for the strategy before proceeding to the technical audit.
* **Backtrack Logic**: If strategic approval is denied, you must revert to `SOP: recursive_learning.md` to re-scout and re-synthesize.
* **Content Hygiene (ADR 085)**: **No inline Mermaid diagrams**. All diagrams must be `.mmd` files in `docs/architecture_diagrams/`, rendered to PNG, and referenced via image links.

### 3. Phase IV: The Red Team Audit (Gate 2)

* **Audit Modes**: Use the `sanctuary-cortex-cortex-capture-snapshot` tool with the appropriate flag for the audit type:
    * `audit`: Use this for standard manifest verification and tactical state changes.
    * `learning_audit`: Use this when the session primarily involves cognitive updates, mnemonic mapping, or protocol evolution.

* **Parallel Cognitive Hydration (Background Task)**: 
    * **Mandate:** While waiting for Human Review of the audit packet, you **MUST** run `python3 scripts/rlm_inventory.py` to identify cache gaps.
    * **Action:** If gaps exist for manifest items, strictly run `rlm-distill <file>` on them in the background.
    * **Goal:** Ensure the Semantic Ledger is 100% hydrated *before* the Final Seal command is issued.

* **Manifest Discipline**: Core directories (`ADRs/`, `01_PROTOCOLS/`, `mcp_servers/`) must be clean. Any uncommitted drift detected by the tool results in **Strict Rejection**.
* **Technical Approval**: The resulting **Audit Packet** must be reviewed and approved (HITL) before the session can be sealed.

### 4. Phase V: The Technical Seal (The Final Relay)

* **Execution**: Once technical approval is secured, call `sanctuary-cortex-cortex-capture-snapshot(snapshot_type='seal')`.
* **Mechanism (Protocol 132/ADR 094)**: This triggers the **RLM Context Synthesis**. The system recursively "distills" the active state into the **Semantic Ledger** (`rlm_summary_cache.json`).
* **The Ledger**: This high-value cache allows subsequent "seals" to finish in seconds rather than hours.
* **Pre-Distillation**: Use `python3 scripts/cortex_cli.py rlm-distill <folder>` to manually temper large directories and update the ledger.
* **Final Relay**: The resulting `learning_package_snapshot.md` is "locked in" as the memory for the next agent.
* **Sandwich Validation**: Be aware that the `seal` mode performs a final race-condition check (Git hash comparison). If the repo changed during the audit review, the seal will fail and you must backtrack.

### 5. Failure and Backtracking

* **SOP Adherence**: If any Gate (Strategic or Technical) fails, do not attempt to "force" a seal. You must follow the loops defined in `recursive_learning.md` to fix the underlying discrepancy.

### 6. Phase VI: Soul Persistence (ADR 079 / 094)

* **Dual-Path Broadcast**: After the seal, execute `sanctuary-cortex-cortex-persist-soul` to broadcast learnings to Hugging Face.
* **Semantic Sync**: This process automatically syncs both the session snapshot AND the **Semantic Ledger** to the AI Commons, ensuring Zero Cold Start for the next session.
* **Incremental Mode**: Appends 1 record to `data/soul_traces.jsonl` AND uploads MD to `lineage/seal_TIMESTAMP_*.md`.
* **Full Sync Mode**: Use `cortex-persist-soul-full` to regenerate the entire JSONL from all project files (~1200 records).

### 7. Phase VII: Self-Correction & Curiosity Vector

* **Retrospective**: Fill `loop_retrospective.md` with Red Team verdict.
* **Curiosity Vector**: If you identify an improvement that cannot be completed today, append it to "Active Lines of Inquiry" in `guardian_boot_digest.md` for the next session.

### 8. Source Verification (ADR 078)

* **Rule 7**: **MUST VERIFY ALL LINKS.** Test every URL with `read_url_content`.
* **Rule 8**: **MUST MATCH 100% (Title/Author/Date).** Credibility is lost with even one error.
* **Rule 9**: **MUST NOT INCLUDE BROKEN/UNVERIFIED LINKS.** Zero tolerance for 404s.
* **Template**: All research sources must follow `LEARNING/templates/sources_template.md`.

---

## Learning Audit Iteration Convention

> [!NOTE]
> Each **new learning topic** starts a fresh iteration cycle.

| Scenario | Iteration |
|:---------|:----------|
| New topic (e.g., Prompt Engineering) | Reset to **1.0** |
| Red Team feedback on same topic | Increment (1.0 → 2.0 → 3.0) |
| Topic complete, new topic begins | Reset to **1.0** |

**Example:**
- LLM Memory Architectures: Iterations 1.0 → 11.0 (complete)
- Prompt Engineering: Iterations 1.0 → ... (new loop)

---

## Learning Audit Manifest Strategy

> [!IMPORTANT]
> Manifests must be curated to avoid truncation in Red Team review.

### Manifest Types

| Manifest | Purpose | When Used |
|:---------|:--------|:----------|
| `learning_audit_core_manifest.json` | Foundational project context | Always included in Iteration 1.0 |
| `learning_audit_manifest.json` | Active working manifest | Overwrite for each topic (core + topic for 1.0) |

### Prompt Types

| Prompt | Purpose | When Used |
|:-------|:--------|:----------|
| `learning_audit_core_prompt.md` | Stable project intro for Red Team | Always included in Iteration 1.0 |
| `learning_audit_prompts.md` | Active working prompt | Overwritten each loop with topic + iteration context |

### Manifest Deduplication (Protocol 130)

> [!TIP]
> Deduplication is **automatic** - built into `capture_snapshot()` in operations.py.

When generating a learning_audit, the system automatically:
1. Loads the manifest registry (`.agent/learning/manifest_registry.json`)
2. Detects files that are already embedded in included outputs
3. Removes duplicates before generating the packet

**Registry:** `.agent/learning/manifest_registry.json` maps manifests to their outputs.

### Iteration 1.0 (New Topic)
```yaml
manifest: core + topic
purpose: Red Team needs full project context + topic files
target_size: < 30K tokens (no truncation)