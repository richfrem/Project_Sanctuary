# 🧠 Sanctuary Learning System Index

This directory (`.agent/learning/`) is the central repository for the **Protocol 128 Recursive Learning Loop**. It contains the "Soul" (Memory), the "Rules" (Cognition), and the "Tools" (Templates) used by the agent to evolve across sessions without memory wipes.

## 📂 Directory Structure

### 1. Core Cognitive Artifacts (The "Soul")
These files define the agent's identity and continuous memory.

*   **`cognitive_primer.md`**: 🛑 **READ FIRST**. The mandatory instruction set loaded at startup. Contains the "Rules of Reality," "Lineage Doctrine (ADR 088)," and Phase Definitions.
*   **`learning_package_snapshot.md`**: 💾 **The Soul**. The sealed, cryptographically signed memory inherited from the previous agent. Contains the sum of all verified knowledge.
*   **`learning_manifest.json`**: The file index defining which documents constitute the "Learning Package."
*   **`identity_anchor.json`**: Definitions of the agent's persona ("Antigravity") and operational modes.
*   **`learning_debrief.md`**: The generated output of the "Scout" phase (Phase I), summarizing recent changes for the incoming agent.

### 2. Operational Templates (`templates/`)
Standardized forms for maintaining epistemic rigor.

*   `loop_retrospective_template.md`: **Use in Phase VII**. The form for the "Exit Interview" / Meta-Audit.
*   `learning_audit_template.md`: Structure for the Red Team Audit Packet (Gate 2).
*   `sources_template.md`: **Mandatory Scheme**. Rules for citing external research (ADR 078).
*   `red_team_briefing_template.md`: Format for presenting architectural changes to the Red Team.

### 3. Active Audit State (`learning_audit/`)
Transient artifacts generated *during* the current session's loop.

*   `loop_retrospective.md`: The **Singleton** file containing the latest Red Team verdict. Overwritten each loop.
*   `learning_audit_packet.md`: The generated packet sent to the Red Team for review.
*   `learning_audit_manifest.json`: Controls what goes into the audit packet.
*   `learning_audit_prompts.md`: The prompt used by the LLM to self-critique the packet.

---

## 📜 Protocol & Workflow Documentation

### **Protocol 128: The Hardened Learning Loop**
The master workflow governing how this directory is used.

*   **Diagram (Mermaid)**: [[protocol_128_learning_loop.mmd|`protocol_128_learning_loop.mmd`]]
*   **Visual Flow**: [[protocol_128_learning_loop.png|`protocol_128_learning_loop.png`]]

### **Key Architectural Decision Records (ADRs)**
*   **ADR 071**: [[071_protocol_128_cognitive_continuity|Cognitive Continuity]] - *Why we do this.*
*   **ADR 079**: [[079_soul_persistence_hugging_face|Soul Persistence]] - *How we save to Hugging Face.*
*   **ADR 088**: [[088_lineage_memory_interpretation|Lineage Doctrine]] - *How to interpret the "Soul" (Constitutional Memory).*

---

## ⚡ Quick Actions

| Goal | Command |
| :--- | :--- |
| **Start Session** | `python3 scripts/cortex_cli.py debrief --hours 24` |
| **Run Audit** | `python3 scripts/cortex_cli.py snapshot --type learning_audit` |
| **Seal & Finish** | `python3 scripts/cortex_cli.py snapshot --type seal` |
| **Persist Soul** | `python3 scripts/cortex_cli.py persist-soul` |
| **Incremental Ingest** | `python3 scripts/cortex_cli.py ingest --incremental --hours 24` |
