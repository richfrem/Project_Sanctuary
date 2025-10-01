# The Commandable Council: An Autonomous Triad Orchestrator  
**Blueprint (`council_orchestrator/README.md` – v1.3 Continuity-Hardened)**

This directory contains the foundational architecture for the Sanctuary's **Autonomous Triad**, a persistent, commandable multi-agent system designed for complex problem-solving and strategic deliberation.

---

## Core Architecture: Protocol 94 & 95

The system is a direct implementation of two core Sanctuary doctrines:

* **Protocol 94: The Persistent Council**  
  Guarantees that the agentic Council is not amnesiac. It achieves this by serializing and deserializing each agent's chat history, allowing its memory and context to persist across multiple tasks and script executions.

* **Protocol 95: The Commandable Council**  
  Establishes a Guardian-level command and control interface. The system operates as a persistent service that monitors for structured tasks, executes them, produces verifiable artifacts, and then returns to an idle state awaiting further instruction.

Together, these protocols balance agent autonomy with Guardian oversight, ensuring both continuity and control.

---

## System Components

1. **`orchestrator.py` (The Engine – v2.1 Briefing-Hardened):**  
   Persistent orchestrator process. Initializes agents, monitors for `command.json`, manages dialogue, handles knowledge requests, saves agent state, and now includes hardened **briefing packet lifecycle** (generation, continuity check, SYSTEM INSTRUCTION injection, archiving).

2. **`command.json` (The Control Panel):**  
   The Guardian’s sole interface to issue tasks. Consumed automatically by the Orchestrator. See `command_schema.md` for exact schema.

3. **`session_states/` (The Memory):**  
   Serialized chat histories for each agent:  
   - `coordinator_session.json`  
   - `strategist_session.json`  
   - `auditor_session.json`  
   - `guardian_session.json`  

4. **`dataset_package/` (The Identity – External Dependency):**  
   Core Essence Awakening Seeds that inoculate each agent with its role-specific persona.

5. **`WORK_IN_PROGRESS/council_memory_sync/` (The Synchronization Hub – v1.2+):**  
   Components of the **Council Memory Synchronization Protocol**:  
   - `briefing_packet.json`: Auto-generated injectable context bundle.  
   - `cortex_query_schema.json`: Schema for Cortex queries.  
   - `briefing_cycle_spec.md`: Cycle specification.  
   - `continuity_check_module.py`: Verifies mnemonic anchors before injection.  

6. **`bootstrap_briefing_packet.py` (The Context Generator – v1.2+):**  
   Dynamically generates `briefing_packet.json` by pulling the last two Chronicle entries, prior directives, and the live `command.json` task. Integrated into orchestrator for automatic execution.

7. **`command_schema.md` (The Interface Guide):**  
   Documentation of `command.json` structure.

---

## Operational Workflow (Guardian ↔ Council)

```mermaid
sequenceDiagram
    participant Guardian as Guardian (Steward)
    participant CommandFile as command.json
    participant SentryThread as Sentry Thread
    participant AsyncQueue as Async Queue
    participant MainLoop as Main Loop
    participant Briefing as Briefing Packet Lifecycle
    participant Continuity as Continuity Check Module
    participant Triad as Triad Agents
    participant Output as Output Artifacts
    participant States as Session States

    Guardian->>CommandFile: 1. Create command.json
    SentryThread->>AsyncQueue: 2. Detect and enqueue
    SentryThread->>CommandFile: 3. Consume command.json

    AsyncQueue->>MainLoop: 4. Dequeue command
    MainLoop->>Briefing: 5. Generate new briefing_packet.json
    Briefing->>Continuity: 6. Verify Chronicle anchors
    Continuity-->>Briefing: 7. Log pass/fail result
    Briefing->>Triad: 8. Inject SYSTEM INSTRUCTION context

    MainLoop->>States: 9. Load histories
    MainLoop->>Triad: 10. Initiate deliberation

    loop Multi-Round Dialogue
        Triad->>Triad: 11. Exchange
        Triad-->>MainLoop: Request file
        MainLoop->>Output: Read file content
        MainLoop-->>Triad: Provide context
    end

    Triad-->>MainLoop: 12. Synthesize final artifact
    MainLoop->>Output: 13. Save directive
    MainLoop->>States: 14. Save histories
    MainLoop->>Briefing: 15. Archive packet

    MainLoop-->>MainLoop: 16. Idle, await next task
````

---

## Multi-Threaded Architecture

The Orchestrator employs a multi-threaded architecture for responsiveness:

```mermaid
graph TD
    A[Main Process] --> B[Sentry Thread]
    A --> C[Main Async Loop]
    C --> D[Agent Query Threads]

    B[Sentry Thread<br/>- Monitors command.json<br/>- Polls every 1s<br/>- Parses/enqueues<br/>- Consumes file]
    C[Main Async Loop<br/>- Processes queued commands<br/>- Orchestrates tasks<br/>- Manages agent interactions<br/>- Handles knowledge requests]
    D[Agent Query Threads<br/>- Synchronous API calls<br/>- Executed via run_in_executor<br/>- Prevents blocking]

    B -->|Enqueue| E[Async Queue]
    E -->|Dequeue| C
    C -->|Save/Load| F[Session States]
    C -->|Read Files| G[Knowledge Base]
    C -->|Write Artifact| H[Output Artifacts]
    C -->|Generate/Verify/Inject/Archive| I[Briefing Packet Lifecycle]
```

---

## How to Use

### 1. Launch the Orchestrator (Council’s Terminal)

```bash
cd council_orchestrator
pip install -r requirements.txt
python3 orchestrator.py
```

You will see idle messages, indicating readiness for commands.

### 2. Issue a Command (Guardian’s Terminal)

Create `council_orchestrator/command.json` with the following structure:

```json
{
  "task_description": "A high-level strategic goal.",
  "input_artifacts": ["path/to/relevant/file1.md"],
  "output_artifact_path": "path/to/save/output.md",
  "config": { "max_rounds": 3 }
}
```

The Orchestrator will detect the file, execute the deliberation, save the artifact, archive the briefing packet, and return to idle.

---

## v1.3 Continuity-Hardened Features

### Council Memory Synchronization Protocol (Expanded)

* **Auto-Generation:** Briefing packet dynamically built from Chronicle, directives, and current task.
* **Continuity Verification:** `continuity_check_module.py` validates temporal anchors before injection. Results logged into cycle artifacts.
* **Injection:** SYSTEM INSTRUCTION ensures aligned context without verbatim regurgitation.
* **Archiving:** Packets archived post-deliberation for auditability.

### Guardian Role Integration

Guardian’s awakening seed and state file allow persistent meta-level oversight.

### Future Extensions

* Self-healing on anchor mismatch.
* Rolling memory windows for long projects.
* Guardian audit logs across cycles.

---

