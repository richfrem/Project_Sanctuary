**Blueprint (`council_orchestrator/README.md` - v1.1 Hardened):**

# The Commandable Council: An Autonomous Triad Orchestrator

This directory contains the foundational architecture for the Sanctuary's **Autonomous Triad**, a persistent, commandable multi-agent system designed for complex problem-solving and strategic deliberation.

## Core Architecture: Protocol 94 & 95

The system is a direct implementation of two core Sanctuary doctrines:
*   **Protocol 94: The Persistent Council:** Guarantees that the agentic Council is not amnesiac. It achieves this by serializing and deserializing each agent's chat history, allowing its memory and context to persist across multiple tasks and script executions.
*   **Protocol 95: The Commandable Council:** Establishes a Guardian-level command and control interface. The system operates as a persistent service that monitors for structured tasks, executes them, produces verifiable artifacts, and then returns to an idle state awaiting further instruction.

This architecture provides the optimal balance between agent autonomy and Steward oversight.

## System Components

1.  **`orchestrator.py` (The Engine):** The main, persistent Python script. This is the "brain" of the system that runs continuously. It is responsible for initializing the agents, monitoring for commands, managing the dialogue, handling knowledge requests, and saving agent states.
2.  **`command.json` (The Control Panel):** An ephemeral JSON file that acts as the sole command interface. To assign a task to the Council, the Steward (Guardian) creates this file. The Orchestrator detects it, executes the task, and deletes the file upon completion.
3.  **`session_states/` (The Memory):** A directory containing the serialized chat history for each agent. The explicit filenames are:
    *   `coordinator_session.json`
    *   `strategist_session.json`
    *   `auditor_session.json`
4.  **`dataset_package/` (The Identity - External Dependency):** The Orchestrator inoculates each agent by reading its full Core Essence from the Awakening Seeds located in the project's central `dataset_package/` directory.

## Operational Workflow

The system operates as a continuous loop, managed across two terminals: one for the Council and one for the Guardian.

```mermaid
sequenceDiagram
    participant G as Guardian (Steward)
    participant CFile as command.json <br> `council_orchestrator/`
    participant Sentry as Sentry Thread <br> (File Watcher)
    participant Queue as Async Queue
    participant Main as Main Loop <br> (Async Task Processor)
    participant SA as Session States <br> `session_states/*.json`
    participant Triad as Triad Agents
    participant KB as Knowledge Base <br> Project Files
    participant A as Output Artifact <br> e.g., `WORK_IN_PROGRESS/`

    G->>CFile: 1. Creates/Updates command.json

    loop Continuous Monitoring (1s poll)
        Sentry->>CFile: 2. Monitors for file
    end

    %% Sentry detects and queues the command
    Sentry->>Queue: 3. Detects command.json, parses and enqueues
    Sentry->>CFile: 4. Deletes command.json (consumes file)

    %% Main Loop processes the queued command
    Main->>Queue: 5. Dequeues command
    Main->>SA: 6. Loads Agent Histories
    SA-->>Main: 7. Agents are stateful

    %% Main Loop delegates to the Triad
    Main->>+Triad: 8. Initiates Deliberation (Task)

    loop Deliberation Cycle (e.g., 3 Rounds)
        Triad->>Triad: 9. Agents converse
        Triad-->>Main: Agent requests knowledge
        Main->>KB: 10. Reads requested file
        KB-->>Main: Returns file content
        Main-->>Triad: 11. Injects knowledge into chat
    end

    %% Triad completes its work
    Triad-->>-Main: 12. Final proposal is synthesized

    Main->>A: 13. Writes final output artifact
    Main->>SA: 14. Saves updated Agent Histories

    %% Main Loop returns to idle
    Main-->>Main: 15. Returns to Idle/Awaiting State
```

### Multi-Threaded Architecture Overview

The v2.0 Orchestrator employs a multi-threaded architecture for optimal performance and responsiveness. The following diagram illustrates the key threads and their responsibilities:

```mermaid
graph TD
    A[Main Process] --> B[Sentry Thread]
    A --> C[Main Async Loop]
    C --> D[Agent Query Threads]

    B[Sentry Thread<br/>- Monitors command.json<br/>- Polls every 1 second<br/>- Parses and enqueues commands<br/>- Consumes command file]
    C[Main Async Loop<br/>- Processes queued commands<br/>- Orchestrates task execution<br/>- Manages agent interactions<br/>- Handles knowledge requests]
    D[Agent Query Threads<br/>- Synchronous API calls<br/>- Executed via run_in_executor<br/>- Prevent blocking async loop]

    B -->|Enqueue Command| E[Async Queue]
    E -->|Dequeue Command| C
    C -->|Save/Load State| F[Session States<br/>JSON Files]
    C -->|Read Files| G[Knowledge Base<br/>Project Files]
    C -->|Write Artifact| H[Output Artifact<br/>e.g., WORK_IN_PROGRESS/]
```

## How to Use

### 1. Launch the Orchestrator (The Council's Terminal)
In a dedicated terminal, start the persistent service. This only needs to be done once per session.

```bash
# Navigate to the orchestrator's directory
cd council_orchestrator

# Install dependencies (first time only)
pip install -r requirements.txt

# Launch the orchestrator
python3 orchestrator.py
```
The terminal will display an "Idle" message, indicating it is ready for a command.

### 2. Issue a Command (The Guardian's Terminal or VS Code)
To assign a task, create or edit the `council_orchestrator/command.json` file. The structure must be as follows:

```json
{
  "task_description": "A high-level strategic goal for the Triad to solve.",
  "input_artifacts": [
    "path/to/relevant/file1.md"
  ],
  "output_artifact_path": "path/where/to/save/the/final_result.md",
  "config": {
    "max_rounds": 3
  }
}
```

The Orchestrator will automatically detect the file, begin the task, and provide real-time updates in its own terminal. Once complete, it will delete the `command.json` file and await the next mission.


