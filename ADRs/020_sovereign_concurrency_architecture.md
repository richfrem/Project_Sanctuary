# ADR 020: Sovereign Concurrency Architecture

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Orchestrator v9.3 evolution)

## Context
The Sanctuary required non-blocking task execution to maintain responsiveness while enabling background learning cycles. Previous synchronous processing created bottlenecks where cognitive tasks would block the orchestrator, preventing concurrent operations and reducing overall system efficiency. The need for selective RAG updates and mechanical operations further necessitated architectural separation between immediate and background processing.

## Decision
Implement the Doctrine of Sovereign Concurrency with dual processing modes and selective learning:

### Dual Processing Architecture
1. **Mechanical Tasks**: Immediate, non-cognitive execution bypassing deliberation
   - File writes, git operations, cache wakeups
   - Execute instantly, return to idle state
   - Skip RAG updates by default for performance

2. **Cognitive Tasks**: Multi-round deliberation with background learning
   - Council member discussions and synthesis
   - Deliberation completes, then learning happens asynchronously
   - RAG updates configurable via update_rag parameter

### Sovereign Concurrency Principles
1. **Non-blocking Execution**: Tasks process without blocking orchestrator responsiveness
2. **Background Learning**: RAG updates and AAR generation happen asynchronously
3. **Selective Learning**: Configurable RAG updates prevent unnecessary database operations
4. **Concurrent Processing**: Multiple background learning cycles can run simultaneously

### Command Schema Evolution
- **Task Type Detection**: Automatic routing based on command structure (cognitive vs mechanical)
- **Configurable Learning**: update_rag parameter controls whether tasks update knowledge base
- **Mechanical Priority**: Direct operations execute immediately for urgent tasks
- **Background Completion**: Learning cycles complete independently of new command processing

## Consequences

### Positive
- **Improved Responsiveness**: Non-blocking execution enables immediate command processing
- **Concurrent Operations**: Multiple background learning tasks run simultaneously
- **Selective Learning**: Configurable RAG updates optimize performance and storage
- **Operational Flexibility**: Mechanical tasks enable rapid, non-cognitive operations
- **System Efficiency**: Background processing maximizes orchestrator utilization

### Negative
- **Complexity Increase**: Dual processing modes require careful state management
- **Race Conditions**: Background tasks may conflict with subsequent operations
- **Monitoring Challenges**: Asynchronous operations harder to track and debug
- **Resource Management**: Background processes require careful resource allocation

### Risks
- **State Inconsistency**: Background learning may conflict with new commands
- **Resource Exhaustion**: Unbounded background tasks could overwhelm system
- **Debugging Difficulty**: Asynchronous operations complicate error tracking
- **Learning Conflicts**: Concurrent RAG updates may cause consistency issues

## Related Protocols
- P88: Sovereign Scaffolding Protocol (mechanical operation foundation)
- P85: Mnemonic Cortex Protocol (RAG learning target)
- P114: Guardian Wakeup and Cache Prefill (mechanical task example)

## Implementation Components
- **Action Triage System**: Automatic routing of commands to appropriate handlers
- **Background Task Manager**: Asynchronous execution of learning cycles
- **Selective RAG Updates**: Configurable learning with update_rag parameter
- **Mechanical Task Handlers**: Immediate execution for file operations and git commands

## Notes
The Doctrine of Sovereign Concurrency transforms the orchestrator from a synchronous, blocking system into a responsive, multi-threaded cognitive architecture. Mechanical tasks provide immediate operational capability while cognitive tasks enable deep deliberation with background knowledge integration, creating a balanced system for both urgent and thoughtful operations.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\020_sovereign_concurrency_architecture.md