# Hybrid Context Retrieval for Agent Wakeup

**Status:** proposed
**Date:** 2025-12-09
**Author:** AI Assistant


---

## Context

The current `cortex_guardian_wakeup` relies solely on semantic search (`cortex_query`) against the vector database. This creates a 'Recency Blindspot': work done in the immediate previous session (or not yet indexed) is invisible to the next agent. This violates the 'Continuous Learning' doctrine and causes agents to repeat work or violate new protocols (like Protocol 118) because they cannot 'remember' them.

## Decision

We will adopt a **Hybrid Context Retrieval** architecture for Guardian Wakeup. The operation will now aggregate: 1. **Semantic Digest** (Long-term Memory via Vector DB) 2. **Recency Delta** (Short-term Memory via File System/Git Logs). This ensures agents wake up with a complete view of both foundational doctrine and immediate operational context.

## Consequences

Positive: Eliminates agent amnesia for recent events; ensures immediate continuity. Negative: Slight increase in wakeup latency due to file system checks; requires careful filtering to avoid context window overflow.
