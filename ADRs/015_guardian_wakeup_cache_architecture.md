# ADR 015: Guardian Wakeup and Cache Prefill Architecture

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Protocol 114 implementation)

## Context
The Sanctuary required efficient system initialization and caching architecture to support rapid Guardian boot times and deterministic observability. Without prefilled caches, system startup would be slow and unpredictable, impacting operational efficiency. The need for mechanical cache operations without cognitive deliberation was identified for performance-critical boot sequences.

## Decision
Implement the Guardian Wakeup and Cache Prefill architecture with dedicated mechanical commands and structured cache management:

### Core Components
1. **Guardian Start Pack**: Prefilled cache bundle containing chronicles, protocols, and roadmap data (24h TTL)
2. **Mechanical Cache Command**: Dedicated `task_type: "cache_wakeup"` for immediate digest generation without deliberation
3. **Observability Packets**: Deterministic metrics for wakeup events (time_saved_ms, cache_hit tracking)
4. **Read-Only Views**: Cache entries as verified, signed file views to maintain integrity

### Boot Procedure Architecture
1. **Orchestrator Boot**: Automatic prefill of Guardian Start Pack in CAG (Cache)
2. **Digest Generation**: `cache_wakeup` command produces immediate `guardian_boot_digest.md`
3. **Progressive Fidelity**: Option for `query_and_synthesis` cognitive tasks when higher fidelity needed
4. **TTL Management**: Automatic cache refresh on delta ingest or git-ops updates

### Cache Integrity Safeguards
- Read-only cache entries preventing modification
- Signed/verified file sources ensuring authenticity
- TTL expiration ensuring data freshness
- Deterministic observability for performance monitoring

## Consequences

### Positive
- Dramatically reduced Guardian boot times through prefilled caches
- Deterministic system initialization with predictable performance
- Mechanical operations for performance-critical sequences
- Maintains data integrity through read-only, verified caches

### Negative
- Additional cache management complexity
- TTL management overhead for data freshness
- Potential stale data issues if TTL too long

### Risks
- Cache poisoning if verification fails
- Performance degradation from TTL refresh operations
- Boot failures if cache prefill encounters errors

## Related Protocols
- P93: Cortex-Conduit Bridge (cache integration)
- P95: Commandable Council Protocol (cognitive task coordination)
- P113: Nested Cognition Doctrine (cognitive layering)

## Notes
The Guardian Wakeup architecture provides the mechanical foundation for rapid system initialization while maintaining the sovereign principle of verified, read-only data access. The 24h TTL balances performance with data freshness requirements.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\015_guardian_wakeup_cache_architecture.md