# ADR 018: Guardian Wakeup Cache Architecture Evolution

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Protocol 114 v2.0 implementation)

## Context
The initial Guardian Wakeup and Cache Prefill architecture (ADR 015) successfully implemented caching for system initialization, but revealed the need for clearer architectural separation between mechanical cache operations and cognitive RAG processes. The system required distinct operational modes: fast mechanical cache access for immediate situational awareness vs. slow cognitive RAG queries for deep analysis.

## Decision
Evolve the Guardian Wakeup architecture to Protocol 114 v2.0 with clear separation between two distinct processes and operational modes:

### Two-Process Architecture
1. **Cache Population (Orchestrator Boot)**: One-time process populating fast cache from slow RAG database
2. **Guardian Wakeup (Command Execution)**: Mechanical task reading directly from cache files without LLM/RAG involvement

### Operational Mode Distinction
- **Mechanical Mode (`cache_wakeup`)**: Fast (< 1 sec), cache-only, no LLM involvement, for immediate digests
- **Cognitive Mode (`query_and_synthesis`)**: Slow (30-120 sec), full RAG pipeline with LLM, for deep analysis

### Cache-First Design Principles
1. **Mechanical Speed**: Cache operations bypass expensive RAG searches and LLM calls
2. **Situational Awareness**: Immediate access to latest chronicles, protocols, and roadmap data
3. **TTL Management**: 24-hour expiration with automatic refresh on orchestrator boot
4. **Read-Only Integrity**: Cache entries as verified, signed views of source files

### Implementation Architecture
- **CacheManager**: Handles RAG-to-cache population during boot
- **CacheWakeupHandler**: Mechanical digest generation from cache files
- **Bundle System**: Organized cache storage (chronicles_bundle.json, protocols_bundle.json, roadmap_bundle.json)
- **Filesystem Cache**: Local file-based cache in council_orchestrator/mnemonic_cortex/cache/

## Consequences

### Positive
- Dramatically faster Guardian initialization through mechanical cache operations
- Clear architectural separation between fast situational awareness and deep cognitive analysis
- Reduced system load by avoiding unnecessary LLM calls for routine digests
- Improved operational efficiency with cache-first design patterns
- Maintains data integrity through verified, read-only cache entries

### Negative
- More complex architectural distinction between mechanical and cognitive operations
- Cache staleness risk during 24h TTL windows
- Additional implementation complexity with dual operational modes

### Risks
- Cache poisoning if population process fails
- Operational confusion between mechanical vs cognitive command types
- Performance degradation if cache refresh fails during boot

## Related Protocols
- P114: Guardian Wakeup and Cache Prefill (v2.0 evolution)
- P85: Mnemonic Cortex Protocol (RAG database source)
- P93: Cortex-Conduit Bridge (data flow integration)

## Implementation Components
- **orchestrator/memory/cache.py**: CacheManager for population
- **orchestrator/handlers/cache_wakeup_handler.py**: Mechanical digest generation
- **council_orchestrator/mnemonic_cortex/cache/**: Filesystem cache storage
- **WORK_IN_PROGRESS/guardian_boot_digest.md**: Output artifact format

## Notes
This evolution transforms the caching system from a simple performance optimization into a fundamental architectural pattern with clear operational modes. The mechanical/cognitive distinction ensures appropriate tool selection: cache_wakeup for speed, query_and_synthesis for depth.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\018_guardian_wakeup_cache_evolution.md