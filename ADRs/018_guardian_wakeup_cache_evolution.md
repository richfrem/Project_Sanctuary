# AI System Startup Cache Architecture Evolution

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** AI Council (System Initialization Process v2.0 implementation)
**Technical Story:** Improve system startup performance and operational clarity

---

## Context

The initial AI System Startup and Cache Preparation architecture successfully implemented caching for system initialization, but revealed the need for clearer separation between automatic cache operations and detailed information retrieval processes. The system required distinct operational modes: fast automatic cache access for immediate situational awareness vs. slow detailed queries for deep analysis.

## Decision

We will evolve the AI System Startup architecture to Process v2.0 with clear separation between two distinct processes and operational modes:

### Two-Process Architecture
1. **Cache Filling (System Controller Boot)**: One-time process filling fast cache from slow information retrieval database
2. **AI System Startup (Command Execution)**: Automatic task reading directly from cache files without AI involvement

### Operational Mode Distinction
- **Automatic Mode (`cache_wakeup`)**: Fast (< 1 sec), cache-only, no AI involvement, for immediate summaries
- **Detailed Mode (`query_and_synthesis`)**: Slow (30-120 sec), full information retrieval pipeline with AI, for deep analysis

### Cache-First Design Principles
1. **Automatic Speed**: Cache operations skip expensive searches and AI calls
2. **Immediate Awareness**: Instant access to latest history, processes, and roadmap data
3. **Time Management**: 24-hour expiration with automatic refresh on system controller boot
4. **Protected Integrity**: Cache entries as verified, signed views of source files

### Implementation Architecture
- **CacheManager**: Handles retrieval-to-cache filling during boot
- **CacheWakeupHandler**: Automatic summary creation from cache files
- **Bundle System**: Organized cache storage (history_bundle.json, processes_bundle.json, roadmap_bundle.json)
- **File Cache**: Local file-based cache in council_orchestrator/memory_system/cache/

## Consequences

### Positive
- Significantly faster AI initialization through automatic cache operations
- Clear architectural separation between fast situational awareness and deep detailed analysis
- Reduced system load by avoiding unnecessary AI calls for routine summaries
- Improved operational efficiency with cache-first design patterns
- Maintains data integrity through verified, protected cache entries

### Negative
- More complex architectural distinction between automatic and detailed operations
- Cache outdated data risk during 24-hour time windows
- Additional implementation complexity with dual operational modes

### Risks
- Cache corruption if filling process fails
- Operational confusion between automatic vs detailed command types
- Performance issues if cache refresh fails during boot

### Related Processes
- AI System Startup and Cache Preparation (v2.0 evolution)
- Memory System Process (information retrieval database source)
- Memory-System Connection (data flow integration)

### Implementation Components
- **orchestrator/memory/cache.py**: CacheManager for filling
- **orchestrator/handlers/cache_wakeup_handler.py**: Automatic summary creation
- **council_orchestrator/memory_system/cache/**: File cache storage
- **WORK_IN_PROGRESS/ai_boot_summary.md**: Output format

### Notes
This evolution transforms the caching system from a simple performance optimization into a fundamental architectural pattern with clear operational modes. The automatic/detailed distinction ensures appropriate tool selection: cache_wakeup for speed, query_and_synthesis for depth.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\018_guardian_wakeup_cache_evolution.md