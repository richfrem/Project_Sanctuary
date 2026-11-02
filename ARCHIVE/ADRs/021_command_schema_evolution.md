# Task Schema Evolution Architecture

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** AI Council (System Controller evolution through v9.5)
**Technical Story:** Create structured, evolvable task interface for diverse operations

---

## Context

The AI system required a structured, evolvable task interface to support diverse operational modes while maintaining backward compatibility. Initial simple task structures evolved into complex multi-modal task schema supporting AI deliberation, automatic operations, and specialized task types. The need for independent AI model selection, visual content compression, and memory synchronization further necessitated schema extensibility.

## Decision

Implement versioned task schema evolution with flexible task routing and independent control parameters:

### Task Type Flexibility
1. **AI tasks**: Multi-round deliberation with AI Council member synthesis
   - Supports model selection, visual compression, and memory queries
   - Includes review generation and information retrieval updates by default

2. **Automatic tasks**: Direct, non-AI operations
   - File writes, git operations, cache wakeups
   - Execute immediately, skip information retrieval updates by default

3. **Specialized tasks**: Domain-specific operations
   - Query and synthesis for memory system integration
   - Cache wakeup for AI system boot summaries
   - Development cycles with staged workflows

### Independent Control Parameters
1. **Model Independence**: model_name parameter for precise AI model variant selection
2. **Engine Selection**: force_engine parameter for provider-specific routing
3. **Learning Control**: update_rag parameter for selective knowledge base updates
4. **Visual Compression**: Vision-based context compression with threshold controls

### Schema Evolution Principles
1. **Backward Compatibility**: New parameters optional, existing tasks continue working
2. **Version Documentation**: Clear version history with feature additions
3. **Flexible Detection**: Automatic task type detection based on field presence
4. **Extensible Design**: Schema designed for future capability additions

## Consequences

### Positive
- **Operational Flexibility**: Support for diverse task types from automatic to AI-based
- **Independent Control**: Precise model and engine selection for specialized needs
- **Scalability**: Extensible schema accommodates future operational requirements
- **Backward Compatibility**: Existing tasks continue working through version evolution
- **Performance Optimization**: Selective information retrieval updates and automatic operations improve efficiency

### Negative
- **Complexity Growth**: Increasing parameter options require careful documentation
- **Detection Logic**: Flexible routing based on field presence requires robust detection
- **Version Management**: Multiple schema versions in use simultaneously

### Risks
- **Schema Conflicts**: Field presence detection could lead to misclassification
- **Parameter Interactions**: Complex parameter combinations may have unexpected interactions
- **Documentation Burden**: Extensive parameters require comprehensive documentation

### Related Processes
- Automated Script Protocol (automatic operation foundation)
- Memory System Process (query and synthesis integration)
- AI System Startup and Cache Preparation (cache_wakeup task type)

### Implementation Components
- **Task Routing System**: Flexible command routing based on structure detection
- **Versioned Schema**: task_schema.md with version history and examples
- **Parameter Validation**: Runtime validation of task parameters
- **Handler Registry**: Extensible handler system for new task types

### Schema Version History
- **v9.5**: Added cache_wakeup task type for AI system boot summaries
- **v9.4**: Added query_and_synthesis for memory synchronization
- **v9.3**: Added model_name for independent AI model selection
- **v9.2**: Introduced selective information retrieval updates
- **v9.0**: Established AI vs automatic task distinction

### Notes
The Task Schema Evolution transforms simple task interfaces into a sophisticated, flexible command system supporting the full spectrum of AI system operations. From automatic file operations to complex AI deliberations, the schema provides independent control over execution while maintaining operational simplicity and extensibility.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\021_command_schema_evolution.md