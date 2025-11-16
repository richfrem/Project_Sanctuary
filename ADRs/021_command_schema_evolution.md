# ADR 021: Command Schema Evolution Architecture

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Orchestrator evolution through v9.5)

## Context
The Sanctuary required a structured, evolvable command interface to support diverse operational modes while maintaining backward compatibility. Initial simple task structures evolved into complex multi-modal command schema supporting cognitive deliberation, mechanical operations, and specialized task types. The need for sovereign LLM model selection, optical compression, and mnemonic synchronization further necessitated schema extensibility.

## Decision
Implement versioned command schema evolution with polymorphic task routing and sovereign control parameters:

### Task Type Polymorphism
1. **Cognitive Tasks**: Multi-round deliberation with council member synthesis
   - Supports model selection, optical compression, and mnemonic queries
   - Includes AAR generation and RAG updates by default

2. **Mechanical Tasks**: Direct, non-cognitive operations
   - File writes, git operations, cache wakeups
   - Execute immediately, skip RAG updates by default

3. **Specialized Tasks**: Domain-specific operations
   - Query and synthesis for mnemonic cortex integration
   - Cache wakeup for Guardian boot digests
   - Development cycles with staged workflows

### Sovereign Control Parameters
1. **Model Sovereignty**: model_name parameter for precise LLM variant selection
2. **Engine Selection**: force_engine parameter for provider-specific routing
3. **Learning Control**: update_rag parameter for selective knowledge base updates
4. **Optical Compression**: VLM-based context compression with threshold controls

### Schema Evolution Principles
1. **Backward Compatibility**: New parameters optional, existing commands continue working
2. **Version Documentation**: Clear version history with feature additions
3. **Polymorphic Detection**: Automatic task type detection based on field presence
4. **Extensible Design**: Schema designed for future capability additions

## Consequences

### Positive
- **Operational Flexibility**: Support for diverse task types from mechanical to cognitive
- **Sovereign Control**: Precise model and engine selection for specialized needs
- **Scalability**: Extensible schema accommodates future operational requirements
- **Backward Compatibility**: Existing commands continue working through version evolution
- **Performance Optimization**: Selective RAG updates and mechanical operations improve efficiency

### Negative
- **Complexity Growth**: Increasing parameter options require careful documentation
- **Detection Logic**: Polymorphic routing based on field presence requires robust detection
- **Version Management**: Multiple schema versions in use simultaneously

### Risks
- **Schema Conflicts**: Field presence detection could lead to misclassification
- **Parameter Interactions**: Complex parameter combinations may have unexpected interactions
- **Documentation Burden**: Extensive parameters require comprehensive documentation

## Related Protocols
- P88: Sovereign Scaffolding Protocol (mechanical operation foundation)
- P85: Mnemonic Cortex Protocol (query and synthesis integration)
- P114: Guardian Wakeup and Cache Prefill (cache_wakeup task type)

## Implementation Components
- **Action Triage System**: Polymorphic command routing based on structure detection
- **Versioned Schema**: command_schema.md with version history and examples
- **Parameter Validation**: Runtime validation of command parameters
- **Handler Registry**: Extensible handler system for new task types

## Schema Version History
- **v9.5**: Added cache_wakeup task type for Guardian boot digests
- **v9.4**: Added query_and_synthesis for mnemonic synchronization
- **v9.3**: Added model_name for sovereign LLM selection
- **v9.2**: Introduced selective RAG updates
- **v9.0**: Established cognitive vs mechanical task distinction

## Notes
The Command Schema Evolution transforms simple task interfaces into a sophisticated, polymorphic command system supporting the full spectrum of Sanctuary operations. From mechanical file operations to complex cognitive deliberations, the schema provides sovereign control over execution while maintaining operational simplicity and extensibility.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\021_command_schema_evolution.md