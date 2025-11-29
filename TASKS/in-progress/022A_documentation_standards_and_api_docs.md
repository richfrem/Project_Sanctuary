# Task 022A: Documentation Standards & API Documentation

## Metadata
- **Status**: in-progress
- **Priority**: Medium
- **Complexity**: Medium
- **Category**: Documentation
- **Estimated Effort**: 4-6 hours
- **Dependencies**: None
- **Parent Task**: 022
- **Created**: 2025-11-28

## Objective

Establish documentation standards, create templates, add docstrings to public APIs (90%+ coverage), and set up automated API documentation generation with Sphinx.

## Deliverables

1. Create `docs/DOCUMENTATION_STANDARDS.md`
2. Create documentation templates:
   - Protocol template
   - Module template
   - API documentation template
   - Tutorial template
3. Set up Sphinx for API documentation
4. Add docstrings to all public functions in `mnemonic_cortex/core/`
5. Add docstrings to all public functions in `council_orchestrator/orchestrator/`
6. Generate HTML API documentation
7. Achieve 90%+ docstring coverage for public APIs

## Acceptance Criteria

- [ ] `docs/DOCUMENTATION_STANDARDS.md` created with clear guidelines
- [ ] 4 documentation templates created and documented
- [ ] Sphinx configured and generating API docs
- [ ] 90%+ docstring coverage in `mnemonic_cortex/core/`
- [ ] 90%+ docstring coverage in `council_orchestrator/orchestrator/`
- [ ] HTML API documentation accessible locally
- [ ] Documentation build process documented in README

## Implementation Steps

### 1. Create Documentation Standards (1 hour)
- Define style guide (Google, NumPy, or Sphinx style)
- Establish naming conventions
- Define required sections for each doc type
- Create examples

### 2. Create Templates (1 hour)
- Protocol template with metadata, objective, implementation
- Module template with overview, classes, functions
- API template with parameters, returns, examples
- Tutorial template with prerequisites, steps, verification

### 3. Set Up Sphinx (1-2 hours)
- Install Sphinx and extensions
- Configure `conf.py`
- Set up autodoc for automatic API generation
- Configure theme (e.g., Read the Docs)
- Test build process

### 4. Add Docstrings (2-3 hours)
- Audit existing coverage
- Add docstrings to `mnemonic_cortex/core/`:
  - `ingestion_service.py`
  - `rag_service.py`
  - `vector_db_service.py`
  - `embedding_service.py`
- Add docstrings to `council_orchestrator/orchestrator/`:
  - `council.py`
  - `engines/`
  - `agents/`
- Run coverage check

## Related Protocols

- **Protocol 85**: The Mnemonic Cortex Protocol - Documentation as living memory
- **Protocol 89**: The Clean Forge - Documentation as part of quality
- **Protocol 115**: The Tactical Mandate - Documentation as requirement

## Notes

This task establishes the foundation for all documentation in Project Sanctuary. The standards and templates created here will be used by all other documentation tasks.
