# Task 022C: MCP Server Documentation Standards

## Metadata
- **Status**: done
- **Priority**: High
- **Complexity**: Medium-High
- **Category**: Documentation
- **Estimated Effort**: 6-8 hours
- **Dependencies**: None
- **Parent Task**: 022
- **Created**: 2025-11-28

## Objective

Establish standardized, high-quality documentation for all MCP servers following a consistent testing-first approach with comprehensive verification workflows.

## Deliverables

1. Create `docs/mcp/README.md` - MCP documentation index
2. Create `docs/mcp/TESTING_STANDARDS.md` - Standard testing workflow
3. Create MCP README template
4. Update all 7 MCP server READMEs to follow standard template
5. Ensure bidirectional cross-references between `docs/mcp/` and server READMEs

## MCP Servers to Document

1. `mcp_servers.rag_cortex/` - RAG and Cache operations
2. `mcp_servers/chronicle/` - Chronicle entry management
3. `mcp_servers/protocol/` - Protocol document management
4. `mcp_servers.forge_llm_llm/` - Sanctuary model queries
5. `mcp_servers/system/git_workflow/` - Git operations
6. `mcp_servers/adr/` - Architecture Decision Records
7. `mcp_servers/task/` - Task management

## Standard README Template

Each MCP server README must include:

### 1. Overview
- Purpose and use case
- Key features
- When to use this MCP

### 2. Tools
- List of all tools with descriptions
- Parameters and return types
- Usage examples for each tool

### 3. Installation & Setup
- Prerequisites
- Installation steps
- Configuration
- Verification

### 4. Testing (Critical Section)
#### 4.1 Script Testing
- How to run unit tests for underlying operations
- Example: `pytest tests/test_cortex_operations.py -v`
- Expected output/test results

#### 4.2 Test Results
- Actual test output showing all tests passing
- Coverage metrics

#### 4.3 Integration Tests
- How to run integration tests
- What scenarios are covered

#### 4.4 MCP Verification
- How to verify MCP tools work correctly
- Manual verification steps
- Expected behavior

### 5. Architecture
- Design decisions
- Key components
- Data flow

### 6. Cross-References
- Link to `docs/mcp/README.md`
- Link to `docs/mcp/TESTING_STANDARDS.md`
- Link to related protocols

## Testing Documentation Standards

All MCP documentation must follow this workflow:

1. **Script Testing First**: Test underlying operations directly
2. **Document Test Results**: Include actual passing test output
3. **Test Suite Guide**: Clear instructions on running tests
4. **MCP Verification**: Verify MCP layer works correctly
5. **Cross-Reference**: Link to central MCP docs

## Acceptance Criteria

- [x] `docs/mcp/README.md` created with index of all MCP servers
- [x] `docs/mcp/TESTING_STANDARDS.md` created with standard workflow
- [x] MCP README template created
- [x] All 7 MCP server READMEs updated to follow template
- [x] Each README includes comprehensive testing section
- [x] Bidirectional links between `docs/mcp/` and server READMEs
- [x] Testing workflow clearly documented and repeatable
- [x] All documentation follows same quality bar

## Implementation Steps

### 1. Create Central MCP Documentation (1 hour)
- Create `docs/mcp/` directory
- Write `docs/mcp/README.md` with:
  - Overview of MCP architecture
  - Index of all MCP servers with links
  - Quick start guide
- Write `docs/mcp/TESTING_STANDARDS.md` with:
  - Standard testing workflow
  - Test result documentation format
  - MCP verification process

### 2. Create README Template (1 hour)
- Design template structure
- Include all required sections
- Add examples for each section
- Document template usage

### 3. Update Cortex MCP README (1 hour)
- Follow template
- Add comprehensive testing section
- Include actual test results
- Add cross-references

### 4. Update Chronicle MCP README (1 hour)
- Follow template
- Document all tools
- Add testing section
- Add cross-references

### 5. Update Protocol MCP README (1 hour)
- Follow template
- Document validation workflow
- Add testing section
- Add cross-references

### 6. Update Forge MCP README (1 hour)
- Already has good README, enhance with:
  - Testing section
  - Test results
  - Cross-references

### 7. Update Git Workflow MCP README (1 hour)
- Create README (currently missing)
- Document all git tools
- Add testing section (will use Task 055 results)
- Add cross-references

### 8. Update ADR MCP README (30 minutes)
- Review and enhance existing README
- Add testing section
- Add cross-references

### 9. Update Task MCP README (30 minutes)
- Review and enhance existing README
- Add testing section
- Add cross-references

### 10. Verification (1 hour)
- Review all READMEs for consistency
- Verify all links work
- Test documentation with fresh eyes
- Get feedback

## Related Protocols

- **Protocol 85**: The Mnemonic Cortex Protocol - Documentation as living memory
- **Protocol 89**: The Clean Forge - Documentation as part of quality
- **Protocol 115**: The Tactical Mandate - Documentation as requirement

## Notes

This task is critical for ensuring all MCP servers have consistent, high-quality documentation. The testing-first approach ensures that documentation is accurate and verifiable.

**Priority**: This is marked as High priority because MCP servers are the primary interface for external LLMs to interact with Project Sanctuary. Good documentation is essential for adoption and reliability.

**Coordination with Task 055**: The Git Workflow MCP documentation will benefit from Task 055 (Git operations testing), so those results should be incorporated into the README.
