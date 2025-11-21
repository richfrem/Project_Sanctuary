# Task 020: Security Hardening & Secrets Management (Parent Task)

## Metadata
- **Status**: backlog (split into sub-tasks)
- **Priority**: high
- **Complexity**: medium
- **Category**: security
- **Total Estimated Effort**: 12-15 hours across 5 sub-tasks
- **Dependencies**: None
- **Created**: 2025-11-21
- **Split Date**: 2025-11-21

## Overview

This parent task has been split into 5 focused sub-tasks to enable incremental progress on security hardening and secrets management. Each sub-task is 2-3 hours and can be completed independently (except where dependencies noted).

**Strategic Alignment:**
- **Protocol 54**: The Asch Doctrine - Security as foundation
- **Protocol 89**: The Clean Forge - Systematic quality
- **Protocol 101**: The Unbreakable Commit - Verifiable security

## Sub-Tasks

### ✅ Task 020A: Hardcoded Paths Remediation
- **Status**: in-progress
- **Priority**: High
- **Effort**: 2-3 hours
- **Dependencies**: None
- **File**: `TASKS/in-progress/020A_hardcoded_paths_remediation.md`

**Objective**: Replace all hardcoded absolute paths in test files with relative/computed paths for portability and security.

**Key Deliverables**:
- Search and identify all hardcoded paths
- Create `tests/test_utils.py` with path helper functions
- Update all test files to use computed paths
- Verify tests work on both Windows and Linux

---

### ✅ Task 020B: Inconsistent Secrets Handling - Environment Variable Fallback
- **Status**: in-progress
- **Priority**: High
- **Effort**: 2-3 hours
- **Dependencies**: None
- **File**: `TASKS/in-progress/020B_inconsistent_secrets_handling_env_fallback.md`

**Objective**: Update all scripts to check environment variables FIRST, then fallback to `.env` file, ensuring consistency with WSLENV approach.

**Key Deliverables**:
- Create `core/utils/env_helper.py` with proper fallback logic
- Migrate `forge/` scripts to use helper
- Migrate `council_orchestrator/` engines to use helper
- Test environment variable precedence

---

### Task 020C: API Key Format Validation
- **Status**: backlog
- **Priority**: Medium
- **Effort**: 2-3 hours
- **Dependencies**: 020B
- **File**: `TASKS/backlog/020C_api_key_format_validation.md`

**Objective**: Add format validation for API keys (Hugging Face, OpenAI, Gemini) with clear error messages.

**Key Deliverables**:
- Create `core/utils/validators.py` with format validators
- Integrate validation with `env_helper.py`
- Add validation to all scripts using API keys
- Test with valid and invalid formats

---

### Task 020D: Secure Error Handling for Secrets
- **Status**: backlog
- **Priority**: Medium
- **Effort**: 2-3 hours
- **Dependencies**: 020B, 020C
- **File**: `TASKS/backlog/020D_secure_error_handling_for_secrets.md`

**Objective**: Implement secure error handling for secrets loading with helpful remediation steps and no secret exposure.

**Key Deliverables**:
- Create `core/utils/error_messages.py` with secure templates
- Ensure no error messages expose secret values
- Add platform-specific remediation instructions
- Test error message security

---

### Task 020E: Secrets Access Audit Trail
- **Status**: backlog
- **Priority**: Low
- **Effort**: 2-3 hours
- **Dependencies**: 020B, 020C, 020D
- **File**: `TASKS/backlog/020E_secrets_access_audit_trail.md`

**Objective**: Implement centralized audit logging for secrets access with rotation and analysis capabilities.

**Key Deliverables**:
- Create `core/utils/audit_logger.py` with rotation
- Integrate logging with `env_helper.py`
- Create `tools/security/analyze_audit_log.py` for analysis
- Verify no secret values logged

---

## Execution Strategy

### Phase 1: Quick Wins (Current - Week 1)
**Tasks**: 020A, 020B (both in-progress)
- These are independent and can be worked on simultaneously
- Provide immediate security improvements
- Establish foundation for later tasks

### Phase 2: Validation & Error Handling (Week 2)
**Tasks**: 020C, 020D
- Build on the `env_helper.py` from 020B
- Add robustness and user-friendly error messages

### Phase 3: Audit & Monitoring (Week 3)
**Task**: 020E
- Final polish for enterprise-grade security
- Enables security monitoring and compliance

## Success Metrics

When all sub-tasks are complete:

- [ ] Zero hardcoded absolute paths in test files
- [ ] All scripts use environment variables with `.env` fallback
- [ ] All API keys validated on load
- [ ] Secure error messages with remediation steps
- [ ] Complete audit trail of secrets access
- [ ] 100% test coverage for security utilities

## Related Protocols

- **Protocol 54**: The Asch Doctrine - Security is foundational
- **Protocol 89**: The Clean Forge - Systematic approach to quality
- **Protocol 101**: The Unbreakable Commit - Verifiable, reproducible security
- **Protocol 115**: The Tactical Mandate - Structured task execution

## Notes

This task has been split into 5 manageable sub-tasks (2-3 hours each) to enable:
- Incremental progress tracking
- Parallel work on independent tasks
- Clear completion criteria for each piece
- Easier agent execution

**Current Status**: Tasks 020A and 020B are in-progress and ready to work on immediately.

For detailed implementation instructions, see the individual task files listed above.
