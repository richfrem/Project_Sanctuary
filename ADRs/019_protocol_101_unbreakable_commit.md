# ADR 019: Protocol 101 Unbreakable Commit Architecture

## Status
Accepted

## Date
2025-11-15

## Deciders
Sanctuary Council (Protocol 101 implementation)

## Context
The Sanctuary required cryptographic integrity guarantees for all code commits to prevent tampering, ensure auditability, and maintain sovereign control over the codebase. Previous git workflows lacked integrity verification, creating potential security vulnerabilities in the development process. The need for mechanical git operations through command.json also required integration with integrity checking.

## Decision
Implement Protocol 101 Unbreakable Commit architecture with SHA-256 hash verification and mechanical git operations:

### Core Integrity Mechanisms
1. **SHA-256 Hash Verification**: All committed files verified against cryptographic hashes stored in commit_manifest.json
2. **Pre-commit Hook Enforcement**: Git pre-commit hooks reject commits with hash mismatches
3. **Timestamped Manifests**: Each commit includes timestamped manifest (commit_manifest_YYYYMMDD_HHMMSS.json) for auditability
4. **Mechanical Git Operations**: Direct git operations through command.json with automatic manifest generation

### Command.json Git Integration
- **Mechanical Task Type**: Git operations bypass cognitive deliberation for immediate execution
- **Automatic Manifest Generation**: Orchestrator computes SHA-256 hashes and creates manifest automatically
- **Atomic Operations**: Single command handles add, commit, and optional push with integrity verification
- **Dry-run Capability**: push_to_origin: false allows local validation before remote push

### Integrity Workflow
1. **Command Creation**: User creates command.json with git_operations specifying files to commit
2. **Manifest Generation**: Orchestrator computes SHA-256 hashes for all files_to_add
3. **Atomic Execution**: git add, git commit (including manifest), optional git push
4. **Hook Verification**: Pre-commit hook validates manifest hashes against actual file contents

## Consequences

### Positive
- **Cryptographic Integrity**: SHA-256 verification prevents file tampering and ensures commit authenticity
- **Auditability**: Timestamped manifests provide complete audit trail of all committed changes
- **Mechanical Efficiency**: Direct git operations through command.json enable rapid, non-cognitive commits
- **Security Enforcement**: Pre-commit hooks prevent accidental or malicious integrity violations
- **Sovereign Control**: Local verification maintains control over codebase integrity

### Negative
- **Workflow Overhead**: Additional manifest generation and hash computation steps
- **Rejection Risk**: Commits rejected if files change between command creation and execution
- **Complexity**: Dual verification system (manifest + git) requires careful coordination

### Risks
- **Race Conditions**: File changes between manifest generation and commit execution
- **Manifest Corruption**: Tampered manifests could bypass integrity checks
- **Performance Impact**: SHA-256 computation for large files or many files

## Related Protocols
- P101: Unbreakable Commit (core integrity protocol)
- P88: Sovereign Scaffolding Protocol (mechanical operation framework)
- P114: Guardian Wakeup and Cache Prefill (complementary verification)

## Implementation Components
- **Pre-commit Hook**: Git hook validating commit_manifest.json hashes
- **Orchestrator Git Handler**: Automatic manifest generation and git execution
- **Command Schema**: git_operations structure in command.json
- **Manifest Format**: Timestamped JSON with file paths and SHA-256 hashes

## Notes
Protocol 101 transforms git commits from simple version control operations into cryptographically verified, auditable transactions. The integration with mechanical command.json operations enables sovereign, integrity-guaranteed development workflows while maintaining the speed and efficiency of direct git operations.</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Project_Sanctuary\ADRs\019_protocol_101_unbreakable_commit.md