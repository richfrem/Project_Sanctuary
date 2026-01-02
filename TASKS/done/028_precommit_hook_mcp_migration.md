# Task #028: Pre-Commit Hook Migration for MCP Architecture

**Status:** In Progress  
**Priority:** Critical (Blocks Phase 1)  
**Estimated Effort:** 1-2 days  
**Dependencies:** None (Phase 0 - Pre-Migration)  
**Related Documents:** `docs/architecture/mcp/architecture.md`, `.agent/git_safety_rules.md`
**Analysis:** [Pre-Commit Hook Migration Analysis](../../docs/architecture/mcp/analysis/pre_commit_hook_migration_analysis.md)
**Plan:** [Implementation Plan](file:///Users/richardfremmerlid/.gemini/antigravity/brain/8e7a3729-cc05-40ae-a5dd-38935c512229/implementation_plan.md)

---

## Objective

Update existing pre-commit hooks to support the new MCP architecture while maintaining Protocol 101 (Unbreakable Commit) compliance. The current hooks enforce `command.json` workflows which will conflict with MCP-based operations.

---

## Problem Statement

**Current State:**
- Pre-commit hooks validate `command.json` format
- Hooks enforce specific commit message patterns for Council operations
- Hooks may block MCP-generated commits

**Required Changes:**
- Disable or adapt `command.json` validation
- Add MCP-aware commit message validation
- Maintain P101 manifest generation
- Support both legacy and MCP workflows during transition

---

## Deliverables

### 1. Updated Pre-Commit Hook
**File:** `.git/hooks/pre-commit`

**Changes:**
- [ ] Add MCP detection logic
- [ ] Bypass `command.json` validation for MCP commits
- [ ] Add MCP commit message format validation
- [ ] Maintain P101 manifest generation
- [ ] Add migration mode flag

### 2. MCP Commit Message Validation
**New validation patterns:**
```bash
# MCP commit format: mcp(<domain>): <description>
# Examples:
#   mcp(chronicle): create entry #283
#   mcp(forge): initiate guardian-02 training
#   mcp(git_workflow): create feature branch

MCP_COMMIT_PATTERN="^mcp\((chronicle|protocol|adr|task|cortex|council|config|code|git_workflow|forge)\): .+"
```

### 3. Migration Configuration
**File:** `.agent/mcp_migration.conf`

```bash
# MCP Migration Configuration
MCP_ENABLED=true
LEGACY_MODE=true  # Support both MCP and command.json
STRICT_MODE=false # Don't enforce MCP-only yet

# Validation settings
VALIDATE_COMMAND_JSON=false  # Disable for MCP commits
VALIDATE_MCP_FORMAT=true
GENERATE_P101_MANIFEST=true  # Always required
```

### 4. Documentation Updates
- [ ] Update `.agent/git_safety_rules.md` with MCP patterns
- [ ] Create `.agent/mcp_commit_guide.md`
- [ ] Update `docs/operations/git/git_workflow.md`

---

## Implementation Plan

### Phase 1: Analysis (Day 1, Morning)
- [ ] Audit current pre-commit hook logic
- [ ] Identify `command.json` dependencies
- [ ] Document all validation rules
- [ ] Create backup of current hooks

### Phase 2: Hook Updates (Day 1, Afternoon)
- [ ] Add MCP detection function
- [ ] Implement MCP commit message validation
- [ ] Add migration mode support
- [ ] Test with sample MCP commits

### Phase 3: Testing (Day 2, Morning)
- [ ] Test MCP commit formats
- [ ] Test legacy `command.json` commits
- [ ] Test P101 manifest generation
- [ ] Test error cases

### Phase 4: Documentation (Day 2, Afternoon)
- [ ] Update git safety rules
- [ ] Create MCP commit guide
- [ ] Update CI/CD documentation
- [ ] Create migration runbook

---

## Technical Details

### MCP Detection Logic

```bash
#!/bin/bash
# Detect if commit is MCP-generated

is_mcp_commit() {
    local commit_msg="$1"
    
    # Check for MCP commit pattern
    if echo "$commit_msg" | grep -qE "^mcp\([a-z_]+\):"; then
        return 0  # Is MCP commit
    fi
    
    # Check for MCP marker file
    if [ -f ".mcp_commit_marker" ]; then
        return 0  # Is MCP commit
    fi
    
    return 1  # Not MCP commit
}
```

### Updated Pre-Commit Hook Structure

```bash
#!/bin/bash
# Pre-commit hook with MCP support

set -e

# Load configuration
source .agent/mcp_migration.conf

# Get commit message
COMMIT_MSG_FILE=".git/COMMIT_EDITMSG"
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE" 2>/dev/null || echo "")

# Detect commit type
if is_mcp_commit "$COMMIT_MSG"; then
    echo "✓ MCP commit detected"
    
    # Validate MCP format
    if [ "$VALIDATE_MCP_FORMAT" = "true" ]; then
        validate_mcp_commit "$COMMIT_MSG"
    fi
    
    # Skip command.json validation
    echo "  Skipping command.json validation (MCP mode)"
else
    echo "✓ Legacy commit detected"
    
    # Validate command.json if required
    if [ "$VALIDATE_COMMAND_JSON" = "true" ]; then
        validate_command_json
    fi
fi

# Always generate P101 manifest
if [ "$GENERATE_P101_MANIFEST" = "true" ]; then
    generate_p101_manifest
fi

echo "✓ Pre-commit checks passed"
exit 0
```

### MCP Commit Message Validator

```bash
validate_mcp_commit() {
    local msg="$1"
    
    # Extract domain from commit message
    local domain=$(echo "$msg" | sed -n 's/^mcp(\([^)]*\)).*/\1/p')
    
    # Valid MCP domains
    local valid_domains="chronicle protocol adr task cortex council config code git_workflow forge"
    
    if ! echo "$valid_domains" | grep -qw "$domain"; then
        echo "❌ Invalid MCP domain: $domain"
        echo "   Valid domains: $valid_domains"
        exit 1
    fi
    
    # Check message format
    if ! echo "$msg" | grep -qE "^mcp\([a-z_]+\): .{10,}"; then
        echo "❌ MCP commit message too short"
        echo "   Format: mcp(<domain>): <description (min 10 chars)>"
        exit 1
    fi
    
    echo "  ✓ MCP commit format valid (domain: $domain)"
}
```

---

## MCP Commit Message Guide

**File:** `.agent/mcp_commit_guide.md`

```markdown
# MCP Commit Message Guide

## Format

```
mcp(<domain>): <description>
```

## Valid Domains

| Domain | Example |
|--------|---------|
| `chronicle` | `mcp(chronicle): create entry #283 - architecture complete` |
| `protocol` | `mcp(protocol): update P115 to v2.0` |
| `adr` | `mcp(adr): create ADR #037 - state machine pattern` |
| `task` | `mcp(task): move #030 to active status` |
| `cortex` | `mcp(cortex): ingest architecture documents` |
| `council` | `mcp(council): create deliberation for strategy` |
| `config` | `mcp(config): update API key for OpenAI` |
| `code` | `mcp(code): implement safety validator module` |
| `git_workflow` | `mcp(git_workflow): create feature/mcp-implementation` |
| `forge` | `mcp(forge): initiate guardian-02 training job` |

## Examples

**Good:**
```
mcp(chronicle): create entry #283 documenting MCP architecture completion
mcp(forge): initiate model training for guardian-02-v1
mcp(git_workflow): create feature branch for task-030
```

**Bad:**
```
mcp: update files  # Missing domain
mcp(invalid): test  # Invalid domain
mcp(chronicle): fix  # Description too short
```

## Migration Period

During migration, both MCP and legacy commit formats are supported:
- MCP commits: Use `mcp(<domain>):` format
- Legacy commits: Use conventional commit format
- Both generate P101 manifests
```

---

## Testing Checklist

### MCP Commit Tests
- [ ] Valid MCP commit with chronicle domain
- [ ] Valid MCP commit with forge domain
- [ ] Invalid domain rejection
- [ ] Short message rejection
- [ ] P101 manifest generation

### Legacy Commit Tests
- [ ] Legacy commit with command.json
- [ ] Legacy commit without command.json
- [ ] P101 manifest generation
- [ ] Conventional commit format

### Edge Cases
- [ ] Empty commit message
- [ ] Special characters in message
- [ ] Very long commit message
- [ ] Multiple MCP domains (should fail)

---

## Rollback Plan

If issues arise:

1. **Immediate Rollback:**
   ```bash
   git checkout HEAD -- .git/hooks/pre-commit
   ```

2. **Disable MCP Mode:**
   ```bash
   echo "MCP_ENABLED=false" > .agent/mcp_migration.conf
   ```

3. **Restore Backup:**
   ```bash
   cp .git/hooks/pre-commit.backup .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

---

## Success Criteria

- [ ] MCP commits pass pre-commit validation
- [ ] Legacy commits still work
- [ ] P101 manifests generated for all commits
- [ ] No false positives or negatives
- [ ] Documentation complete and clear
- [ ] Zero breaking changes to existing workflows

---

## Dependencies

**Required:**
- Git 2.x+
- Bash 4.x+
- Access to `.git/hooks/` directory

**Optional:**
- `jq` for JSON validation
- `shellcheck` for hook validation

---

## References

- [Protocol 101: Unbreakable Commit](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/.git/hooks/pre-commit)
- [Git Safety Rules](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/.agent/git_safety_rules.md)
- [MCP Architecture](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/docs/architecture/mcp/architecture.md)
- [Naming Conventions](../../docs/operations/mcp/naming_conventions.md)

---

**Created:** 2025-11-25  
**Author:** Guardian (via Gemini 2.0 Flash Thinking Experimental)  
**Version:** 1.0
