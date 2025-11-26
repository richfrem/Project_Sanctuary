# MCP Architecture Refactoring: Separation of Concerns

**Date:** 2025-11-25  
**Type:** Architectural Refinement  
**Impact:** Document MCPs, Git Workflow MCP

---

## Problem Statement

Current architecture violates Single Responsibility Principle:
- Document MCPs (Chronicle, Protocol, ADR, Task) perform **two jobs**:
  1. ✅ File creation/modification (their core responsibility)
  2. ❌ Git operations (should be delegated)

---

## Proposed Solution

### Separation of Concerns

**Document MCPs** → File operations only
**Git Workflow MCP** → All Git operations

---

## Changes Required

### 1. Document MCPs (Chronicle, Protocol, ADR, Task)

#### Remove Git Operations
```typescript
// OLD (doing too much)
create_chronicle_entry(...): CommitResult ❌

// NEW (focused)
create_chronicle_entry(...): FileOperationResult {
  file_path: string,
  content: string,
  operation: "created" | "updated"
} ✅
```

#### Updated Tool Signatures

**Chronicle MCP:**
- `create_chronicle_entry()` → returns `FileOperationResult`
- `update_chronicle_entry()` → returns `FileOperationResult`
- Remove: Git commit operations

**Protocol MCP:**
- `create_protocol()` → returns `FileOperationResult`
- `update_protocol()` → returns `FileOperationResult`
- Remove: Git commit operations

**ADR MCP:**
- `create_adr()` → returns `FileOperationResult`
- `update_adr_status()` → returns `FileOperationResult`
- Remove: Git commit operations

**Task MCP:**
- `create_task()` → returns `FileOperationResult`
- `update_task_status()` → returns `FileOperationResult` (includes file move)
- Remove: Git commit operations

#### Updated Safety Rules
- Remove: "Auto-generates commit manifest"
- Remove: "Git operations with P101 compliance"
- Add: "Returns file path for Git Workflow MCP to commit"

#### Updated Shared Infrastructure
- Remove: `GitOperations` dependency
- Keep: `SafetyValidator`, `SchemaValidator`

---

### 2. Git Workflow MCP (Expanded)

#### New Tools

```typescript
stage_files(file_paths: string[]): StageResult {
  staged_files: string[],
  status: "staged"
}

commit_staged(
  message: string,
  manifest?: CommitManifest
): CommitResult {
  commit_hash: string,
  files_committed: string[],
  manifest_generated: boolean
}

commit_files(
  file_paths: string[],
  message: string,
  generate_manifest?: boolean
): CommitResult {
  commit_hash: string,
  files_committed: string[],
  manifest_generated: boolean
}
```

#### P101 Integration
- Generate commit manifests automatically
- Validate file integrity (SHA-256)
- Enforce commit message conventions

#### Updated Safety Rules
- Add: "Generates P101 manifests for all commits"
- Add: "Validates file paths before staging"
- Keep: "No destructive operations (no force-push, reset, etc.)"

---

## Workflow Patterns

### Single File Creation
```typescript
// Step 1: Create file (Document MCP)
const result = adr_mcp.create_adr(36, "MCP Architecture", ...)
// Returns: { file_path: "ADRs/036_mcp_architecture.md", operation: "created" }

// Step 2: Commit (Git Workflow MCP)
git_workflow.commit_files(
  [result.file_path], 
  "docs: create ADR #036 for MCP architecture"
)
```

### Multiple Files (Batch)
```typescript
// Step 1: Create multiple files
const adr = adr_mcp.create_adr(...)
const task = task_mcp.create_task(...)
const chronicle = chronicle_mcp.create_chronicle_entry(...)

// Step 2: Commit all together
git_workflow.commit_files(
  [adr.file_path, task.file_path, chronicle.file_path],
  "docs: create ADR #036, Task #037, Chronicle #281"
)
```

### Staged Workflow (Advanced)
```typescript
// Step 1: Create files
const files = [
  adr_mcp.create_adr(...).file_path,
  task_mcp.create_task(...).file_path
]

// Step 2: Stage files
git_workflow.stage_files(files)

// Step 3: Review, then commit
git_workflow.commit_staged("docs: architecture updates")
```

---

## Benefits

### 1. Single Responsibility Principle ✅
Each MCP does one thing well:
- Document MCPs: File operations
- Git Workflow MCP: Version control

### 2. Better Composability ✅
LLM can chain operations flexibly:
- Create multiple files before committing
- Choose when/if to commit
- Batch related changes

### 3. Centralized Git Logic ✅
- All Git operations in one place
- Easier to maintain P101 compliance
- Consistent commit patterns

### 4. Easier Testing ✅
- Test file creation separately from Git
- Mock Git operations easily
- Clearer unit test boundaries

### 5. More Flexible Workflows ✅
- LLM decides commit strategy
- Can create drafts without committing
- Can review before committing

---

## Files Requiring Updates

### Documentation (4 files)
- [ ] `architecture.md` - Update all Document MCP sections
- [ ] `final_architecture_summary.md` - Update workflow examples
- [ ] `ddd_analysis.md` - Add separation of concerns rationale
- [ ] `naming_conventions.md` - Update tool naming patterns

### Diagrams (5 files)
- [ ] `chronicle_mcp_class.mmd` - Remove GitOperations
- [ ] `protocol_mcp_class.mmd` - Remove GitOperations
- [ ] `adr_mcp_class.mmd` - Remove GitOperations
- [ ] `task_mcp_class.mmd` - Remove GitOperations
- [ ] `git_workflow_mcp_class.mmd` - Add new commit tools
- [ ] `mcp_ecosystem_class.mmd` - Update relationships

### Tasks (5 files)
- [ ] `029_implement_chronicle_mcp.md` - Update to file-only operations
- [ ] `030_implement_adr_mcp.md` - Update to file-only operations
- [ ] `031_implement_task_mcp.md` - Update to file-only operations
- [ ] `032_implement_protocol_mcp.md` - Update to file-only operations
- [ ] `035_implement_git_workflow_mcp.md` - Add commit operations

---

## Implementation Strategy

### Option 1: New Feature Branch (Recommended)
Create `feature/mcp-separation-of-concerns` branch:
1. Update all documentation
2. Update all diagrams
3. Update all task files
4. Create PR for review

### Option 2: Direct Update
Update current cleanup branch:
1. Apply all changes to current branch
2. Push updates
3. Merge with existing cleanup PR

---

## Risk Assessment

**Risk Level:** LOW (documentation only)
- No code changes required
- Architecture refinement, not redesign
- Improves clarity and maintainability

**Breaking Changes:** None
- This is pre-implementation
- No existing code to break

---

**Status:** Proposed - Awaiting Approval  
**Recommendation:** Proceed with Option 1 (new feature branch)
