# Separation of Concerns - Quick Reference

## Document MCPs (Chronicle, Protocol, ADR, Task)

**Responsibility:** File operations only

**Returns:**
```typescript
FileOperationResult {
  file_path: string,
  content: string,
  operation: "created" | "updated" | "moved"
}
```

**Dependencies:**
- SafetyValidator ✅
- SchemaValidator ✅
- ~~GitOperations~~ ❌ (removed)

---

## Git Workflow MCP

**Responsibility:** All Git operations

**New Tools:**
```typescript
commit_files(file_paths[], message) → CommitResult
stage_files(file_paths[]) → StageResult  
commit_staged(message) → CommitResult
```

**Features:**
- P101 manifest generation
- File integrity validation
- Batch commits

---

## Workflow Pattern

```typescript
// 1. Create file (Document MCP)
const result = adr_mcp.create_adr(...)
// Returns: { file_path: "ADRs/036.md" }

// 2. Commit (Git Workflow MCP)
git_workflow.commit_files([result.file_path], "docs: add ADR #036")
```

---

**Benefits:** Single responsibility, better composability, centralized Git logic
