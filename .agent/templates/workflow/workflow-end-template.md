# Workflow End Checklist

**Date**: [DATE]
**Status**: Ready for Merge

## 1. Documentation Check (The Bookend)
- [ ] **Pre-Flight Verified**: Confirmed that `workflow-start` correctly identified the spec/branch.
- [ ] `sanctuary-start.md` exists and pre-flight checklist is complete
- [ ] `spec.md` is up to date vs reality
- [ ] `plan.md` reflects final architecture
- [ ] `tasks.md` is 100% checked off (Including Phase 0 and Phase N)
- [ ] `sanctuary-retrospective.md` is completed and linked

## 2. Technical Check
- [ ] All SOP steps from `tasks.md` are completed
- [ ] All tests passed (if applicable)
- [ ] Linter/Formatter run (if applicable)
- [ ] No temporary files left behind
- [ ] No broken links in documentation

## 3. Git Operations
- [ ] Committed all changes
- [ ] Pulled latest `main` (if long-running)
- [ ] Created Pull Request (or merged if authorized)
- [ ] Deleted feature branch (local) after merge
