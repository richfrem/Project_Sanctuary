# TASK: Harmonize Dockerfiles and Requirements (Protocol 129 Implementation)

**Status:** complete
**Priority:** High
**Lead:** Unassigned
**Dependencies:** Analyze Dependency Management for Protocol 129 ADR
**Related Documents:** None

---

## 1. Objective

Implement the standardized dependency management and Docker optimization strategy defined in the ADR.

## 2. Deliverables

1. Updated Dockerfiles for all fleet services (COPY after INSTALL).
2. Unified requirements structure (Core + Specialized).
3. **Strict Locking Implementation**: All `requirements.txt` generated from `.in` files.
4. **Dev/Runtime Separation**: Creation of `requirements-dev.txt`.
5. **CI Guardrail**: Check script to fail builds on inline `pip install`.

## 3. Acceptance Criteria

- All 8 Dockerfiles updated to use optimized COPY ordering.
- Requirements files consolidated or synchronized according to ADR 073.
- **Locking Verified**: All container requirements are pinned/locked.
- **CI Enforcement**: Manual `pip install` in Dockerfiles triggers a failure.
- **Local Sync**: Local dev environment is a safe superset of container deps.

## Notes

**Status Change (2025-12-23):** backlog → complete
Successfully implemented locked requirements (.in -> .txt workflow), synchronized local/container environments, harmonized Dockerfiles, and verified lean runtime via clean rebuild. All acceptance criteria met.
