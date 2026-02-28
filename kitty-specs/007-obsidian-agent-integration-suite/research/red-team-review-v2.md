# Red Team Architectural Review v2: Obsidian Agent Integration Suite

**Feature Branch**: `007-obsidian-agent-integration-suite`  
**Reviewer**: Red Team (Claude Opus 4.6)  
**Review Date**: 2026-02-27  
**Bundle Version**: v2 (Generated 2026-02-27 06:17:10)  
**Prior Review**: v1 (2026-02-26) — 2 Critical, 3 High, 4 Medium, 3 Low findings

---

## Executive Summary

The v2 bundle represents a substantial and responsive rearchitecture. The Work Package count has expanded from 6 to 10, and **10 of the 12 v1 findings have been directly addressed**, several with impressive depth. The two Critical findings from v1 (concurrency safety and subtask ID collision) are fully resolved. The addition of WP05 (Markdown Mastery + shared `obsidian-parser`), WP07 (Dynamic Views), and WP10 (Integration Testing) closes the most dangerous capability and testing gaps identified in v1.

The v2 bundle introduces **0 new Critical** and **0 new High** findings. However, the rapid restructuring has left behind a trail of **stale cross-references** between documents that could cause execution confusion, and a few lower-severity gaps remain. This review identifies **1 High**, **3 Medium**, and **3 Low** severity findings.

---

## v1 Finding Resolution Tracker

| v1 ID | Severity | Finding | v2 Status | Resolution |
|---|---|---|---|---|
| CRITICAL-001 | Critical | No concurrency safety | **✅ Resolved** | WP06 adds atomic writes (T026), `.agent-lock` (T027), mtime detection (T028). WP09 adds snapshot isolation (T043). |
| CRITICAL-002 | Critical | Subtask ID collision | **✅ Resolved** | IDs are now globally unique T001–T050. No collisions. |
| HIGH-001 | High | Empty dependency fields | **⚠️ Partial** | Most WPs populated. WP02 still has `dependencies: []`. See HIGH-001v2. |
| HIGH-002 | High | 3 orphaned skills | **✅ Resolved** | WP05 covers Markdown Mastery, WP07 covers Bases + Canvas. All 6 skills now have implementation paths. |
| HIGH-003 | High | Wikilink parser underestimates complexity | **✅ Resolved** | WP05 T022 explicitly handles headings, block refs, aliases, embeds as distinct types. |
| MEDIUM-001 | Medium | No frontmatter error handling | **✅ Resolved** | WP09 T042 wraps frontmatter parsing in try/except with skip-and-warn. |
| MEDIUM-002 | Medium | Backlink resolution performance | **✅ Resolved** | WP08 T037 builds in-memory graph index with mtime invalidation. |
| MEDIUM-003 | Medium | Regex link refactoring fragility | **✅ Resolved** | WP04 T016 now excludes code fences and mandates `--dry-run`. |
| MEDIUM-004 | Medium | No integration testing | **✅ Resolved** | WP10 is a dedicated integration testing suite with concurrency simulation (T048). |
| LOW-001 | Low | Git pre-flight scope too narrow | **✅ Resolved** | WP09 T041 explicitly checks both the main repo and the vault root. |
| LOW-002 | Low | HF rate-limit unhandled | **✅ Resolved** | WP09 T045 implements explicit exponential backoff. |
| LOW-003 | Low | No shared vault path config | **❌ Open** | Carried forward. See LOW-001v2. |

**Scorecard**: 10/12 resolved, 1 partial (carried forward as HIGH-001v2), 1 open (carried forward as LOW-001v2).

---

## New Findings

### HIGH-001v2: Stale Cross-References Across Bundle Documents

**Category**: Execution Feasibility  
**Affected Components**: `red-team-prompt.md`, `plan.md` (Constitution Check), `research.md` (Findings Table)

The rapid restructuring from 6 WPs to 10 has left multiple documents referencing the old numbering scheme. An implementing agent following these references will pull context from the wrong Work Package.

Specific instances:

1. **`red-team-prompt.md`** still reads: "The 6 generated Work Packages (`WP01` through `WP06`)." There are now 10 WPs (WP01–WP10).

2. **`plan.md`**, Constitution Check section: "Ensure WP05 (Forge Soul) does not push to HF without a Git clean-state check." Forge Soul is now **WP09**, not WP05. WP05 in v2 is Markdown Mastery, which has no HF interaction. An agent reading this check will validate the wrong package.

3. **`research.md`**, Finding F-006 references: "WP03 (CRUD Skill) and WP06 (Refactoring) must strictly format outputs as `[[wikilinks]]`." In v2, CRUD is **WP06** and Refactoring is **WP04**. These labels now point to the wrong packages.

4. **`plan.md`**, Work Packages summary still opens with: "The implementation of `007-obsidian-agent-integration-suite` is structured into 6 sequential Work Packages." The text then correctly lists 10 — but the leading sentence is wrong and an agent may use it as a count validation.

While these are "just documentation bugs," in an agentic system where agents parse these documents as execution instructions, stale pointers are functional defects.

**Recommended Adjustment**: Perform a systematic cross-reference sweep across all bundle documents. Search for any instance of `WP0[0-9]` and verify it maps to the correct v2 package. Update the red-team-prompt.md to reflect the 10-WP scope. This is a 30-minute task that prevents hours of downstream confusion.

---

### MEDIUM-001v2: WP09 Missing Dependency on WP05 (Shared Parser)

**Category**: Execution Feasibility  
**Affected Components**: WP09

WP09 (Forge Soul) declares `dependencies: ["WP03", "WP06"]`. It depends on the HF schema mapping (WP03) and the CRUD primitives (WP06). However, the Forge Soul exporter must strip attachment embeds (`![[image.png]]`) and parse frontmatter using the shared `obsidian-parser` module, which is built in **WP05**.

WP09's T044 ("Payload Formulation — Strip all binaries") specifically needs the embed-vs-link discrimination logic from WP05's T022. Without the parser, WP09 would need to reinvent embed detection regex, violating the "single gatekeeper" principle the v2 architecture established.

**Recommended Adjustment**: Add `"WP05"` to WP09's dependency list: `dependencies: ["WP03", "WP05", "WP06"]`.

---

### MEDIUM-002v2: WP02 Dependency on WP01 Not Declared

**Category**: Execution Feasibility  
**Affected Components**: WP02

WP02 (Analyze Kepano Skills) has `dependencies: []`, yet its Context & Constraints section states: "any JS/TS plugin dependencies mapped by Kepano must be strictly evaluated against our 'Direct Filesystem Read' architecture rules." Those architecture rules are the output of WP01. If WP02 runs before WP01 completes, the analysis has no architectural framework to evaluate against, and its synthesis report may recommend patterns that conflict with the ADR.

Separately, WP04 (Legacy Scrubbing) also has `dependencies: []`. This is arguably correct — scrubbing dead links and MCP references doesn't depend on the ADR — but it would be worth adding a comment to the frontmatter confirming this is intentional, so an auditor doesn't flag it as an oversight.

**Recommended Adjustment**: Set WP02 to `dependencies: ["WP01"]`. Add an inline comment to WP04 confirming independent execution is intentional.

---

### MEDIUM-003v2: `.agent-lock` Protocol Detection Mechanism Undefined

**Category**: Security & State Integrity  
**Affected Components**: WP06 (T027)

WP06's T027 defines a `.agent-lock` file protocol and parenthetically mentions "or if we build a mechanism identifying the Obsidian desktop app is focusing the file." The `.agent-lock` file approach requires someone or something to create the lock — but the subtask doesn't specify who creates it, when, or how.

There are three plausible interpretations, each with different engineering requirements:

1. **Agent creates the lock before writing, deletes after** — This is a standard advisory lock. But it doesn't protect against Obsidian's concurrent writes; it only prevents two agents from colliding with each other.

2. **Human creates the lock manually** — Fragile. The user would need to remember to place and remove a dotfile whenever they open Obsidian, which defeats the purpose of autonomous operation.

3. **Agent detects Obsidian process** — The agent could check for a running Obsidian process via `pgrep obsidian` or inspect the vault's `.obsidian/workspace.json` lock. This is the most robust approach but is platform-specific and not specified.

The mtime detection (T028) provides a solid backup, but the `.agent-lock` protocol should have a clear owner and lifecycle defined.

**Recommended Adjustment**: Specify in T027 that the `.agent-lock` is a *bidirectional advisory lock* created by the agent before any write batch and removed after. Document that it does not protect against Obsidian writes (that's T028's job). Optionally, add a T027.5 subtask for process-level detection (`pgrep` or equivalent) as a "warm vault" warning signal, not a hard gate.

---

### LOW-001v2: No Shared Vault Path Configuration (Carried from v1)

**Category**: Execution Feasibility  
**Affected Components**: WP05, WP06, WP07, WP08, WP09, WP10

Six Work Packages now need the vault root path, but no shared configuration mechanism is defined. Each skill will independently parameterize it, creating drift risk if the vault moves or if testing requires switching between real and synthetic vaults.

**Recommended Adjustment**: Define a `plugins/obsidian-integration/config.yaml` or environment variable (`SANCTUARY_VAULT_PATH`) in WP05's scaffolding subtask (T020). All downstream skills import from this single source.

---

### LOW-002v2: Architecture Document Missing `obsidian-parser` Module

**Category**: Execution Feasibility  
**Affected Components**: `obsidian-plugin-architecture.md`

The `plan.md` directory layout correctly shows `obsidian-parser/` as a top-level module under `plugins/obsidian-integration/`. However, the architecture document (`obsidian-plugin-architecture.md`) still shows the old directory layout without it. Since the architecture document is the "Architectural Blueprint" that agents reference for structural decisions, it should reflect the shared parser module that is now a cornerstone of the design.

**Recommended Adjustment**: Update Section 2 of `obsidian-plugin-architecture.md` to include `obsidian-parser/` in the directory layout and add a Section 3.0 defining it as the shared utility that all skills depend on.

---

### LOW-003v2: WP10 Concurrency Test Doesn't Simulate Obsidian's Process

**Category**: Security & State Integrity  
**Affected Components**: WP10 (T048)

WP10's T048 tests 10 asynchronous agent threads competing against each other — validating that the `.agent-lock` and atomic write mechanisms prevent agent-vs-agent corruption. This is valuable, but it doesn't test the primary risk scenario: **agent writes vs. a concurrent Obsidian desktop process**.

Obsidian's file watcher and in-memory cache operate independently of any advisory lock the agent places. The real-world corruption vector is Obsidian overwriting agent changes from its stale cache, or the agent reading a half-flushed Obsidian write. This can't be fully simulated without running Obsidian, but a lightweight approximation is possible.

**Recommended Adjustment**: Add a test scenario to T048 where a background thread simulates Obsidian behavior: periodically reading and rewriting a note file (simulating Obsidian's auto-save) while agent threads attempt concurrent CRUD operations. Assert that mtime detection (T028) catches every simulated conflict and no silent overwrites occur. This won't be a perfect Obsidian simulation, but it exercises the defensive code paths.

---

## Summary Table

| ID | Severity | Category | Finding | Affected WPs |
|---|---|---|---|---|
| HIGH-001v2 | High | Execution Feasibility | Stale WP cross-references across 4+ documents from v1 numbering | red-team-prompt, plan, research |
| MEDIUM-001v2 | Medium | Execution Feasibility | WP09 missing dependency on WP05 (shared parser) | WP09 |
| MEDIUM-002v2 | Medium | Execution Feasibility | WP02 missing dependency on WP01 (architecture rules) | WP02 |
| MEDIUM-003v2 | Medium | Security & State Integrity | `.agent-lock` creation/lifecycle not specified | WP06 |
| LOW-001v2 | Low | Execution Feasibility | No shared vault path configuration (carried from v1) | WP05–WP10 |
| LOW-002v2 | Low | Execution Feasibility | Architecture document missing `obsidian-parser` module | Architecture doc |
| LOW-003v2 | Low | Security & State Integrity | Concurrency test doesn't simulate Obsidian's independent process | WP10 |

---

## Conclusion

This is a strong v2. The architecture has matured significantly — the shared `obsidian-parser` module, the layered concurrency defenses (atomic writes → mtime detection → snapshot isolation), and the dedicated integration testing suite address the most dangerous risks from v1 with real engineering depth.

The single High finding (stale cross-references) is a housekeeping task that should take under an hour. The three Medium findings are targeted adjustments to dependency declarations and protocol clarity. None of the v2 findings require architectural rethinking.

**Recommendation**: Fix HIGH-001v2 (stale references) and MEDIUM-001v2 + MEDIUM-002v2 (dependency declarations) before starting WP05. These are 5-minute YAML edits plus a document sweep. All other findings can be addressed during implementation. **The architecture is approved for execution.**
