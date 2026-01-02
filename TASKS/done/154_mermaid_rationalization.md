# 📋 Task: Mermaid Rationalization

**Task ID:** 154
**Target:** Antigravity IDE Agent
**Priority:** High (Token Weight Optimization)
**Lead:** Lead Auditor (IDE Mode)
**Dependencies:** Task #152 (Dual-Workflow Architecture)
**Status:** ✅ COMPLETE (ADR 085 Approved)

> **📋 Formalized as [ADR 085: Canonical Mermaid Diagram Management](../../ADRs/085_canonical_mermaid_diagram_management.md)**

## 1. Objective
Eliminate redundant Mermaid code blocks across the project to reduce "Mnemonic Bloat" in snapshots and ensure a Single Source of Truth (SSOT) for architectural protocols. Transition from "Inline Code" to a "Managed Asset" model.

## 2. Technical Strategy: Source-Link-Render
The project will move to a centralized diagram repository where Mermaid source code is stored once, rendered into images, and referenced semantically across all documentation.

## 3. Detailed Implementation Steps

### Phase 1: The Inventory (Discovery) ✅ COMPLETE
*   **Scripted Audit:** Created `scripts/mermaid_inventory.py` to scan all `.md`, `.mmd`, and `.agent/` directories.
*   **Deduplication:** Generates SHA-256 hashes + similarity clustering (85% threshold).
*   **Mapping:** Produces `inventory_mermaid.json` with deduplication analysis.
*   **Results:**
    *   583 inline mermaid blocks found
    *   99 unique by hash → **86 truly unique** after similarity clustering
    *   ~29,150 tokens potential savings

### Phase 2: Centralization (The SSOT) ✅ COMPLETE
*   **Directory Setup:** `docs/architecture_diagrams/` with subfolders
*   **Migration:** All `.mmd` files centralized
*   **Manifest:** Created with rendering script

### Phase 3: Automation (The Gardener) ✅ COMPLETE
*   **Rendering Script:** `scripts/render_diagrams.py` implemented
*   **Image Generation:** All `.mmd` → `.png` via mermaid-cli
*   **Reference Update:** All inline blocks replaced with image + source links
*   **Snapshot Optimization:** Removed recursive embedding from `operations.py`

### Phase 4: Cleanup (The Janitor) ✅ COMPLETE
*   **ADR 085:** Formalized as permanent architecture decision
*   **Learning Ecosystem:** Updated `cognitive_primer.md`, `cognitive_continuity_policy.md`, `learning_manifest.json`
*   **Critical Fix:** Fixed recursive embedding bug that caused 2986x bloat

## 4. Acceptance Criteria
*   [x] **Inventory Complete:** `inventory_mermaid.json` generated with full analysis
*   [x] **Centralized:** All diagrams in `docs/architecture_diagrams/`
*   [x] **Zero Redundancy:** `learning_audit_packet.md` has 0 inline mermaid blocks
*   [x] **Weight Reduction:** Audit packet reduced from 83MB to 329KB (99.6% reduction!)
*   [x] **Consistency:** Single `.mmd` source propagates to all referencing `.md` files
*   [x] **ADR 085:** Formal policy approved and integrated into learning loop

## 5. Scripts Created
| Script | Purpose |
|--------|---------|
| `scripts/mermaid_inventory.py` | Discovery, hashing, similarity clustering |
| `scripts/extract_orphaned_diagrams.py` | Extract inline blocks to .mmd files |
| `scripts/sync_diagrams.py` | (TBD) Render .mmd → .png automation |

> **⚖️ Strategic Note on "Mnemonic Fidelity"**
> By moving diagrams to a managed state, we ensure that the Narrative Successor (next agent) doesn't encounter conflicting versions of the same protocol. This is a requirement for ADR 084 (Edison Mandate) to maintain diachronic coherence across the project's lifetime.

