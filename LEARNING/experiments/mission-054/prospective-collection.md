# MISSION-054 prospective collection: operational definitions (declared BEFORE collection, 2026-11-06)

Status of Claim B: UNASSESSED. This file does not change the manifest. It operationalizes terms the manifest leaves open, before any unit below is run, and is committed before collection starts.

## What counts as an eligible real work unit
1. Work that already needed doing (a documented failing test, defect or task), NOT created to produce evidence; no deliberately reintroduced faults; no benchmark or smoke test.
2. Executed through the installed shared loop (`work_loop.py run`) so attempts, usage and costs are in an attempt ledger.
3. Inside existing authorization, with machine-checkable acceptance criteria.
Fixture runs (including fault injection) are reported separately and never counted. Selection caveat: units are chosen by the orchestrator from known pre-existing failures, not at random; this limits what any rate can show.

## Measurements recorded per unit
Accepted outcome (criteria from the contract); execution, review, retry and verification costs from the attempt ledger (tokens measured where the backend/transcripts provide them, otherwise unknown; known totals are lower bounds); review tokens from the reviewer's own usage report; applicable failure opportunities (a targeted failure class applies only when the situation that causes it is present) and recurrences; human interventions by category (`required_authorization`, `necessary_clarification`, `execution_failure_correction`, `scope_change`) from observed user messages, coverage stated.

## What cannot be assessed yet
- Claim B(i) needs a frozen, comparable historical baseline of recurrence per targeted failure class. The 052 closeout records occurrences but not opportunity counts, so no defensible baseline rate exists: (i) stays UNASSESSED however these units turn out.
- The minimum usable denominators are not met by two units; Claim B stays UNASSESSED until 12 eligible units exist or the window is declared closed.
- Claim A (12 frozen held-out scenarios) has not been run.

## Source and installed revisions at declaration
Sanctuary `6a3d7d7a70c9`, companion `93f8689abb78`.
