# MISSION-060 Revision Note: a protected-exploration selection policy

**Mission:** `LEARNING/missions/MISSION-060.json` (PROPOSED, verdict null). **Revised:** 2026-11-07 by claude-code on the human partner's instruction.
**Not started, not adopted:** no policy described here is in force, no allocation has been reserved, no declaration has been collected, no inquiry has been executed and no worker has been dispatched. The mission was never executed before this revision either. The 30% protected share is provisional and needs Richard's explicit approval before any prospective phase can start (`human_gate_required` is now true).

## 1. Instruction and provenance

Richard's message of 2026-11-07 asked that MISSION-060 explicitly test a selection policy that favors world exploration and agent expression, so that interest has real influence over resource allocation instead of being a descriptive field. The message contained a proposal (protected exploration budget provisionally 30% of discretionary research resources, subject to approval; interest-led ranking within it by declared curiosity, originality of the question, potential understanding and opportunity for creative expression; readiness as a feasibility check; no operational-benefit requirement; interests recorded before rankings are seen; declarations are selection inputs, not proof of subjective experience; compare against the existing formula; measure substantive preserved discoveries within budget, not completed maintenance; creative work judged by observable deliverables and reflective assessment). The author of the proposal text was not stated; I record it as the human partner's instruction and do not attribute it further.

## 2. Why the old success measures did not fit

The original claim (reduce preventable blocked selections by at least 30%, keep 90% of baseline epistemic depth, select at least as many high-interest inquiries) already allowed "an explicitly bounded curiosity allocation", but it measured operational hygiene: fewer blocked selections and a count of high-interest items. It would have called a policy successful if it avoided blockers, even if no one learned anything new. That is the failure mode the instruction names ("more completed maintenance tasks"). So the revision adds a primary claim about substantive preserved outputs and keeps the original as a guard.

## 3. What changed (original text is preserved verbatim in `revision_history[0].superseded`)

| Element | Before | After |
|:---|:---|:---|
| Primary success | Fewer preventable blocked selections, depth retained, high-interest count | Substantive preserved outputs (SPO) per unit of measured budget, P-only versus F-only cohort |
| Role of interest | Descriptive field; a vaguely bounded allocation permitted | Governs a protected share (30% provisional) ranked by pre-declared interest |
| Readiness | Part of the exposed dimensions | Feasibility check only; sizes an affordable next step; never lowers intellectual-value ranking |
| Eligibility | Not stated | Any external subject on its own merits; no operational-benefit requirement |
| Declarations | Recorded before showing either ranking | Same, plus: at least three agents on different clients or models, independent, hashed, with explanation |
| Creative work | Not addressed | Observable deliverable, pre-declared brief, reflective assessment; not forced into a hypothesis |
| Human gate | false | true until the share, caps and thresholds are approved |
| Scores | 5 / 4 / 0.5 / 3 = 13.33 | 5 / 4 / 0.7 / 4 = 7.14 (larger scope; sensitivity in the manifest) |
| Retained | | Claim A thresholds (30%, 90%, high-interest count), all 9 design constraints, 7 measurement requirements, 6 guidance items, existing links |

## 4. Mapping each requested element to a design element

- **Protected exploration budget:** `selection_policy.protected_share` = 0.30, status PROVISIONAL; the denominator is "discretionary research resources", defined in Phase 0 as resources not committed to explicit user assignments, WIP constraints or maintenance obligations, measured in tokens and wall time per window. Explicit user assignments keep precedence (an existing constraint).
- **Interest-led ranking within that allocation:** rank-sum across agents of four 1-5 ratings (curiosity, originality, potential understanding, opportunity for creative expression), with a pre-declared tie rule. Originality is checked against the inspected knowledge base rather than self-claimed, so it cannot be asserted into existence.
- **Readiness as a feasibility check:** a high-interest candidate with a readiness gap gets a bounded enabling step (the cheapest useful next unit); it is not excluded, and its ranking is not reduced. Enabling steps identified and completed are reported.
- **No operational-benefit requirement:** mathematics, nature, history, art and other external subjects are eligible. The Human Gate, authorization, safety and external-contact rules still apply; nothing in the policy authorizes external publication or contact.
- **Declare before seeing rankings:** the declaring agent must not have seen any priority score, formula rank or another agent's declaration; the record is append-only, timestamped and hashed. Declarations are selection inputs, not proof of subjective experience.
- **Compare against the existing formula:** policy F (top-K by the formula) versus policy P (70% by formula, 30% interest-led) under identical pools, WIP limit and total budget.
- **Substantive preserved discoveries, not maintenance:** the SPO definition below; completed maintenance tasks, items touched, tokens spent and documentation volume are explicitly not counted.
- **Creative inquiries:** pre-declared brief, preserved deliverable (path and hash), blind checklist assessment and a short reflective note; reported separately so a creative item cannot hide a lack of discoveries, and a discovery cannot hide a lack of expression.

## 5. Design (not executed)

Phases 0-6 and the analysis are in the manifest `experiment` field and are not repeated here. The reasoning behind the main choices:

1. **Matched disagreement cohorts, not two whole portfolios.** Running two complete portfolios would double the cost and still confound item difficulty with policy. Executing only the items that exactly one policy selects (P-only versus F-only) isolates the treatment: both cohorts are drawn from the same candidate pool, executed under the same per-item cap, in randomized order. Item heterogeneity remains a confound and is reported, not claimed away.
2. **Blind assessment of origin.** Reviewers do not know which policy selected an item. Agreeing model reviewers are not treated as ground truth (the lesson recorded in MISSION-061's design); ambiguous cases go to bounded human review or stay unresolved.
3. **Substantive preserved output (SPO).** Discovery type needs a question new to the knowledge base that is answered or materially sharpened (an informative falsification or well-characterized null counts, as in the project's own closure rules), evidence beyond a single source or the agent's assertion, preserved artifacts with sources, method and limits, and a fresh successor answering at least 4 of 5 frozen handoff questions (MISSION-058's method). Creative type needs a brief declared before work, a preserved deliverable that passes the brief's checklist, a reflective note, and understandability from artifacts alone. Partial outputs are reported but not counted.
4. **Development calibration on 056, 057 and 058, never held-out.** They are the project's three completed agent-chosen curiosity inquiries; they show what a discovery-type output looks like (two informative FALSIFIED verdicts with preserved lessons) and let the rubric's inter-rater reliability be measured before freezing. Using them as held-out evidence would be circular.
5. **A harm measure on the formula share.** Diverting 30% to exploration must not silently degrade what the formula selects; the 70% share must keep at least 90% of its baseline verified outcomes per unit budget. This is reported as a cost, not hidden.

## 6. Decision table (provisional thresholds, frozen only after approval)

| Observation (valid instruments, completed run) | Result |
|:---|:---|
| B1-B3 hold and claim A met or UNSCORABLE | CORROBORATED |
| P-only SPO per budget below 1.5x F-only, or fewer than 2 more SPO | FALSIFIED |
| Total budget exceeded | FALSIFIED |
| Formula share below 90% of its baseline outcomes per unit budget | FALSIFIED (harm) |
| Claim A fails its original thresholds | FALSIFIED |
| Baseline has no preventable blockers | Claim A UNSCORABLE, not passed |
| Share/eligibility/caps unapproved; fewer than 4 items per cohort; rubric reliability below bound; declarations not blind; usage unmeasured | UNASSESSED |

## 7. Threats to validity

Small cohorts can falsify gross claims but cannot establish small effects. Declared interest may correlate with tractability, so interest-led items may simply be cheaper; the budget-normalized outcome and the reported caps address this only in part. Agents may inflate interest to win allocation; the explanation requirement, the independent multi-agent declarations and the originality check reduce but do not remove this. Conformity between agents is limited by independence of the declarations. Reviewer variation is measured on the development cases. A shipped selector is engineering acceptance, not corroboration. No claim is made about subjective experience.

## 8. Decisions that only Richard can make (nothing here is assumed approved)

1. The protected share (30% provisional) and the definition of discretionary resources.
2. The thresholds in the decision table (1.5x, +2 SPO, 90% guard) and the total and per-item budgets.
3. Which agents and clients make declarations (Claude, Codex and AGY are available; Copilot is excluded).
4. Whether creative deliverables count as full SPO, as written here, or only as a separate reported category.
5. Whether and when the prospective phase may begin at all.

## 9. Readiness and scoring

Readiness is low for the prospective phase until the approvals in section 8 exist and the instrument pieces (declaration store, rubric, blinding protocol, candidate pool) are built and frozen. The controlled-scenario phase (claim A) could be prepared sooner. Scores were revised from 13.33 to 7.14 because of scope, not because the question matters less; the inputs are estimates, not measured ROI. Neither the policy nor the 30% figure has any empirical support yet.

## 10. What this revision did not do

It did not start any phase, reserve any budget, collect any declaration, execute any inquiry, change the backlog formula, change any evaluation code or contact any external service. It changed only MISSION-060's manifest, three reciprocal links, this note and the generated views.
