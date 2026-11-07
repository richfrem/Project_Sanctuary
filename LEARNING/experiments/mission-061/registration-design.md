# MISSION-061 Registration Design Note

**Mission:** `LEARNING/missions/MISSION-061.json` (PROPOSED, verdict null). **Registered:** 2026-11-07 by claude-code from a draft selected by ChatGPT.
**Not started:** no pilot, evaluator inspection, case construction, labeling, worker dispatch or evaluation-code change has been performed. Registration changed only the manifest, justified reciprocal links, this note and generated views. Bibliographic claims below are motivation, not evidence for the hypothesis.

## 1. Reading theme and selection (attributed to ChatGPT)

Theme: "When does an agent actually know its work succeeded?" Selected as the primary paper: *Benchmarks as Software* (Terminal-Bench). Stated interest rationale (ChatGPT's, not measured): reliable learning requires knowing when the measuring instrument is wrong; a valid alternative rejected by an implementation-specific test is a useful counterexample, not a reason to force conformity. This connects Anti-Asch inquiry with practical engineering and leaves room for the hypothesis to fail.

## 2. Verified bibliographic details and reading summaries

All three abstract pages were fetched on 2026-11-07 and matched on title, authors and submission date (all submitted Thu 5 Nov 2026, v1, cs.AI).

| arXiv | Title | Authors | v1 (UTC) | Links |
|:---|:---|:---|:---|:---|
| 2611.08417 (primary) | Benchmarks as Software: Terminal-Bench and the Case for Continuous Benchmarks | E. Kelly Buchanan, Crystal Zhou, Scott W. Linderman, Christopher Re | 18:18:39 | https://arxiv.org/abs/2611.08417 , /pdf/2611.08417 , /html/2611.08417v1 , DOI 10.48550/arXiv.2611.08417 (registration pending), CC BY 4.0 |
| 2611.07689 | MCPHallu: Benchmarking Reasoning, Execution, and Memory Hallucinations in MCP Agents | Yizhen Jiang, Haowei Guo, Tianyi Bai, Zengjie Hu, Lu Ma, Zhengyang Zhao, Haoze Sun, Peng Pei, Wentao Zhang | 09:14:26 | https://arxiv.org/abs/2611.07689 , code https://github.com/J1zz/mcp_hallu |
| 2611.07652 | Don't Prompt Your Agents. Equip Them: Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems | Vladimiros Peilivanidis (ARPA Hellenic Logical Systems; sole author; cs.AI, cross-list cs.SE) | 08:38:51 | https://arxiv.org/abs/2611.07652 , code https://github.com/arpahls/skillware |

**Reading coverage and limits (disclosed).** Text was read through a fetch tool that summarizes pages with a small model, so extraction errors are possible and quotes were limited to short fragments. Terminal-Bench: the HTML text appeared complete through the appendices (figures were images, so only captions and stated values were visible); key figures below were cross-checked in a second pass. MCPHallu and Skillware: only the first 100,000 of about 179,580 and 238,096 characters were read; later sections (for Skillware 5.3-10 including the benchmark details, security section and references; for MCPHallu the end of the appendices) were NOT read. Nothing here is an independent reproduction.

### Primary: Benchmarks as Software (author-reported; verified passages)
- Introduces "benchmark maintenance debt" and a continuous-validation workflow applied to Terminal-Bench 2.0; issues found in 28 of 89 tasks (stated as "roughly a third"; 28/89 is derived); repairs released as Terminal-Bench 2.1.
- Audit rubric (Section 4) has three defect categories: instruction-verification mismatch, task drift, environment issues. The mismatch category includes "overspecified tests that reject behavior satisfying the task as written" (quote confirmed). Table 2 counts as read: 16 mismatch, 12 drift, 8 environment, tasks may carry several labels.
- Examples of strict or implementation-specific verifiers (paraphrased): a verifier required a database file to be preserved byte for byte though the instructions never said so (query-optimize, Section 5); for five tasks agents often produced valid solutions that TB 2.0 failed to recognize (Section 6.1); other benchmarks had graders rejecting correct reference solutions (Section 7). Appendix A.3 contrasts a version that edited instructions to match the test with TB2.1 relaxing the test to accept any correct solution.
- Repairs cut both ways: some only patched symptoms (relaxing a regex, broadening an acceptance condition) and some made tasks stricter by removing false-positive credit (Sections 1, 6.1). This is why the hypothesis requires no additional false acceptance.
- Accuracy claim: the abstract says "accuracy increased by over 10 percentage points"; the body does not report an aggregate that size. A linear fit implies a small average shift, and Section 8 gives up to 12 percentage points for an individual agent. Leaderboard positions moved by up to three ranks.
- Time and resource limits are part of the measurement (abstract: fixed wall-clock and output-token budgets can materially change apparent performance; Section 6.2: over half of failed trials were timeouts and budgets were swept up to 32x; binding resources were raised to at least 1.5x the verifier need). The abstract's 30-point gap on a smaller model that shrinks to 5 was not found in the body text and is unverified.
- Ground truth and its limits: two frontier-model auditors agreed on the primary defect category at kappa 0.80 over 89 tasks; "Agreement measures consistency rather than correctness"; maintainers gate every repair; workflow precision 57.1% and recall 71.4% were measured retrospectively against a release the workflow helped build. In the text as read there is NO labeled set of valid and invalid alternative solutions - which is the gap this mission targets.
- A likely typo ("over 77 pp above the fit" in Section 6.1) is not relied on.

### Supporting: MCPHallu (author-reported; partially read)
358 tasks, five domains, 36 production MCP servers in containers; 14 agents from eight vendors, 5,012 trajectories; average score 0.642. The Unreachable Goal subtype averaged 0.458 and 23.2% of its runs passed (the figure is tied to that subtype). A pass means correctly declaring infeasibility; extra tool calls are penalized (full score within a 2-call probing window, then linear decay). In the part read, zero-call runs passed 65.0% and runs with four or more calls 0%. Scoring is mostly deterministic with a 30% LLM-judge component for two subtypes. Relevance: an evaluator that scores "stop" versus "persist" encodes a design choice (the probing window) that could itself reject a defensible alternative policy.

### Supporting: Skillware (author-reported; partially read)
Bundles pair a typed contract, deterministic Python, directives and tests. The paper labels the model's skill selection and parameter filling "stochastic"; execution of the bundled code is deterministic. Tests check that `execute()` honors the manifest contract, not that business logic is correct. The author's group created both the framework and the benchmark, and no independent replication was seen in the part read. Relevance: deterministic code does not establish correct selection, valid evaluation or safe execution. No framework is adopted by this registration.

## 3. Hypothesis versus paper claims
The mission hypothesis is the registrant's, not any paper's. The papers motivate it; none reports a frozen held-out comparison of an original and a revised evaluator on a labeled solution set.

## 4. Hypothesis and decision table (verbatim hypothesis in the manifest)
| Frozen held-out outcome | Verdict |
|:---|:---|
| Original evaluator has zero false rejections | Non-diagnostic for the relative claim; cannot be CORROBORATED. Preserve the finding; do not invent a defect. |
| False-rejection reduction below 50% | FALSIFIED |
| Revised evaluator accepts any faulty implementation the original rejected | FALSIFIED |
| Counts or uncertainty below the frozen minimum | Undecided (no corroborated verdict) |
| Reduction at least 50%, no added false acceptance, counts and uncertainty met | CORROBORATED for the selected evaluator only |
Shipping a fix is separate from corroboration. No claim of generality beyond the selected evaluator or about the full Terminal-Bench suite.

## 5. Experimental design (not executed)
1. Inspect first (read-only): the primary paper in full, existing evaluation capabilities, fixtures and acceptance contracts. Reuse before adding.
2. Select ONE consequential, bounded evaluator and record the selection rationale in git BEFORE any confirmatory case is examined. Candidates to inspect (none selected): the criterion-level outcome evaluation in the companion `evaluating-agent-outcomes` skill; `counterexample_eval.py` from MISSION-041; the `eval_gate.py` adapter checks; staged-scenario assertions if MISSION-053 produces them.
3. Define a valid solution from the ORIGINAL task requirements, not from the evaluator implementation. Record which behaviors are required and which are incidental.
4. Build separate development and held-out sets of valid alternative implementations, genuinely faulty implementations and boundary cases. Expected outcomes come from independent means: executable checks where possible, bounded review for ambiguous cases. Ambiguous cases stay explicitly unresolved and are never silently relabeled. Do not author held-out labels to favor the repair. Two agreeing model reviewers are not objective ground truth (the primary paper's own caveat).
5. Develop any repair on development cases only.
6. Commit and hash-freeze: original evaluator, revised evaluator, held-out cases, expected outcomes, analysis plan, resource budget, minimum diagnostic counts and the uncertainty rule (to be specified at freeze; paired comparison on identical cases with interval estimates is a proposal, not yet frozen).
7. Run both evaluators on identical held-out cases once. Report false rejection, false acceptance, counts, uncertainty, runtime and review/repair costs.
Never run the full Terminal-Bench suite. Do not build a retrospective "held-out" set from the recent bootstrap repairs.

## 6. Reciprocal links and non-links
Linked (bidirectional, soft unless stated): MISSION-041 (supplies the mutant/equivalent-curation and freeze-before-scoring discipline and a candidate harness; 061 tests the converse error), MISSION-036 (its gate claims at most 5% false rejection; 061 offers a measurement method), MISSION-054 (owns the evaluation surface and shares paths; its savings claims presuppose valid verdicts), MISSION-055 (owns the companion/adapter boundary that decides where a generic evaluator fix lands; overlapping paths), MISSION-060 (061 records value, interest and readiness separately; illustrative, not evidence for 060). Not linked: MISSION-053 (its scenario assertions do not exist yet; revisit at selection); MISSION-015 and 049 (lexical overlap only).

## 7. Scoring and readiness (provisional)
Strategic leverage 5, epistemic depth 4, cost weight 0.4, friction 3; priority = (5 x 4) / (0.4 x 3) = 16.67. Estimates, not measured ROI; the leverage rests on an unverified premise that evaluator defects exist in the chosen evaluator. Readiness is separate from interest: moderate-to-low until an evaluator is selected. Missing prerequisites: the evaluator is not chosen; no labeled alternative-solution set exists; reviewer budget, minimum counts and the uncertainty rule are unspecified; the chosen evaluator may have no diagnostic defect (a legitimate, preserved outcome); possible lease overlap with IN_PROGRESS missions 054 and 055 on `evaluating-agent-work`.
