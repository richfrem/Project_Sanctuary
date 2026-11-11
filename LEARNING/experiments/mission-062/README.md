# MISSION-062 — finite sumset threshold experiment

**Scientific result: CORROBORATED within the registered finite region.** Every one of the 2,634 primary cells passed: k=3, h=5..16, q=3h..250. This is neither an asymptotic theorem nor a result for larger h/q or other k. The finite threshold observed through q=250 was **q0(h,3)=h+2 for h=2..16**, substantially below the sufficient proof bound. An algebraic explanation is an unstarted inquiry, not a proved generalization.

## Evidence and preregistration

- [protocol.md](protocol.md) and [freeze.json](freeze.json) were committed at `f7bc7c4e5bd0efa950b67a6d9fe252a28b69277b` in a separate command before running the evaluator.
- [results.json](results.json) contains the full decision, validity checks, finite thresholds and resource evidence. [runs/](runs/) contains complete raw frequency tables; the evaluator also checks that each table counts exactly binomial(q,k) literal subsets.
- [fast.c](fast.c) enumerates normalized shapes with weak compositions and exact integer markings, then integrates by span with translation multiplicity q-span. [reference.py](reference.py) independently enumerates actual subsets and iteratively constructs sumsets using Python sets. It neither normalizes nor uses weak compositions.
- The validity gate passed **375 independent full-histogram comparisons**, covering both k=3 and k=4, h=2..6, every q=k..40. All four published calibration thresholds reproduced through their published verification limits, including the immediately preceding failing cells. Arithmetic-progression checks passed for h=2..64; the interval-addition argument in the protocol covers the identity for all h>=1.
- The actual [v1 PDF](https://arxiv.org/pdf/2611.07303v1) was checked before freeze: formula, strict-dominance interpretation, proof bound and threshold table. The PDF SHA-256 is in freeze.json. v2 exists; this experiment deliberately retains the registration's v1. Lean files were not independently checked.

The primary result uses strict dominance of every listed candidate over every unlisted size, with ties and absent candidates failing. It does not impose a frequency ordering among the candidates.

## Secondary observations

For h=2..16, the passing suffix through q=250 begins at h+2. Thus q0/h falls from 2 to 1.125 across those finite cases. The descriptive log-log slope is 0.7424; the additive offset in h+2 makes this an unsuitable claim of a power-law exponent. For k=4 and h=2,3,4, the finite thresholds through q=80 were 8,16,24. These observations were secondary, do not rescue the primary claim, and do not establish infinite-domain thresholds.

A possible explanation for k=3 is that a normalized triple has a single primitive integer relation among its barycentric coefficient vectors, which could determine collision-chain sizes. This is a hypothesis for a future proof, not a demonstrated theorem here. The candidate was recorded in the existing inbox as SEED-20261111-e6ca6c; it was not started.

## OpenRouter orchestration and what actually helped

Richard requested the installed cli-agent path, low-cost OpenRouter execution, supervised work, and documented learning. JEV recommended a 3B micro model with score 0.022. The supervisor treated that weak recommendation as advisory and selected the live-catalog coding specialist **qwen/qwen3-coder-30b-a3b-instruct**. MISSION-071's smoke result established that installed dispatch plumbing works; it was not rerun or generalized into evidence of task competence.

The worker was output-only: no tools, filesystem, shell, git, or experiment execution. The parent owned file writes, independent reference, compiler/tests, protocol freeze, execution and verdict. `run_agent.py` has no OpenRouter backend; direct installed `openrouter_client.py` was used with fallback disabled, without claiming its launches had run_agent's claim/cwd guards.

Three drafts were returned:

1. The first failed the all-q output contract and -Werror compilation, and recomputed shapes unnecessarily.
2. The second compiled but enumerated a span equal to QMAX despite allocating only through H*(QMAX-1), causing a development-case buffer overrun.
3. The third fixed that boundary but still used hist=prefix instead of hist+=prefix, counting normalized shapes instead of translated subsets.

After the three-call cap, the supervisor made the one-line recurrence correction and added the source header, before the freeze. [development-checks.json](development-checks.json) records independent development acceptance. The **unmodified worker** failed its original exact-count contract, as recorded by the shared evaluator in [worker-assessment.json](worker-assessment.json). Scientific corroboration belongs to the validated, corrected instrument; it does not certify the worker's autonomous delivery.

The prediction of a complete correct counter in one worker call did not hold. A useful lesson is to test conservation laws and an independently different algorithm before trusting optimized counting code; compiling and returning plausible CSV is insufficient. This lesson is recorded as L-89dac80a. Future reuse/benefit is unverified. Generic lesson retrieval occurred after dispatch and returned no matches; it is not represented as a successful pre-dispatch learning intervention. No comparison with the JEV-recommended 3B model was run.

## Cost, resources and incidents

[usage.json](usage.json) records **4,284 input + 4,045 output = 8,329 worker tokens**, with **$0.0030696 client-priced cost** across three calls, matching the live catalog's $0.15/M input and $0.60/M output rates. All calls returned OK without fallback. This is token-based pricing, not an independently audited vendor invoice. Parent orchestration, review and repair tokens are unattributed; full mission-token-budget compliance and total financial cost therefore remain unverified. No measured ROI or total-cost advantage is claimed. The legacy scorer emits a cost-weighted rubric ratio using its inferred model tier; that value is not measured economic ROI and is not used as evidence of savings here.

The frozen validity, primary and secondary run took **3.762 s**, **3.453 CPU s**, with a conservative parent-plus-child maximum RSS of **28,344,320 bytes**. A separate 300-second upper-bound reserve for pre-freeze development compute was frozen; it is not presented as measured runtime. The required region was completed without shrinking it. There were no network/model calls inside evaluate.py.

Observed operational failures were retained:

- The worktree's default Python 3.13 failed TLS certificate validation during metadata lookup. Homebrew Python succeeded with TLS verification still enabled.
- The development sanitizer process paused in macOS crash handling; the parent terminated only that owned process after about 150 s. Its launcher lacked a timeout. Later instrument/test subprocesses are bounded; the paused process did not produce held-out evidence.
- The first freeze-commit attempt was correctly refused because generated backlog projections had not been compiled after checkout changed the manifest. The supported compile command fixed this before any held-out run.
- The first generic worker-assessment adapter invocation lacked a required failure condition. The contract wiring was corrected; its criteria map the original prompt and were authored after worker delivery, not preregistered as a model comparison.
- A metering invocation initially assumed a project-installed skill existed inside the isolated worktree. It did not; the canonical Sanctuary skill command was then used. This was an operator path error, not a model or scientific failure.

## Reproduction and delivery

Build with the compiler and flags recorded in freeze.json, verify all frozen hashes, then run evaluate.py. The historical result is retained; rerunning is not needed to recover it. Source integration, local checkout state, export and remote preservation are reported separately at closeout. No extra mission was launched to fill an evaluation sample.
