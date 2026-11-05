# Executable property falsification

`../scripts/counterexample_eval.py` is a stdlib-only runner for **trusted** small
Python components. A subprocess imposes a wall deadline on imports, generators,
components, and oracles. This is not a security sandbox: components must not
spawn descendants or access external resources. A child is killed/reaped on
expiry; arbitrary descendant processes and excessive memory are not contained.
Do not run hostile code. Successful outcomes are deterministic for a seed and
fixed Python/runtime/fixtures when the deadline does not intervene; timings and
deadline outcomes are environment-dependent.

## Claim contract

Write a component function accepting one JSON-compatible case and returning a
JSON-compatible value. The claim module must export:

```python
def generate(rng):          # claim-informed valid bounded cases
    return rng.randrange(100)

def example_generate(rng):  # preregistered ordinary example-generation baseline
    return rng.randrange(100)

def expected(case):        # independent specification/model, not component code
    return case + case

def check(case, actual):   # literal bool; may express other invariants
    return actual == expected(case)
```

Freeze the claim, domain, seed, both generators, budgets, thresholds, and split
before scoring. Inputs are deep-copied before component execution; an input
mutation cannot contaminate the oracle. Candidate exceptions on valid inputs
are counterexamples; oracle/generator/import failures are errors. There is no
automatic natural-language compiler, shrinker, or oracle-validity proof.
Properties must be reviewed independently of the implementation.

From the repository root:

```bash
python3 -B plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/counterexample_eval.py search \
  --component LEARNING/experiments/mission-041/fixtures/heldout.py:lease_strict_expiry \
  --claim LEARNING/experiments/mission-041/fixtures/claim_lease.py \
  --seed 41 --trials 64 --seconds 2

python3 -B plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/counterexample_eval.py benchmark \
  --manifest LEARNING/experiments/mission-041/fixtures/manifest.json \
  --split heldout --seed 41 --trials 64 --seconds 2 --component-seconds 600
```

The baseline compares independently specified expected values with component
outputs on ordinary generated examples. The property method evaluates `check`
on claim-informed examples. Both receive equal trial and time **ceilings**, not
necessarily equal actual work, since they stop on the first failure. Each target
uses the same seed for both methods. Aggregate component time includes both
methods, references, and non-equivalence witness checks. The runner serializes
searches; subprocess startup and termination add small scheduling overhead.

## Evidence semantics

- `COUNTEREXAMPLE`: witnessed rejection or candidate exception, with replay input.
- `EXHAUSTED`: the allotted evaluations found no failure; **not proof**.
- `TIMEOUT`: inconclusive; never detection, never proof of correctness.
- `ERROR`: broken evaluation infrastructure/property/import, not detection.

The manifest supplies separate development/heldout operators, target selectors,
and distinguishing witness cases. Witnesses are used only for curation, never
the scored search streams. A reference must pass and its mutant must fail the
witness. Failed curation does not silently remove mutants: it invalidates the
measurement and retains the denominator. Timeouts/errors also invalidate any
otherwise passing verdict. Reports retain every result, elapsed wall seconds,
budgets, inputs, and SHA-256 hashes of the runner/manifest/fixture sources.

Detection = killed mutants / all declared mutants in that split. Delta is the
property rate minus the example rate in percentage points. False alarms =
references rejected by property search / all component references. Baseline
reference results are retained and any baseline rejection invalidates the run.
The summary tests >=80% detection, >=20pp gain, <=5% false alarms, and <=600s
maximum component runtime. `THRESHOLDS_MET` applies only to the supplied fixture
set; it is not a full-mission or general correctness certification. Empty results
are invalid. Six references are too few to establish a population <=5% alarm rate.

## MISSION-041 evidence

See `LEARNING/experiments/mission-041/results.md`, `preregistration.md`,
`freeze.json`, and raw JSON reports. The fixture builder is provenance, not part
of scoring; do not regenerate or tune held-out fixtures after seeing outcomes.
The author-created holdout is procedurally separated, not independently blinded.
Full-domain reference enumeration is feasible for five components; replay uses
reduced exhaustive cases plus seeded samples. Run:

```bash
python3 -B LEARNING/experiments/mission-041/verify_references.py
PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest plugins/sanctuary-core/.apm/skills/evaluating-agent-work/tests/test_counterexample_eval.py
```

After supervisor integration, refresh installed APM copies with `apm install`
from the repository root. That operation is outside this worker's allowed scope.
