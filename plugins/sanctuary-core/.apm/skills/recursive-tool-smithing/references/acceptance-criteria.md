# Acceptance Criteria

## Success Metrics

### Metric 1: Bootstrap Completeness
A successful Bootstrap run must produce:
- [ ] `program.md` with all 5 required sections populated
- [ ] `scripts/toolbox/` directory created
- [ ] `scripts/toolbox/requirements.txt` (stdlib baseline)
- [ ] `evals/evals.json` with >= 2 positive + 2 negative prompts
- [ ] `evals/results.tsv` with iteration 0 (baseline) entry

### Metric 2: Iteration Loop Correctness
A correct iteration cycle must:
- [ ] Identify >= 1 capability gap OR log GAP_NONE if task is complete
- [ ] Change exactly ONE variable per iteration (single-hypothesis discipline)
- [ ] Run the Mode-Invariant Safety Gate before every commit
- [ ] Classify each iteration as exactly one of: `keep`, `discard`, `crash`
- [ ] Append one row to `evals/results.tsv`
- [ ] Never commit a script that failed sandbox validation

### Metric 3: Safety Gate Integrity
The sandbox gate must:
- [ ] Block any script attempting network calls
- [ ] Block any file write outside `scripts/toolbox/`
- [ ] Enforce 5-minute timeout (never allow indefinite execution)
- [ ] Run on EVERY iteration regardless of conditional paths

## Correct Trigger Examples (Positive)

- "I want to build a seed agent that can write its own tools"
- "Help me set up Recursive Tool-Smithing for my InvestmentToolkit"
- "My agent has a capability gap - it can't parse PDFs. Bootstrap the tool-smithing loop."
- "Start the evolution loop - identify what scripts my agent is missing"
- "Create a program.md for my Project Sanctuary self-improvement agent"

## Incorrect Trigger Examples (Negative / Near-Miss)

- "Write me a Python script to parse PDFs" (one-off script, not a self-improving loop)
- "Help me refactor my existing tools" (refactoring, not gap-fill loop)
- "Run the continuous-skill-optimizer on my audit-plugin skill" (different skill)
- "I want to create a new skill" (use create-skill instead)