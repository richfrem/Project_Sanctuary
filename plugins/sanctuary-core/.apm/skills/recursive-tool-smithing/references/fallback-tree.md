# Highly Procedural Fallback Trees

## Primary Method Failure Points

### Failure: Script generation produces syntax errors

```
1. Run: python3 -m py_compile scripts/toolbox/<script>.py
2. If SyntaxError:
   a. Attempt one self-correction pass (fix the specific reported line).
   b. Re-run py_compile.
   c. If still fails: classify `crash`, log reason, delete candidate, narrow hypothesis.
   d. Do NOT attempt more than 1 self-correction per iteration.
```

### Failure: Sandbox timeout (> 5 minutes)

```
1. Kill the process (os.kill or timeout wrapper).
2. Classify `crash`.
3. Log: "TIMEOUT: script exceeded 5-minute sandbox limit"
4. Next hypothesis: reduce algorithm scope (e.g., process 1 item instead of batch).
5. Resume loop from last known good.
```

### Failure: No capability gap found (all tools available)

```
1. Log: "GAP_NONE: task completed with existing toolbox"
2. Run baseline benchmark to confirm no regression.
3. Update program.md "Current Gaps" table: mark solved.
4. Propose: run continuous-skill-optimizer to improve trigger descriptions.
5. Report cycle as SUCCESS.
```

### Failure: evals/evals.json missing or malformed

```
1. Check file exists and is valid JSON.
2. If missing: generate default eval set (2 positive, 2 negative prompts) from skill description.
3. If malformed: log error, use inline default evals, flag for human fix.
4. Continue loop with fallback evals.
```

### Failure: program.md missing in Iteration Mode

```
1. Log: "BOOTSTRAP_REQUIRED: program.md not found"
2. Switch to Bootstrap Mode.
3. Run Discovery Phase interview.
4. Do NOT attempt gap-fill until program.md is written and confirmed.
```

### Failure: Git commit fails

```
1. Check: are there unstaged changes? Run git status.
2. If merge conflict: halt, present conflict to user.
3. If hook failure: read hook output, fix the reported issue (e.g., encoding), retry once.
4. If still fails: store script as `scripts/toolbox/<name>.py.pending` and alert human.
5. Do NOT use --no-verify.
```

## Anti-Patterns (Vaccination List) {#anti-patterns}

These are known failure modes. Before committing any script, verify it does NOT exhibit:

| Anti-Pattern | What It Looks Like | Fix |
|-------------|-------------------|-----|
| Hardcoded paths | `open("/Users/richard/data.csv")` | Use relative paths from `__file__` |
| Swallowed exceptions | Bare `except: pass` | Catch specific exceptions, log them |
| No `--test` block | Missing `if __name__ == "__main__"` | Always add a self-contained test |
| Mutating global state | Writing to files outside toolbox | Enforce sandbox scope |
| No type hints | `def process(data):` | All signatures must have annotations |
| Magic numbers | `if len(data) > 47:` | Name constants at module level |
| Dynamic eval | `eval(user_input)` | Never allowed - sandbox violation |
