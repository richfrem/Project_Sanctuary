> Executor review: DRAFT recommendations only, not adopted or frozen. User authorized Flash workers and preparation delegation. Quota-rule changes remain pending explicit decision. Reject section E: the existing any-backend 25% stop rule is deliberate, not a false abort. Reject compiling thresholds.json with py_compile; validate JSON separately. Claimed source line numbers are inaccurate; confirmed anchors are choose.py:84 and run_study.py:55. Heartbeat return code must also be checked. Usage attribution still requires live validation and was not resolved by this plan. No study phases ran. Dispatch succeeded in 74.761 seconds; token usage was not reported and remains unknown. The generic control-plane status probe reports ENABLED/INCONSISTENT with no database or installed guards; no control-plane repairs were authorized or attempted. Study remains unfrozen pending quota resolution and applicable gate reconciliation.

### MISSION-060 Operational Preparation & Actionable Patch Plan

This plan prepares the study environment, rebalances assignments to AGY Gemini 3.8 Flash to protect depleted Claude and Codex quotas, and repairs instrument defects prior to freeze. **No study phases or held-out calls are executed.**

---

### 1. Mandatory vs. Optional Changes Summary

| Category | Component | Description | Priority |
| :--- | :--- | :--- | :--- |
| **Mandatory** | `run_study.py` | Fix hardcoded `--agent claude-code` in `heartbeat()` to match checkout agent (`codex` / active orchestrator). | **P0 (Crash / Lease Loss)** |
| **Mandatory** | `choose.py` | Fix modulo assignment `workers[(start + i) % 2]` which crashes with `IndexError` when `workers = ["flash"]`. | **P0 (Crash)** |
| **Mandatory** | `thresholds.json` | Set `scorer: "flash"`, `workers: ["flash"]`, `successor_for: {"flash": "flash"}`, `reviewer_for: {"flash": "luna"}`. | **P0 (Policy & Quota)** |
| **Mandatory** | `run_study.py` | Update `cmd_calibrate` raters from `("haiku", "flash")` to `("flash", "luna")` for independent cross-family calibration. | **P0 (Calibration)** |
| **Mandatory** | `run_study.py` | Filter `check_stop` headroom to active study backends; avoid false abort on excluded `copilot` or awaiting Luna reset. | **P0 (Premature Abort)** |
| **Mandatory** | Local Environment | Recreate missing local helper `/private/tmp/study060/codex-medium.sh` and ensure `/private/tmp/study060/neutral` is clean. | **P0 (Pre-condition)** |
| **Optional** | `common.py` | Add transcript detection fallbacks in `measure_agy` and `measure_claude` if timestamps or paths drift. | **P1 (Telemetry Robustness)** |
| **Optional** | `run_study.py` | In `cmd_item`, record successor and reviewer token usage into `assess/ITEM-NN.json` alongside worker usage. | **P1 (Auditability)** |

---

### 2. Exact Code Anchors and Proposed Replacements

#### A. Heartbeat Owner (`run_study.py`)
- **Issue:** Hardcoded `--agent claude-code`. The lease is checked out by `codex` per Section 4 Step 1. Calling heartbeat with `claude-code` fails to renew the lease, dropping lock ownership after 60 minutes.
- **Anchor (`run_study.py` lines 306–308):**
```python
def heartbeat():
    subprocess.run([sys.executable, str(COORD), "heartbeat", "--mission", "MISSION-060", "--agent", "claude-code", "--extend", "60"], capture_output=True)
```
- **Replacement:**
```python
def heartbeat():
    agent = os.environ.get("STUDY_AGENT", "codex")
    subprocess.run([sys.executable, str(COORD), "heartbeat", "--mission", "MISSION-060", "--agent", agent, "--extend", "60"], capture_output=True)
```

---

#### B. Worker Assignment Modulo (`choose.py`)
- **Issue:** `(start + i) % 2` hardcodes a 2-element worker list. With all-Flash workers (`["flash"]`), index `1` raises `IndexError`.
- **Anchor (`choose.py` lines 577–583):**
```python
    workers = T["assignment"]["workers"]
    assign = {}
    for cohort, start in (("P-only", 0), ("F-only", 1)):
        members = [cid for cid, co in combined if co == cohort]
        for i, cid in enumerate(members):
            assign[cid] = workers[(start + i) % 2]
```
- **Replacement:**
```python
    workers = T["assignment"]["workers"]
    assign = {}
    for cohort, start in (("P-only", 0), ("F-only", 1)):
        members = [cid for cid, co in combined if co == cohort]
        for i, cid in enumerate(members):
            assign[cid] = workers[(start + i) % len(workers)]
```

---

#### C. Configuration Rebalance (`thresholds.json`)
- **Issue:** Align with the approved rebalance: Flash workers, Flash successors, Flash scorer, Luna reviewer (cross-family: OpenAI vs. Gemini), Flash + Luna calibration.
- **Anchor (`thresholds.json` lines 85–88):**
```json
  "assignment": {"order_seed": 3060, "workers": ["haiku", "flash"], "reviewer_for": {"haiku": "flash", "flash": "sonnet"}, "successor_for": {"haiku": "flash", "flash": "sonnet"}},
  "declarers": ["sonnet", "flash", "luna"],
  "idea_generators": ["sonnet", "flash"],
  "scorer": "haiku",
```
- **Replacement:**
```json
  "assignment": {"order_seed": 3060, "workers": ["flash"], "reviewer_for": {"flash": "luna"}, "successor_for": {"flash": "flash"}},
  "declarers": ["sonnet", "flash", "luna"],
  "idea_generators": ["sonnet", "flash"],
  "scorer": "flash",
```

---

#### D. Calibration Pair Rebalance (`run_study.py`)
- **Issue:** Dev calibration currently runs `haiku` + `flash`. To preserve Claude headroom (26% remaining, 25% floor) and maintain independent family raters, switch raters to `flash` (Gemini) and `luna` (OpenAI).
- **Anchor (`run_study.py` lines 350–363):**
```python
    res = {}
    for rater in ("haiku", "flash"):
        for n, txt in cases.items():
            head = f"ITEM mission-{n}\nKIND: discovery\nQUESTION: see the artifacts below.\n\nARTIFACTS:\n{txt}"
            r = call(rater, head, P.CALIBRATION_INSTRUCTION, "calibration", f"cal-{rater}-{n}")
            res[(rater, n)] = extract_json(r["text"])["checks"]
    agree, total, detail = 0, 0, {}
    for n in cases:
        for ck in ("D1", "D2", "D3"):
            a, b = bool(res[("haiku", n)][ck]["value"]), bool(res[("flash", n)][ck]["value"])
            agree += a == b
            total += 1
            detail[f"{n}.{ck}"] = dict(haiku=a, flash=b)
    out = dict(agreement=agree / total, agree=agree, total=total, detail=detail, raters=["haiku", "flash"], cases=list(cases))
```
- **Replacement:**
```python
    res = {}
    for rater in ("flash", "luna"):
        for n, txt in cases.items():
            head = f"ITEM mission-{n}\nKIND: discovery\nQUESTION: see the artifacts below.\n\nARTIFACTS:\n{txt}"
            r = call(rater, head, P.CALIBRATION_INSTRUCTION, "calibration", f"cal-{rater}-{n}")
            res[(rater, n)] = extract_json(r["text"])["checks"]
    agree, total, detail = 0, 0, {}
    for n in cases:
        for ck in ("D1", "D2", "D3"):
            a, b = bool(res[("flash", n)][ck]["value"]), bool(res[("luna", n)][ck]["value"])
            agree += a == b
            total += 1
            detail[f"{n}.{ck}"] = dict(flash=a, luna=b)
    out = dict(agreement=agree / total, agree=agree, total=total, detail=detail, raters=["flash", "luna"], cases=list(cases))
```

---

#### E. Stop Rule Headroom Guarding (`run_study.py`)
- **Issue:** `headroom()` checks every backend returned by `usage_probe.py`. If `copilot` is below 25% (or Codex is at 20% prior to its reset), any call to Flash will fail immediately even though Gemini has 85% headroom.
- **Anchor (`run_study.py` lines 294–298):**
```python
def check_stop(key):
    h = headroom()
    low = {b: v for b, v in h.items() if v < T["stop_rules"]["min_backend_window_remaining_pct"]}
    if low:
        raise SystemExit(f"STOP RULE: backend window below {T['stop_rules']['min_backend_window_remaining_pct']}%: {low}. Pausing; ask the human partner.")
```
- **Replacement:**
```python
def check_stop(key):
    h = headroom()
    # Only monitor backends participating in the study (excludes Copilot and uncalled backends)
    study_backends = {"claude", "agy", "codex"}
    target_backend = BACKEND_OF[key]
    if target_backend in h and h[target_backend] < T["stop_rules"]["min_backend_window_remaining_pct"]:
        raise SystemExit(f"STOP RULE: target backend {target_backend} window below {T['stop_rules']['min_backend_window_remaining_pct']}% (remaining {h[target_backend]}%). Pausing; ask the human partner.")
    low = {b: v for b, v in h.items() if b in study_backends and v < T["stop_rules"]["min_backend_window_remaining_pct"]}
    if low and target_backend in low:
        raise SystemExit(f"STOP RULE: backend window below {T['stop_rules']['min_backend_window_remaining_pct']}%: {low}. Pausing; ask the human partner.")
```

---

#### F. Recreate Machine-Local Helpers (Environment Setup)
Before freeze and execution, ensure machine-local paths referenced by `common.py` exist:
1. **Codex medium wrapper** (`/private/tmp/study060/codex-medium.sh`):
```bash
#!/bin/sh
exec codex -c model_reasoning_effort="medium" "$@"
```
*(must be `chmod +x`)*.
2. **Neutral directory**:
```bash
mkdir -p /private/tmp/study060/neutral
cd /private/tmp/study060/neutral && git init -q
```

---

### 3. Protocol Disclosure Requirements (for `protocol.md`)
Record the following disclosed adaptations prior to freeze:
1. **Client Rebalancing:** Workers, successors, and non-manifest formula scorer assigned to AGY Gemini 3.8 Flash (`gemini-3.8-flash-medium`) to conserve depleted Claude (26%) and Codex (20%) quotas.
2. **Reviewer Independence:** Reviewer assigned to Codex GPT-6 Luna (`gpt-6-luna`), ensuring strict cross-family evaluation (OpenAI evaluating Gemini deliverables). Reviews run after Codex window reset (~07:04Z).
3. **Calibration Independence:** Calibration raters rebalanced to Flash + Luna (Gemini + OpenAI), ensuring inter-family calibration without Claude token consumption.
4. **Heartbeat & Coordination:** Lease owner attributed to the active orchestrator.

---

### 4. Zero-Held-Out Validation Plan

Run these verification steps without executing any study phase or spending held-out tokens:

1. **Static Analysis & Syntax Check:**
   ```bash
   python3 -m py_compile thresholds.json common.py run_study.py choose.py prompts.py pool.py analyze.py scenarios.py
   ```
2. **Deterministic Selection Simulation:**
   Run `choose.py` using dummy or existing dev data to verify:
   - Worker assignment completes without `IndexError`.
   - All `workers` in `selection.json` map to `"flash"`.
   - All `reviewer` entries map to `"luna"`.
   - All `successor` entries map to `"flash"`.
3. **Usage Probe Headroom Verification:**
   Verify `run_study.py` correctly parses headroom and does not trip false positives on uncalled backends:
   ```bash
   python3 -c "import run_study; print(run_study.headroom())"
   ```
4. **Neutral Directory Invariant Check:**
   Verify `neutral_check()` passes cleanly:
   ```bash
   python3 -c "import run_study; run_study.neutral_check(); print('Neutral OK')"
   ```
5. **Freeze Preparation:**
   Generate `freeze.json` (SHA256 of all study scripts and `protocol.md`) and verify all hashes match before running `scenarios.py heldout` or prospective phases.
