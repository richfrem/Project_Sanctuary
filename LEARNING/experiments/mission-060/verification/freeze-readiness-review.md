### MISSION-060 Freeze Readiness Review

This review evaluates the prospective implementation files (`run_study.py`, `choose.py`, `analyze.py`, `thresholds.json`) against the approved MISSION-060 handoff and frozen protocol requirements. Six concrete correctness blockers have been identified that must be remediated prior to freezing `protocol.md` and `freeze.json`.

---

### Blocker 1: Item Budget & 25k Cap Exclude Successor and Reviewer Token Costs

* **Exact Code Anchor**:
  * `run_study.py#L193-L195`
  * `run_study.py#L381-L382`
  * `analyze.py#L447-L455`, `L472-L473`
* **Violated Requirement**:
  Approved handoff §1 ("Budget: ceiling 700,000 additional model tokens; 25,000 per item; at most 10 executed items") and Claim B1 ("B1 = P-only outputs per budget at least 1.5x F-only").
* **Mechanism**:
  * In `run_study.py`, `check_budget()` only checks `record["phase"] == "execution"`, ignoring tokens consumed by the successor (`succ-{code}`) and reviewer (`review-{code}`).
  * `cmd_item()` only queries `work-{code}` when writing `worker_usage` to `assess/{code}.json`.
  * In `analyze.py`, `cohort(c)` sums only `worker_usage["additional_tokens"]`. Incurred pipeline costs (successor Flash calls and reviewer Luna calls) are omitted from each cohort's token total, biasing $r_P$ and $r_F$ whenever cross-family review costs diverge.
* **Minimal Remedy**:
  In `cmd_item()`, collect token usage for `work-{code}`, `succ-{code}`, and `review-{code}`, storing a summed `total_additional_tokens` alongside the breakdown in `assess/{code}.json`. In `check_budget()` and `analyze.py`, evaluate the 25k cap and cohort token sums against the composite item tokens.

---

### Blocker 2: Coercion of Reviewer Truth Values Bypasses Boolean Type Invariant

* **Exact Code Anchor**:
  `analyze.py#L423-L424`:
  ```python
  ck = {k: bool(v.get("value")) if isinstance(v, dict) else bool(v) for k, v in a["review"]["checks"].items()}
  ```
* **Violated Requirement**:
  Approved handoff §4 step 3 and calibration standards require reviewer decisions to be typed JSON booleans (`True` / `False`), as checked in calibration (`run_study.py#L252-L253`: `type(a) is not bool or type(b) is not bool: raise ValueError`).
* **Mechanism**:
  In Python, `bool("false")`, `bool("null")`, and `bool("0")` evaluate to `True`. If an LLM reviewer outputs stringified values (e.g. `{"value": "false"}`), `is_spo()` coerces failed checks to `True`, converting failing outputs into Substantive Preserved Outputs (SPOs).
* **Minimal Remedy**:
  Enforce explicit boolean matching:
  ```python
  def _to_bool(v):
      val = v.get("value") if isinstance(v, dict) else v
      return val is True

  ck = {k: _to_bool(v) for k, v in a["review"]["checks"].items()}
  ```

---

### Blocker 3: Blinding Breach During Declarer Dispatch (`cwd` Omission)

* **Exact Code Anchor**:
  `run_study.py#L311-L313`:
  ```python
  for a in T["declarers"]:
      t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
      r = call(a, text, P.RATE_INSTRUCTION, "declaration", f"declare-{a}")
  ```
* **Violated Requirement**:
  Handoff §4 step 6, §6, and `attestation.json` guarantee: *"declarers run in the EMPTY neutral directory so they cannot see priority rankings; declarations/attestation.json records this"*.
* **Mechanism**:
  While `neutral_check()` verifies that `/private/tmp/study060/neutral` exists, `call()` in `cmd_declare()` does not pass `cwd=NEUTRAL`. The declarer CLI sessions run in the repository directory, where `SessionStart` hooks expose backlog metadata, formula priorities, and mission rankings directly to the declarers.
* **Minimal Remedy**:
  Pass `cwd=NEUTRAL` explicitly:
  ```python
  r = call(a, text, P.RATE_INSTRUCTION, "declaration", f"declare-{a}", cwd=NEUTRAL)
  ```

---

### Blocker 4: Unvalidated Declarer Outputs Cause Silent Dropout in Selection

* **Exact Code Anchor**:
  * `run_study.py#L314-L321`
  * `choose.py#L563-L565`:
    ```python
    if proposer.get(cid) == a or cid not in rows:
        continue
    ```
* **Violated Requirement**:
  Selection protocol requires every candidate in `pool.json` to be rated by every declarer (excluding an agent's own submitted ideas).
* **Mechanism**:
  `cmd_declare()` checks integer bounds on returned rows but does not assert that all expected candidate IDs are present. If an agent truncates or omits cards, `choose.py` silently drops the item (`cid not in rows`). The omitted item receives 0 percentile points and fewer ratings, distorting the rank-sum interest score without failing fast.
* **Minimal Remedy**:
  In `cmd_declare()`, assert complete candidate coverage before serializing declarations:
  ```python
  expected = {c["id"] for c in pool if c.get("proposed_by") != a}
  received = {r["id"] for r in rows}
  missing = expected - received
  if missing:
      raise SystemExit(f"STOP: Declarer {a} omitted items: {missing}")
  ```

---

### Blocker 5: Indeterminate State Logic Flaw in Verdict Decision Table

* **Exact Code Anchor**:
  `analyze.py#L486-L494`:
  ```python
  if not all(gates.values()):
      verdict = "UNASSESSED"
      why = [k for k, v in gates.items() if not v]
  elif B["B1"]["met"] is False or not B["B2"]["met"] or A.startswith("FALSIFIED"):
      verdict, why = "FALSIFIED", ...
  elif A.startswith("UNDECIDED") or A == "not run":
      verdict, why = "UNASSESSED", ["claim A undecided or not run"]
  else:
      verdict, why = "CORROBORATED", []
  ```
* **Violated Requirement**:
  Handoff §4 step 7 and §7: Verdict must be `CORROBORATED` only when both Claim A and Claim B (B1, B2, B3) are positively corroborated; unresolved or unverified metrics must resolve to `UNASSESSED`, never fall through.
* **Mechanism**:
  * If validity gates pass, but `B["B1"]["met"]` is `None` (e.g., edge evaluation path), `B["B1"]["met"] is False` evaluates to `False`. The check skips `FALSIFIED` and falls through to `CORROBORATED`.
  * For `B2`, `not B["B2"]["met"]` treats `None` (indeterminate usage) as `True`, improperly falsifying instead of classifying as `UNASSESSED`.
* **Minimal Remedy**:
  Structure the verdict ladder with explicit positive conditions for corroboration:
  ```python
  if not all(gates.values()):
      verdict, why = "UNASSESSED", [k for k, v in gates.items() if not v]
  elif B["B1"]["met"] is False or B["B2"]["met"] is False or A.startswith("FALSIFIED"):
      verdict, why = "FALSIFIED", [k for k, v in [("B1", B["B1"]["met"]), ("B2", B["B2"]["met"])] if v is False] + (["A"] if A.startswith("FALSIFIED") else [])
  elif B["B1"]["met"] is True and B["B2"]["met"] is True and A.startswith("CORROBORATED"):
      verdict, why = "CORROBORATED", []
  else:
      verdict, why = "UNASSESSED", ["indeterminable or incomplete claim results"]
  ```

---

### Blocker 6: Unhandled Review Structure Failures Crash Analysis

* **Exact Code Anchor**:
  `run_study.py#L370-L380` and `analyze.py#L423`:
  ```python
  ck = {k: bool(v.get("value")) if isinstance(v, dict) else bool(v) for k, v in a["review"]["checks"].items()}
  ```
* **Violated Requirement**:
  Audit integrity and robust evidence handling: Malformed model outputs must trigger explicit containment or invalid-evidence flags rather than unhandled Python runtime crashes during analysis.
* **Mechanism**:
  If `extract_json(rv["text"])` extracts a dictionary lacking a top-level `"checks"` key (or containing an unexpected structure), `a["review"]["checks"]` throws an unhandled `KeyError` or `AttributeError` when `analyze.py` executes.
* **Minimal Remedy**:
  In `cmd_item()`, validate that `isinstance(review, dict)` and `"checks" in review` before persisting; in `analyze.py`, use defensive access `a.get("review", {}).get("checks")` and report malformed assessments as unverified outputs.

---

### Summary Checklist for Freeze Readiness

| ID | Issue | Impact Area | State |
|---|---|---|---|
| **B-1** | 25k cap & B1 token denominator omit successor/reviewer | Budget & Claim B1 calculation | **Blocking** |
| **B-2** | `bool()` coercion on reviewer output strings | SPO false positives | **Blocking** |
| **B-3** | Missing `cwd=NEUTRAL` in `cmd_declare` | Blinding violation | **Blocking** |
| **B-4** | Incomplete declarer candidate return unhandled | Candidate rank-sum skew | **Blocking** |
| **B-5** | Fall-through logic to `CORROBORATED` on `met=None` | Verdict safety | **Blocking** |
| **B-6** | Unchecked `a["review"]["checks"]` access | Crash on malformed evidence | **Blocking** |

Remediating these six anchors brings the codebase into alignment with the approved handoff requirements, allowing Step 4 (`protocol.md` and `freeze.json`) to proceed safely.
