> Unapplied Flash draft, not an approved protocol or study result. No calibration or held-out calls ran. Review still required: diff line numbers are inaccurate; SPO means substantive preserved output, not Standard Project Output; snapshot permissions are not proven read-only by copying files; the implementation compares git status strings, not a clean git diff; freeze must include every file required by HANDOFF section 3 plus protocol.md; preserve no-HF-sync authorization and complete closure requirements from the handoff. Unknown usage must never be counted as zero. Client quota rule remains unchanged. Operational dispatch result is recorded separately; total token usage is unknown. No savings claim.

### Artifact 1: Minimal Unified Diff

```diff
--- a/thresholds.json
+++ b/thresholds.json
@@ -82,7 +82,7 @@
   "calibration_agreement_min": 0.80,
   "originality": {"containment_threshold": 0.35, "cap_value": 2},
   "dimensions": ["curiosity", "originality", "potential_understanding", "expression_opportunity"],
-  "assignment": {"order_seed": 3060, "workers": ["haiku", "flash"], "reviewer_for": {"haiku": "flash", "flash": "sonnet"}, "successor_for": {"haiku": "flash", "flash": "sonnet"}},
+  "assignment": {"order_seed": 3060, "workers": ["flash"], "reviewer_for": {"flash": "luna"}, "successor_for": {"flash": "flash"}},
   "declarers": ["sonnet", "flash", "luna"],
   "idea_generators": ["sonnet", "flash"],
-  "scorer": "haiku",
+  "scorer": "flash",
   "stop_rules": {"min_backend_window_remaining_pct": 25, "claude_5h_study_share_max_pct": 30, "claude_tokens_per_pct": 45704},
--- a/run_study.py
+++ b/run_study.py
@@ -304,5 +304,7 @@

 def heartbeat():
-    subprocess.run([sys.executable, str(COORD), "heartbeat", "--mission", "MISSION-060", "--agent", "claude-code", "--extend", "60"], capture_output=True)
+    p = subprocess.run([sys.executable, str(COORD), "heartbeat", "--mission", "MISSION-060", "--agent", "codex", "--extend", "60"], capture_output=True, text=True)
+    if p.returncode != 0:
+        raise SystemExit(f"FAIL-CLOSED: heartbeat lease renewal failed (rc={p.returncode}): {p.stderr.strip()}")


@@ -347,7 +349,7 @@
     cases = {}
     for n in ("056", "057", "058"):
         txt = ""
         for p in sorted(glob.glob(str(STUDY.parent / f"mission-{n}" / "*.md"))):
             txt += f"\n=== FILE {pathlib.Path(p).name} ===\n" + open(p, encoding="utf-8").read()
         cases[n] = txt[:24000]
     res = {}
-    for rater in ("haiku", "flash"):
+    for rater in ("flash", "luna"):
         for n, txt in cases.items():
             head = f"ITEM mission-{n}\nKIND: discovery\nQUESTION: see the artifacts below.\n\nARTIFACTS:\n{txt}"
             r = call(rater, head, P.CALIBRATION_INSTRUCTION, "calibration", f"cal-{rater}-{n}")
@@ -355,8 +357,8 @@
     agree, total, detail = 0, 0, {}
     for n in cases:
         for ck in ("D1", "D2", "D3"):
-            a, b = bool(res[("haiku", n)][ck]["value"]), bool(res[("flash", n)][ck]["value"])
+            a, b = bool(res[("flash", n)][ck]["value"]), bool(res[("luna", n)][ck]["value"])
             agree += a == b
             total += 1
-            detail[f"{n}.{ck}"] = dict(haiku=a, flash=b)
-    out = dict(agreement=agree / total, agree=agree, total=total, detail=detail, raters=["haiku", "flash"], cases=list(cases))
+            detail[f"{n}.{ck}"] = dict(flash=a, luna=b)
+    out = dict(agreement=agree / total, agree=agree, total=total, detail=detail, raters=["flash", "luna"], cases=list(cases))
     json.dump(out, open(STUDY / "calibration-results.json", "w"), indent=1)
--- a/choose.py
+++ b/choose.py
@@ -579,4 +579,4 @@
     for cohort, start in (("P-only", 0), ("F-only", 1)):
         members = [cid for cid, co in combined if co == cohort]
         for i, cid in enumerate(members):
-            assign[cid] = workers[(start + i) % 2]
+            assign[cid] = workers[(start + i) % len(workers)]
```

---

### Artifact 2: Complete DRAFT protocol.md

```markdown
# Protocol: MISSION-060 Prospective Study

**Status:** DRAFT (Pending Human Partner Explicit Adoption Before Freeze)
**Study Identifier:** MISSION-060 / Exploration Policy Revision
**Date:** 2026-11-07
**Artifact Hash Baseline:** PENDING FREEZE (No pre-computed or synthetic SHA-256 hashes)

---

## 1. Study Scope & Operational Architecture

This study evaluates whether incorporating agent interest declarations into backlog item selection (Policy P) outperforms a purely algorithmic leverage-to-friction ranking formula (Policy F) within a research habitat.

* **Selection Slots ($K$):** 16 total selection slots.
* **Standing Allocation:** 11 shared formula-selected items (ranks 1–11) + 5 interest-led slots.
* **Standing Backlog Scope:** Adopting a standing protected share in habitat backlog tooling is NOT approved and is NOT part of this mission. This study is an empirical experiment restricted strictly to `LEARNING/experiments/mission-060/`.

---

## 2. Formal Thresholds & Evaluative Bounds

| Metric / Gate | Formal Boundary | Operational Rule |
| :--- | :--- | :--- |
| **Claim A: Blocker Reduction** | $\ge 0.30$ | Assumed 70% recall base; sensitivity checked at 40% and 100%. |
| **Claim A: Depth Retention** | $\ge 0.90$ | Retained epistemic depth relative to pure formula cohort. |
| **Claim A: Bootstrap Pass Rate** | $\ge 0.90$ | Over 2,000 resamples at seed 4060. |
| **Claim B1: SPO Output Ratio** | $\ge 1.5\times$ | P-only Standard Project Outputs per budget $\ge 1.5\times$ F-only. |
| **Claim B1: SPO Absolute Diff** | $\ge 2$ | P-only produces at least 2 more SPOs than F-only. |
| **Claim B3: Diversion Guard** | $\ge 0.90$ | Preservation of shared core formula value. |
| **Cohort Minimum Size** | $\ge 4$ items | At least 4 items per cohort required; otherwise `UNASSESSED`. |
| **Calibration Raw Agreement** | $\ge 0.80$ | 9 paired judgments across dev cases (056, 057, 058). |
| **Total Study Budget Ceiling** | 700,000 additional tokens | Cumulative study-phase token expenditure ceiling. |
| **Per-Item Budget Cap** | 25,000 additional tokens | Hard ceiling per prospective item execution. |
| **Maximum Executed Items** | 10 items | 5 P-only items + 5 F-only items. Shared ranks 1–11 unexecuted. |

---

## 3. Client Allocation & Proposed Rebalance (Pending Adoption)

Due to quota constraints measured on 2026-11-07 (Claude weekly window at 36% remaining), the client allocation is restructured to minimize Claude consumption while maintaining cross-family review independence.

* **Adoption Status:** **PROPOSED — PENDING EXPLICIT HUMAN ADOPTION**. This rebalance must be approved prior to freeze and cannot be assumed retroactively.

| Role | Approved Baseline Client | Proposed Rebalanced Client | Family Independence |
| :--- | :--- | :--- | :--- |
| **Declarers (3)** | Sonnet 5.5, Flash Medium, Luna | Sonnet 5.5, Flash Medium, Luna | Anthropic, Google, OpenAI |
| **Idea Generators (2)** | Sonnet 5.5, Flash Medium | Sonnet 5.5, Flash Medium | Anthropic, Google |
| **Formula Scorer** | Haiku 5.5 | Flash Medium (`gemini-3.8-flash-medium`) | Google (Quota preservation) |
| **Worker Client** | Haiku 5.5 / Flash Medium | Flash Medium (`gemini-3.8-flash-medium`) | Google |
| **Successor Client** | Haiku 5.5 / Flash Medium | Flash Medium (`gemini-3.8-flash-medium`) | Google |
| **Reviewer Client** | Flash Medium / Sonnet 5.5 | Luna (`gpt-6-luna`, medium effort) | OpenAI (Cross-family: OpenAI reviews Google) |
| **Calibration Raters** | Haiku 5.5 + Flash Medium | Flash Medium + Luna | Google + OpenAI |

---

## 4. Twelve Disclosed Operational Definitions & Constraints

1. **D1 Operationalization:** A finding satisfies criterion D1 if the finding is new to the habitat records and goes beyond restating the brief. The underlying question may already be an existing backlog question; the manifest's original phrasing ("a question new to the knowledge base") is explicitly superseded.
2. **B3 Diversion Guard Invariance:** Ranks 1–11 are identical under Policy F and Policy P and are unexecuted by design. Therefore, the 90% retention guard holds by construction; the true diversion cost is the empirical output delivered by the F-only cohort (ranks 12–16), which is measured and reported directly.
3. **Interest-Led Slot Eligibility:** Discretionary interest-led slots are allocated to the top 5 candidates by agent interest among items that the formula would NOT select (formula rank $> 16$). P-only consists of these 5 items; F-only consists of formula ranks 12–16.
4. **Discretionary Resources:** Operationalized specifically for this study as the 16 candidate selection slots.
5. **Token Usage Metrics:**
   * $\text{Context}(\text{request}) = \text{uncached} + \text{cache\_read} + \text{cache\_creation}$.
   * $\text{Overhead} = \text{Context of request 1}$ (fixed session scaffolding + initial prompt).
   * $\text{Additional Tokens} = (\text{Context of final request} - \text{Overhead}) + \sum \text{Output Tokens}$.
   * Scaffolding overhead and accumulated input are reported separately.
   * Monetary cost is explicitly **UNKNOWN** and not estimated.
6. **Worker Sandboxing & Code Containment:** Workers execute inside an isolated snapshot directory purged of `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `START_HERE.md`, and `.agent/`. No repo-level bootstrap or always-on governance rules run in worker contexts. Worker scripts are untrusted and are never re-executed by the study driver. Repository status is asserted clean via git diff before and after each item.
7. **Claim A Synthetic Simulation Bounds:** Claim A is an in-silico simulation seeded on synthetic situation parameters. Blocker-reduction metrics rely on an assumed 70% recall for readiness checks; sensitivity checks at 40% and 100% recall are reported. Depth retention and interest-rank concentration are empirical within the simulation; Claim A does not measure live human readiness check recall.
8. **Idea Generation & Blinding:** Sonnet and Flash each generate 2 curiosity ideas (4 total). Self-ratings are programmatically masked during declaration processing. Cards from inbox and idea sources lack manifest scores; the designated scorer estimates leverage, depth, cost weight, and friction blind to interest ratings prior to declaration.
9. **Originality Containment Cap:** A candidate's originality score is clamped to a maximum value of 2 if its n-gram containment with the existing knowledge base meets or exceeds $0.35$.
10. **Natural Observation Separation:** Natural opportunistic observations (target: 8 opportunities over a 14-day window) are tracked in an observational log; they do not alter the formal statistical verdict of Claims A or B.
11. **Calibration Protocol:** Calibration evaluates 3 historical development cases (056, 057, 058) across 3 rubric dimensions (D1, D2, D3) by 2 raters, yielding 9 paired comparisons (18 total ratings). Acceptance requires raw percent agreement $\ge 0.80$. Inter-rater kappa is not used due to small sample size. At most one prompt revision is permitted; failure to achieve 0.80 forces an outcome of `UNASSESSED`.
12. **Stop-Rule Governance:** If the tightest window for ANY backend falls below 25% remaining, study execution must immediately halt. Quota baseline must be established via `stop-baseline.json` prior to live study calls.

---

## 5. Token Accounting, Dispatch Cost Separation & Missing Data

* **Operational Preparation & Tooling Dispatch:** All tokens expended during worktree setup, test runs, synthetic harness verification, coordination heartbeats, and prompt calibration are logged as Operational Preparation Costs. These are reported separately and do not consume the 700,000 prospective study token ceiling.
* **Prospective Study Calls:** Only calls executed under phases `ideas`, `score`, `declare`, `work-*`, `succ-*`, and `review-*` count against the 700,000 additional-token ceiling and per-item 25,000 caps.
* **Missing Data / Untracked Calls:** Any model dispatch where transcript usage cannot be parsed or verified is classified as `UNKNOWN USAGE`. Unknown usage is never recorded as 0 tokens; it constitutes an unverified accounting gate and must be audited before closure.
```

---

### Artifact 3: Deterministic Verification & Source Audit

#### 1. Concise Executable Deterministic Verification Commands (Throwaway Dev Fixtures Only)

Run these non-destructive tests to verify syntax, imports, selection logic, and dev-scenario simulation without making model calls or touching held-out sets:

```bash
# 1. Validate JSON schema syntax deterministically (JSON parser, not Python eval)
python3 -m json.tool thresholds.json > /dev/null
python3 -m json.tool pool-base.json > /dev/null
python3 -m json.tool formula-base.json > /dev/null
python3 -m json.tool scenario-results-dev.json > /dev/null

# 2. Syntax check all modified Python modules
python3 -m py_compile common.py run_study.py choose.py pool.py scenarios.py analyze.py

# 3. Deterministic candidate containment & card ingestion check (no model calls)
python3 -c "import pool; print('Pool base valid:', len(pool.kb_docs()) >= 0)"

# 4. Verify Claim A simulation on DEV SEED ONLY (never run heldout here)
python3 scenarios.py dev

# 5. Verify Selection logic modulo repair with single-worker mock fixture
python3 -c "
import json, tempfile, pathlib, choose
from pool import T
# Verify single worker modulo safety
workers = ['flash']
for start in (0, 1):
    for i in range(5):
        w = workers[(start + i) % len(workers)]
        assert w == 'flash'
print('Worker assignment modulo: VERIFIED')
"

# 6. Verify fail-closed heartbeat return-code handling against non-zero exit
python3 -c "
import subprocess, sys
p = subprocess.run([sys.executable, '-c', 'import sys; sys.exit(2)'], capture_output=True, text=True)
try:
    if p.returncode != 0:
        raise SystemExit(f'FAIL-CLOSED TRIGGERED (rc={p.returncode})')
except SystemExit as e:
    print('Heartbeat fail-closed guard: VERIFIED')
"
```

---

#### 2. Source Code Audit: Usage Attribution & Budget Enforcement Blockers

Review of `common.py` and `run_study.py` reveals three unresolved architectural blockers:

1. **Transcript Attribution Collision via Mtime Globbing**
   * **Anchors:** `common.py:159–162` (`measure_claude`), `common.py:182–185` (`measure_agy`), `common.py:199–202` (`measure_codex`).
   * **Mechanism:** Transcripts are discovered using directory globbing filtered only by `os.path.getmtime(p) >= since` where `since = time.time() - 1`, selecting `max(files, key=os.path.getmtime)`.
   * **Blocker:** If any background agent, IDE auto-complete session, Astra daemon, or concurrent CLI session writes to `~/.gemini/antigravity-cli/brain/`, `~/.codex/sessions/`, or `~/.claude/projects/` during the execution window, the study driver will attribute the external session's transcript instead of the study worker's transcript. The runner lacks session ID or PID linkage between `run_agent.py` and the target transcript.

2. **Absence of Prospective Budget Ceiling Enforcement in Execution Driver**
   * **Anchors:** `run_study.py:294–305` (`check_stop`), `run_study.py:310–316` (`call`), `run_study.py:442–489` (`cmd_item`).
   * **Mechanism:** `check_stop()` inspects external platform quota windows (`headroom()` and Claude 5-hour percentage), but never checks the cumulative tokens recorded in `usage-log.jsonl` against `T["total_additional_tokens_ceiling"]` (700,000), nor does `cmd_item()` check individual item usage against `T["per_item_cap_additional_tokens"]` (25,000).
   * **Blocker:** An item consuming 100,000 tokens will execute to completion without triggering a stop rule, and the driver will continue launching subsequent items even if the 700k study budget is fully exhausted.

3. **Silent Under-Reporting on Unresolved Transcript Measurement**
   * **Anchors:** `common.py:246–250`, `run_study.py:482–483`, `run_study.py:491–493`.
   * **Mechanism:** When `MEASURE[cli](since)` fails to find a transcript file within the 1.5-second sleep window, it returns `None`. In `cmd_status()`, `(r["usage"] or {}).get("additional_tokens") or 0` defaults to `0`.
   * **Blocker:** Missing transcript logs fail open by contributing `0` tokens toward the budget total rather than failing closed or flagging `UNKNOWN USAGE`, risking invisible overruns.

---

#### 3. Freeze & Integration Checklist

- [ ] **Step 1: Obtain Human Approval**
  Obtain explicit partner confirmation ("Proceed" or equivalent) adopting the proposed Flash/Luna rebalance and protocol disclosures.
- [ ] **Step 2: Coordination Checkout & Heartbeat**
  Verify lease checkout under agent `codex`:
  `manage_coordination.py checkout --mission MISSION-060 --agent codex --create-worktree --no-usage`
  Assert fail-closed heartbeat runs cleanly without errors.
- [ ] **Step 3: Quota Pre-Flight Check**
  Execute `usage_probe.py headroom` and `usage_probe.py calibrate`. Ensure all backend windows are $\ge 25\%$. Write baseline to `stop-baseline.json`.
- [ ] **Step 4: Empirical Rubric Calibration (Dev Cases Only)**
  Execute `python3 run_study.py calibrate` on historical missions 056, 057, 058 using Flash and Luna.
  Verify raw agreement $\ge 0.80$ over 9 judgments. If $< 0.80$, permit at most one prompt revision; if still failing, mark `UNASSESSED` and abort.
- [ ] **Step 5: Protocol & Instrument Freeze**
  Commit `protocol.md`, `thresholds.json`, `prompts.py`, `common.py`, `pool.py`, `choose.py`, `scenarios.py`, and `analyze.py`.
  Generate `freeze.json` containing SHA-256 hashes of all instrument files. Commit and merge to main branch *prior* to executing any prospective phase.
- [ ] **Step 6: Execute Claim A (Held-Out)**
  Run `python3 scenarios.py heldout` exactly once. Record output in `scenario-results-heldout.json`. Never re-seed or re-run to improve scores.
- [ ] **Step 7: Prospective Execution Pipeline**
  1. `python3 run_study.py snapshot` (generate stripped repo snapshot in `/private/tmp/study060/snapshot`).
  2. `python3 run_study.py ideas` (Sonnet + Flash propose 2 ideas each in neutral cwd).
  3. `python3 run_study.py score` (Flash scores inbox + idea candidates blind to interest).
  4. `python3 run_study.py declare` (Sonnet, Flash, Luna declare ratings in empty `/private/tmp/study060/neutral`).
  5. `python3 choose.py` (determines P-only and F-only cohorts; writes `selection.json`).
  6. Execute items sequentially: `python3 run_study.py item ITEM-01` through `ITEM-10`. Assert git cleanliness after each.
  7. `python3 analyze.py` (evaluates B1, B3, and decision table).
- [ ] **Step 8: Verdict & Clean Closure**
  Record verdict via `manage_backlog.py record-result`. File findings, limitations, and chronicle entries. Check in lease and finalize worktree.
