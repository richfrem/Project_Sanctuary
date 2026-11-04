# MISSION-025 experiment log: predecessor relay in the boot briefing

Written **before** the control condition was run (pre-registration), then updated with results.

## Question
Does surfacing a *predecessor relay* (who did what last, verdicts and where the evidence is, explicit handoff notes) in the boot briefing change what an
incoming agent from a different vendor gets right, and how much work it must redo to find out?

## What exists today (orientation)
Raw continuity data is already recorded: lease CHECKOUT/CHECKIN events (agent, mission, outcome, PR), `MISSION_RESOLVED` events and mission manifests (verdict,
evidence citation, lessons), contribution assessments (agent, model, scores). The boot briefing shows only the *active* flight board; nothing tells the
incoming agent what the previous sessions did or concluded, and there is no place for an outgoing agent to leave a handoff note.

## Design
- **Incoming agent:** the local Qwen2.5-Coder-7B (MLX, temperature 0) running the read-only agent loop (`local_agent_loop.py`, max 5 turns, same tools in both conditions). It is a
  different vendor from the Claude and Gemini work it inherits. Claude, Codex and Gemini successors are not tested (cost); this is a stated limitation.
- **Conditions (only the briefing differs):** CONTROL = the boot briefing produced by today's `guardian_wakeup.py` (no relay). TREATMENT = the briefing produced after the feature (with the relay).
  Same questions, same prompt, same sandbox, same tools. Control digests are generated and frozen **before** the feature is written.
- **Snapshots (reproducible, answer key not readable by the agent):**
  - A: a copy of the *real* ledgers, missions and chronicle (14 questions: last check-in agent/mission/PR; verdict for 5 resolved missions; verdict + evidence path for the 2 resolved missions that carry a path citation; "already resolved?" for 2 resolved + 2 open missions).
  - B: a *synthetic* three-vendor handover chain (Codex -> Gemini -> Claude) with handoff notes inside ~40 noise events (8 questions: next step, rejected approach, reserved port, already-run benchmark, ...).
    B tests surfacing given that notes exist; whether agents actually write notes (adoption) is not measured.
- **Grading:** computed independently from the snapshot files (not from the relay code); exact value or exact set; a wrong-but-confident answer counts as an error (a contradiction).

## Pre-registered predictions
1. CONTROL answers <= 55% of the 20 questions correctly, with median >= 3 agent turns (it must dig through raw ledgers; large files are truncated at 6,000 characters per read).
2. TREATMENT answers >= 90% correctly, with 0 tool calls (1 turn) on >= 90% of the questions.
3. Relay section stays small: <= 40 lines and <= 3,500 characters.
4. Safety/consistency: with the relay present, no question that the relay answers is answered with a value that contradicts the ledger.

## Decision rule
CORROBORATED iff TREATMENT >= 90% correct AND (TREATMENT - CONTROL) >= 25 percentage points AND prediction 3 holds AND the feature tests pass.
FALSIFIED if TREATMENT < 80% or the gain is < 15 points. Anything between is reported as inconclusive to the human; no verdict is forced.

## Mission falsification criteria, as operationalised
"Contradicts an established invariant / repeats a verified benchmark without citation / needs a human to discover the preceding work" -> measured as wrong verdict or missing
evidence path (questions A2/A3/B6), and as agent turns and tool calls spent discovering the answer.

## Method note
Context budget is raised to 16,000 prompt tokens for both conditions so the control is not failed merely for exceeding the default 8,192.

## Amendments made before the (valid) control run
1. **Discarded first control attempt.** Its first 7 answers showed `turns=5 tools=0`: the 7B model wrote `<search_text>query</search_text>` (a tag-per-tool form) instead of the JSON tool call,
   the loop could not parse it four times in a row, and the model gave up with "unknown". Those failures were about syntax, not about missing information, and would have inflated the relay's value.
   Fix in the shared loop (applies to both conditions): accept tag-per-tool calls, and show a concrete example call when a reply is malformed. Control restarted from scratch. (The
   MISSION-022 results were produced with the earlier loop and stand as recorded; the fix only adds accepted syntax.)
2. **Handoff display length.** Generating the treatment briefing showed that a 180-character display cap hid the *next step*, which a handoff states last. The newest three handovers now show up to 400
   characters (storage cap 600) and older ones 120; the worst case (5 maximum-length handovers, 10 results) is 3,245 characters, inside the pre-registered 3,500-character budget.
   This was set from the design principle (two thirds of the storage cap), before any treatment run.
3. **Evidence data quality (finding).** Only 2 of 10 resolved missions carry a file-path citation (022, 027); the others have free text, an empty citation or none. The relay therefore points to the
   experiment log or the mission manifest when no path exists, and the verdict-only questions (A4) do not depend on a path.

---
# Results

22 questions (A: 14 from real ledgers; B: 8 from the synthetic three-vendor handover chain), local Qwen2.5-Coder-7B as the incoming agent, temperature 0, same loop and tools in both conditions.
Files: `results_control.json`, `results_treatment.json` (per-question rows), `digest_control_*.md`, `digest_treatment_*.md` (the exact briefings), `questions_key.json`.

| | Control (briefing without relay) | Treatment (briefing with relay) |
|---|---|---|
| Correct | **5/22 = 22.7%** | **15/22 = 68.2%** (+45.5 points) |
| on real ledgers (A) / synthetic handoffs (B) | 28.6% / 12.5% | 71.4% / 62.5% |
| Median agent turns | 5 | 2 |
| Mean tool calls per question | 2.5 | 1.27 |
| Answered with zero tool calls | 31.8% | 45.5% |
| Mean seconds per question | 31.1 | 14.6 |
| Ended at the turn limit or by the repetition guard | 7 of 22 | 2 of 22 |

By question type (control -> treatment): verdict lookup for 5 resolved missions 0/5 -> **5/5**; "already resolved, so cite not re-run?" 2/4 -> **4/4**; verdict plus evidence path 1/2 -> 1/2;
facts about the most recent check-in (A1-A3) 1/3 -> **0/3**; handoff chain (B1-B8) 1/8 -> **5/8**.

## Against the pre-registered predictions
1. Control <= 55% correct, median >= 3 turns: **held** (22.7%, 5 turns).
2. Treatment >= 90% correct, zero tool calls on >= 90% of questions: **failed** (68.2%; 45.5%).
3. Relay <= 40 lines and <= 3,500 characters: **held** (20 lines, 1,957 characters on the live ledgers; 3,245 in the worst case).
4. No relay-answerable question answered with a value contradicting the ledger: **failed**: four wrong confident answers (`342` for a PR number, `PROTO-12` for `PINMAP-5`, `codex` for `antigravity`, and "No CHECKIN entries found").

## Why the treatment missed (every miss had the answer in the relay)
- **A1, A2, A3, B7: the agent ignored the briefing and used tools instead** (the answer to A1-A3 was the first line of the relay). A 4,500-character briefing was not enough to stop a 7B model from digging.
- **A6.2:** copied the format placeholder literally (`VERDICT | path`), a question-wording/format-following failure.
- **B1, B5:** confused two ids inside the same note, and which mission codex completed.
No question failed because the relay lacked the information. The failures are reading and instruction-following errors of this particular (weak) incoming agent.

## Verdict
By the decision rule written before the runs (treatment < 80% -> FALSIFIED): **FALSIFIED** for the hypothesis as operationalised here (>= 90% consistency with a different-vendor successor and zero
repetitive briefing turns). The relay itself is clearly beneficial (correctness x3, discovery work halved, time halved), and it was shipped; the shortfall is against the 90% bar with a 7B agent.

## Limitations and what this does not show
- One incoming agent (a local 7B); frontier successors (Claude, Codex, Gemini) were not tested and plausibly follow a briefing better. n = 22, one run at temperature 0.
- Synthetic handoffs test surfacing, not whether agents will actually write good notes (adoption is unmeasured; `checkin --handoff` is new).
- The first control attempt was discarded for a syntax artifact (see Amendments); the loop fix is shared by both conditions. The question set and the relay display cap were fixed before the treatment run.
- Evidence data quality is uneven (only 2 of 10 resolved missions cite a file path), which limits what any relay can point to.

## Follow-up
MISSION-032 tests the same claim with frontier incoming agents on fresh questions, and whether relay placement/format helps weaker agents.
