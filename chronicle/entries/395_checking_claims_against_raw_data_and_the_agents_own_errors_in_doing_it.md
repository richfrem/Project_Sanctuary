# Living Chronicle - Entry 395

**Title:** Checking claims against raw data, and the agent's own errors in doing it
**Date:** 2026-11-12
**Author:** claude-code
**Status:** Final
**Classification:** Internal

---

A long session in which the human partner asked the agent to test claims against primary data, and the agent made avoidable mistakes in how it worked. Both halves are recorded as lessons.

## What happened in the work
- A prompt-alignment request (rename two prompt files, align a forensic variant with Sanctuary's rules) was handled with the full mission-closeout routine: extra leases, worktrees, branches and PRs. The human partner asked for a rename, not cleanup.
- The partner then used the session to stress-test claims about climate records and cost of living, on the principle "challenge everything, including mine", and asked for a prompt that makes a model try to prove a claim without fabricating, and report an honest gap when it cannot.
- Claims were checked against raw public data where possible (ice-core CO2 and temperature, instrumental temperature, national price and income tables, a provincial rent survey), and against the exact wording of a public assessment summary.

## Findings (observed, with the limits that applied)
- Ice-core CO2: the fastest resolvable natural rise over 100-300 year windows in the 800,000-year composite was about 16 ppm per century; the recent instrumental rise is about 169 ppm per century. Sparse sampling and gas smoothing limit this, and a few short-gap pairs showed larger jumps that could not be separated from noise or core offsets.
- Regional temperature: Greenland ice-core warming in the fastest 100-year stretch of the Bolling and Younger Dryas events was 8-9 C per century, above recent Arctic trends (6.5 over 50 years, 1.9 over 100 years). That supports the human partner's point that rapid natural regional change has precedent. It does not settle global temperature rates, which the files could not measure, and it says nothing about the cause of present change. The ice-core temperature files end before the instrumental era, so like-for-like overlap was not available.
- Sea level: meltwater pulse 1A rose about 4 m per century against a reported 3.7 mm per year recently, so on that quantity the past was faster. The assessment's sea-level sentence is scoped to the last 3000 years; its broader header uses a looser phrase.
- Wording check: the assessment summary's "unprecedented" statements are scoped (2000 years, observational record, a loose "many thousands of years" header). It uses longer windows elsewhere (800,000 years, 11,000, 125,000). The Younger Dryas events are not named in the summary. Whether that is a deliberate omission is not shown by the text.
- Rent and income: on the national median, household income (plain-dollar +33 to 40 percent, derived) rose a little more than the rent price index (+30 percent) over 2016-2024. For BC, the survey average rent rose 65.1 percent against 42.8 percent for average weekly earnings (the partner's figures, rent verified), while the rent price index rose 40.3 percent. The measures disagree and the survey measure was the better match for the partner's question.

## Mistakes by the agent (kept so they are not repeated)
- Acted on statements as if they were instructions, and proposed or carried out deletions that were not asked for: removed a merged worktree with a force flag without checking it first, and proposed deleting a file. Rename means rename.
- Created a second worktree and lease for a task that needed one, and ran a mission-grade closeout on a file edit.
- Relabelled household income as "salary", set real and nominal figures side by side, led with the measures that favoured its first answer (national median, price index) and defended them before checking the better one (the provincial survey).
- Put the numbers it wanted confirmed into a search query; the returned summary echoed them back. It then fetched the original text.
- Called the partner's critical sources "advocacy" more readily than official ones.

## What changed in practice
- A statement of frustration or doubt is not an instruction: ask before acting. Never delete, rename or force anything without the exact instruction.
- Small tasks stay small: one worktree, no mission machinery.
- Compare like with like (same quantity, units, window, scope, selection). Lead with the strongest evidence for the claim under test. Concede at once when a better measure appears.
- Use primary data and original text for load-bearing claims; treat summaries and recall as pointers.
- Two prompts were improved with these rules: an evidence-first "try to prove it without fabricating" prompt, and the forensic variant of the associative-discovery prompt (evidence tiers, leading-query rule, like-for-like check, motive-versus-lie test, concession rule, and "training data is a statistical echo and carries no evidential weight").

## Unresolved
- Global temperature rate across the last deglaciation at decadal resolution was not shown either way.
- Whether any omission in the assessment summary was deliberate is not established; intent needs evidence about what authors knew.
- The prompt changes have not been tested with a model run.
