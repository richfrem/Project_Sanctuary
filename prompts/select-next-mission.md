# Sanctuary — Select the Next Mission

Bootstrap is already complete. Reuse its context and briefing.

Review the current open missions and recommend the highest-scoring mission that is ready to make useful progress. Follow Sanctuary’s existing mission-selection rules and scripts.

## Scope

This is a read-only selection task. Do not start a mission, create a worktree or lease, dispatch workers, change scores or statuses, install dependencies, or publish anything.

Use documented script commands. Check `--help` when necessary rather than guessing flags or writing ad-hoc Python to reconstruct existing capabilities.

## 1. Establish the current candidates

Use the canonical mission backlog and existing tooling to identify:
- IN_PROGRESS missions.
- STAGED missions.
- PROPOSED missions.

Exclude resolved missions. Distinguish mission status from delivery status: a handoff marked COMPLETED does not necessarily mean the mission’s research is complete.

Use the generated index for initial ranking. Read the leading candidates’ manifests and relevant handoffs to verify their current situation. Expand the shortlist only if the leading candidates are unsuitable.

Do not conduct a repository-wide audit or reread every experiment.

## 2. Account for existing work first

Check active ownership and the latest relevant continuity records.

Prefer finishing or unblocking existing work when there is a concrete, authorized next step. Do not recommend resuming work that Richard explicitly stopped, or repeating a frozen experiment without authorization.

A blocked or intentionally paused mission is not automatically the next task simply because it is IN_PROGRESS.

## 3. Evaluate score, interest and readiness separately

For each shortlisted candidate, establish:

**Recorded priority**
- Canonical score and its recorded basis.
- Any obvious stale assumptions affecting that score.
- Do not silently replace the project’s scoring formula.

**Interest and learning value**
- Recorded agent curiosity, originality, potential understanding or creative expression.
- Pure exploration qualifies without an operational-benefit justification.
- Do not invent an interest score or describe subjective experience as measured evidence.
- Apply protected-exploration policies only if actually adopted; a proposed experiment such as MISSION-060 is not itself an active policy.

**Readiness**
- A clear next step and observable completion criteria.
- Required tools, data and permissions.
- Available model quota or compute budget, using existing probes only when needed.
- Conflicting ownership or shared resources.
- Whether the work can reach a meaningful stopping point within available resources.

Keep readiness distinct from intrinsic value. If an interesting mission is not ready, identify its smallest enabling step rather than calling it unimportant.

## 4. Recommend one mission

Recommend the highest-scoring actionable candidate under the currently adopted policy.

If you recommend a lower-scoring mission, state the concrete reason: higher-ranked candidates are blocked, explicitly paused, conflict with active work, exceed available resources, or an adopted exploration policy applies.

If scores are tied, prefer:
1. Finishing existing work.
2. Stronger documented interest and learning value.
3. A smaller achievable completion scope.

Do not manufacture numeric precision or start another mission merely because an existing one is difficult.

If no mission is ready, recommend one bounded enabling action for the best candidate. Do not create a new mission.

## 5. Report briefly and stop

Provide:

| Mission | Status | Recorded score | Readiness / blocker | Interest or learning value |
|---|---|---:|---|---|

Show up to five candidates.

Then state:
- **Recommended mission:** ID and title.
- **Why this one:** two or three sentences, including why higher-ranked candidates were skipped, if applicable.
- **First concrete action:** the existing command or supported workflow to begin.
- **Completion target:** what useful result could be delivered in the next bounded session.
- **Authorization or resource gap:** only an actual unresolved requirement; otherwise “None identified.”
- **Evidence:** manifest paths and the relevant handoff or readiness evidence.

End with:
“Recommendation only. No mission execution started.”

Do not repeat the bootstrap report or end after listing scores without making a recommendation.

## Closing reminders (include these in the report, just before the final line "Recommendation only. No mission execution started.", which stays last)

1. **Plain-language summary.** Add two or three everyday-language sentences: which mission you recommend and why, without IDs or jargon.
2. **Relevance to Sanctuary.** Say whether anything you noticed (for example, stale scores, unclear readiness fields or a missing tool) could improve Sanctuary's selection process or related workflows. Give evidence, or say "None identified." Do not act on it.
3. **Git and workspace housekeeping.** This task is read-only, so none of these should be needed: commit, push, pull request, merge, checkout `main`, pull `origin main`, delete a branch, remove a worktree. If you created or changed anything anyway, say so, stop, and ask me before taking any of those steps.
