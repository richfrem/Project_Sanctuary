# Protocol 133: Companion Repository & Upstream Skill Improvement Protocol (v1.3)

**Status:** PROPOSED  
**Classification:** Operational Framework  
**Version:** 1.3 (two-repository duty added 2026-11-06; first used 2026-11-03; corrected 2026-11-04 after the catalog refresh; see Section 7)  
**Authority:** Project owner directive (2026-11-03); drafted by Claude Code. Ratify by changing Status to CANONICAL.  
**Linked Protocols:** Protocol 128 (Hardened Learning Loop)

---

## 1. Purpose

Project Sanctuary does not own the generic agent skills it depends on. They live in a **companion repository**, `richfrem/agent-plugins-skills` (MIT), and are installed globally by APM. This protocol says where things live and how to improve an upstream skill safely when Sanctuary work shows it is wrong or missing something, so that fixes land at the source instead of in hand-edited local copies.

## 2. How the pieces fit (verified 2026-11-03)

| Thing | Location |
|---|---|
| Sanctuary's own skills (`sanctuary-core`) | `plugins/sanctuary-core/` in this repository, installed per project (`apm.yml`) |
| Companion repository source | A local clone, by convention a sibling of this checkout (`../agent-plugins-skills`; on this workstation `~/Projects/agent-plugins-skills`). Skills live in `plugins/<plugin>/.apm/skills/<skill>/` |
| Global APM manifest | `~/.apm/apm.yml`. Its dependencies are absolute paths **into the clone** (one per plugin) |
| Installed copies (generated, never edited) | `~/.claude/skills/` (Claude), `~/.agents/skills/` (Copilot, Codex), `~/.gemini/antigravity-cli/skills/` (Antigravity) |

Consequences:
- The installed version of an upstream skill depends on **which dependency entry wins in `~/.apm/apm.yml`**. The global manifest lists both local clone paths and the GitHub dependency `richfrem/agent-plugins-skills`; the GitHub entry is pinned to a commit in `~/.apm/apm.lock.yaml`. Observed 2026-11-04: `apm install --global` refreshed `~/.apm/apm_modules/_local` from the clone but left `~/.claude/skills/cli-agent` at the pinned GitHub commit; only `apm update --global` (after the PR was merged to GitHub `main`) moved the installed skills. So an upstream change reaches the installed skills only **after it is merged and `apm update --global` has run**.
- Installed copies are outputs. Never edit them by hand (upstream Iron Law); the next install overwrites them.
- Sanctuary does not pin the upstream version in `apm.lock.yaml`; the clone's commit is the effective pin.

## 3. Where a change belongs

- **Generic skill behaviour or documentation** (for example a script whose docs describe a flag it does not have): fix upstream, using this protocol.
- **Sanctuary-specific behaviour**: change `plugins/sanctuary-core/` in this repository, not upstream.
- If unsure, ask the human partner before touching either.

## 4. Procedure for improving an upstream skill

Human Gate applies to every state-changing step; "Proceed" is required before the push, the merge and the global install.

1. **Preconditions.** The clone is clean and on an up-to-date `main` (`git -C <clone> status`, `git pull origin main`). Read the clone's `AGENTS.md`: adversarial pass before accepting any significant change; never delete a skill directory, `SKILL.md` or evals unless the human names the exact path; never run or fake the cryptographic human-gate signatures; stop after three failed repair attempts and log the friction in its `references/map-debt.md`.
2. **Feature branch in the clone.** `git checkout -b feat/<topic>`.
3. **Edit the source** under `plugins/<plugin>/.apm/skills/<skill>/`, never the installed copy.
4. **Verify before committing**, because step 6 skips the push hook. At minimum run the upstream audit on the changed skill:
   `python3 plugins/agent-extension-authoring/.apm/skills/authoring-skills/scripts/audit_skill.py <skill-path> --mode source`
   plus the skill's own tests or scripts that exercise the change.
5. **Commit** with explicitly staged paths and a `fix(<skill>)` / `feat(<skill>)` message. The clone's `pre-commit-evolution-guard` **blocks any commit touching `plugins/`** unless `references/map-debt.md` (or a plugin `references/evolution-log.md`) is staged too: log the friction there, as `DEBT-<date>-<slug> (RESOLVED)` with the section fields used by existing entries. The escape `Evolution-Check: none` commit trailer is enforced in CI, so use it only when there really is no friction to log. The `pre-commit-pipeline-guard` only warns when no control-plane task is registered; that does not block.
6. **Push without the hook, as directed by the owner:** `git push --no-verify -u origin feat/<topic>` (only after the commit has actually succeeded: a push with no new commit just creates an empty remote branch at `main`). `--no-verify` skips the **pre-push** hook only; the pre-commit hooks in step 5 still run. Understand what it skips: the clone's `pre-push` runs `pre-push-review-guard`, which blocks pushes from an active task branch until the control-plane task is `DONE`. Using `--no-verify` is an owner decision for this workflow; it is never a way around a failed audit, and it never authorises skipping a human signature gate.
7. **PR and merge:** `gh pr create --repo richfrem/agent-plugins-skills --base main`, then squash-merge on the human's go-ahead.
8. **Return the clone to main:** `git checkout main && git pull origin main`. Do this **before** installing so the installed state equals merged `main`, not the feature branch.
9. **Reinstall globally.** First `apm update --global --dry-run` (it lists every package and file that will move, because the GitHub pin may be many commits behind `main`), then `apm update --global --yes`; this is what actually changes `~/.claude/skills`. The single-plugin local install below only refreshes the `_local` copies. Previous single-plugin guidance, kept for a clone-only setup: Prefer the single-plugin form, which limits the blast radius to the plugin you changed: `apm install "$PWD/plugins/<plugin>" --global --target claude,copilot,codex,antigravity` (absolute path; on APM below 0.35.0 the target is `antigravity`, `agy` needs 0.35.0+). `apm install --global` with no arguments re-reads every installed plugin and is for a deliberate full refresh. Do a `--dry-run` first, and copy `~/.apm/apm.yml` and `apm.lock.yaml` somewhere safe before the real run (undo: restore them, or the clone's `apm_consumer.py list-backups` / `rollback`). Expect about 40 seconds and possible "lossy agent compilation" warnings (Codex drops an agent's `tools` field); read them. Verify: `apm deps list --global`; `diff -rq` each installed copy (`~/.claude/skills/<skill>`, `~/.agents/skills/<skill>`, `~/.gemini/antigravity-cli/skills/<skill>`) against `plugins/<plugin>/.apm/skills/<skill>` shows no differences; and run the changed behaviour from the installed path.
10. **Clean up:** `git branch -D feat/<topic>` and `git push origin --delete feat/<topic>`.
11. **Record in Sanctuary** (chronicle entry or note) what changed upstream and why, and update any Sanctuary document that described the old behaviour.

Antigravity (`agy`) global installs do not deploy hooks; hook-dependent behaviour needs a project install.

## 5. Known hazards (observed 2026-11-03)

- The global manifest lists both local clone paths and `richfrem/agent-plugins-skills` as a GitHub source. The clone's `INSTALL.md` warns against installing the same plugins from both a clone and GitHub in one scope; check `apm deps list --global` for duplicate skills.
- The installed APM was 0.33.0; the clone's `INSTALL.md` recommends 0.35.0 or newer for the `agy` target.
- Global installs write outside any repository, so a mistaken install affects every project on the workstation.

## 6. Relationship to Protocol 128

An upstream skill change is Stage 2 work (execution on a branch, human-gated) and its record belongs in Stage 3 (retrospective) and Stage 4 (chronicle, score). Do not start the work during Stage 1 unless a boot-time check shows an installed skill is broken.

## 7. First use (2026-11-03)

- **Change:** `managing-huggingface-assets/scripts/hf_upload.py` documented `--file`/`--remote-path` but had no command line; run as documented it exited 0 silently and uploaded nothing. Added the CLI with 8 tests, a docs update and a `DEBT-20261103-HF-UPLOAD-NO-CLI (RESOLVED)` map-debt entry. Upstream PR #746 (squash-merged), then the single-plugin global install of `dev-utils` with zero differences across the three install directories.
- **What the run taught (now folded into Section 4):** the evolution guard requires a map-debt entry for any `plugins/` commit; `--no-verify` skips only the pre-push hook; push after the commit succeeds; use the single-plugin install form with `antigravity` on APM 0.33.0; back up `~/.apm/` first; verify all three install directories.

## 8. Two-repository maintenance duty (added 2026-11-06, MISSION-055)

Sanctuary agents are responsible for keeping skills correct in BOTH repositories, not only for editing upstream on request.

1. **Place each change by ownership** (Section 3): generic behavior (orchestration, `cli-agent` launch controls, shared evaluation `evaluating-agent-outcomes`) lives in the companion; Sanctuary policy and adapters (`evaluating-agent-work`, `sanctuary_adapter.py`, `eval_gate.py` shim, guardian references) live here.
2. **One worktree and feature branch per repository**, named for the mission; record path, branch and base SHA in the mission's `LEARNING/experiments/mission-NNN/` notes. The flight board does not protect the companion.
3. **Order**: integrate and merge the companion first, run `apm update --global` and verify installed files equal the merged clone, then integrate the Sanctuary consumer. A consumer must never point at a capability that is not yet installed.
4. **Moving or renaming a capability** (for example `dispatch_guard.py` moved from `orchestrating-agent-work` to `cli-agent`): leave a compatibility shim or update every caller in the same change, then grep both repositories for the old name and path (`README.md`, `architecture.md`, `AGENTS.md`, `START_HERE.md`, `INSTALL.md`, `bootstrapping.md`, skill `SKILL.md` and `references/`).
5. **Keep awareness current**: update the root documents of both repositories and every skill that routes to the changed capability in the same change set. The companion's `AGENTS.md` names Sanctuary as a consumer.
6. **Verify installs**: after merge, global `apm update --global` and Sanctuary `apm install`; compare installed copies with source for Claude, Codex and AGY; check no project-local copy shadows a global skill; restore unrelated `apm.lock.yaml` noise.
7. **Never hand-edit installed copies**; clean up the worktrees and branches in both repositories.

## Revision note (2026-11-04, v1.2)
The first catalog refresh (companion repo PR 747) showed that `apm install --global` alone did not update the installed skills because of the pinned GitHub dependency; steps 9 and the install note above were corrected, and the verification there (`diff -rq` of installed copies against the plugin) is what caught it. The update moved 7 packages from 3f0c0ba to 6bb64dc and reported one interrupted-install backup item it could not remove.
