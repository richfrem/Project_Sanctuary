# Protocol 133: Companion Repository & Upstream Skill Improvement Protocol (v1.0)

**Status:** PROPOSED  
**Classification:** Operational Framework  
**Version:** 1.0  
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
- The installed version of an upstream skill is **whatever the clone's working tree holds when `apm install --global` runs**. A clone sitting on a feature branch or with uncommitted edits installs that state.
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
5. **Commit** with explicitly staged paths and a `fix(<skill>)` / `feat(<skill>)` message.
6. **Push without the hook, as directed by the owner:** `git push --no-verify -u origin feat/<topic>`. Understand what this skips: the clone's `pre-push` runs `pre-push-review-guard`, which blocks pushes from an active task branch until the control-plane task is `DONE`. Using `--no-verify` is an owner decision for this workflow; it is never a way around a failed audit, and it never authorises skipping a human signature gate.
7. **PR and merge:** `gh pr create --repo richfrem/agent-plugins-skills --base main`, then squash-merge on the human's go-ahead.
8. **Return the clone to main:** `git checkout main && git pull origin main`. Do this **before** installing so the installed state equals merged `main`, not the feature branch.
9. **Reinstall globally** from the clone: `apm install --global` (re-reads every installed plugin from the clone), or for a single plugin `apm install "$PWD/plugins/<plugin>" --global --target claude,copilot,codex,agy` with an absolute path. Confirm with `apm deps list --global`, and check the result, for example `diff -rq ~/.claude/skills/<skill> plugins/<plugin>/.apm/skills/<skill>` shows no differences. To undo, use the clone's `apm_consumer.py list-backups` and `rollback`.
10. **Clean up:** `git branch -D feat/<topic>` and `git push origin --delete feat/<topic>`.
11. **Record in Sanctuary** (chronicle entry or note) what changed upstream and why, and update any Sanctuary document that described the old behaviour.

Antigravity (`agy`) global installs do not deploy hooks; hook-dependent behaviour needs a project install.

## 5. Known hazards (observed 2026-11-03)

- The global manifest lists both local clone paths and `richfrem/agent-plugins-skills` as a GitHub source. The clone's `INSTALL.md` warns against installing the same plugins from both a clone and GitHub in one scope; check `apm deps list --global` for duplicate skills.
- The installed APM was 0.33.0; the clone's `INSTALL.md` recommends 0.35.0 or newer for the `agy` target.
- Global installs write outside any repository, so a mistaken install affects every project on the workstation.

## 6. Relationship to Protocol 128

An upstream skill change is Stage 2 work (execution on a branch, human-gated) and its record belongs in Stage 3 (retrospective) and Stage 4 (chronicle, score). Do not start the work during Stage 1 unless a boot-time check shows an installed skill is broken.
