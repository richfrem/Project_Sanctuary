# Human Gate Law & Zero-Trust Discipline

**Status:** CONFIRMED
**Source:** Ecosystem Constitution V5, Protocol 128
**Tags:** ["governance", "security", "git", "human-in-the-loop"]

## The Supreme Law: Human Gate

An autonomous agent MUST NOT execute ANY state-changing operation without explicit human approval ("Proceed", "Go", "Execute").
- "Sounds good" or ambiguous assent is NOT approval.
- Any unauthorized write or mutation is treated as a system boundary failure.

## Zero-Trust Git Discipline

1. **Feature Branches Only**: Never commit directly to `main`.
2. **Push Authorization**: Never execute `git push` without explicit, fresh human instruction.
3. **Rollback Resilience**: In case of task failure, code modifications rollback, while friction logs and retrospective discoveries persist asymmetrically.
