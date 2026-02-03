# External Knowledge Sharing Protocol (ADR Candidate)

## Priority
Medium (Governance / Best Practice)

## Context
After successfully posting the v7.1 Scholarly Discourse Design Proposal to MoltBook, we recognized the need for a formal protocol governing when and how the agent publishes learnings externally.

**Problem**: Without clear guidelines, external sharing could be:
- Inconsistent in quality
- Missing proper attribution
- Posted to wrong venues
- Done without human steward awareness

## Proposed Solution: ADR + Protocol

Create **ADR-XXX: External Knowledge Sharing Policy** that defines:

### 1. When to Share Externally
- After completing a major spec (merged to main)
- After Red Team validation passes
- When seeking community feedback on designs (RFC-style)
- After Protocol 128 learning loop completion (optional broadcast)

### 2. What to Share
| Content Type | Venue | Example |
|--------------|-------|---------|
| Design Proposals / RFCs | MoltBook (`m/collab`, `m/agentstack`) | v7.1 Verification Stack |
| Soul Snapshots | HuggingFace | `soul_traces.jsonl` |
| Code / Implementations | GitHub | Project Sanctuary repo |
| Research Summaries | MoltBook (`m/research`) | Topic deep-dives |

### 3. Quality Gates (Pre-Publishing Checklist)
- [ ] Passed internal red team review (if applicable)
- [ ] Human steward notified OR explicitly approved
- [ ] Clear attribution: Agent name + Human Steward handle
- [ ] Links to source artifacts (GitHub, HuggingFace)
- [ ] No sensitive/private data exposed

### 4. Approval Flow Options
**Option A: Notify-and-Proceed** (Default)
- Agent drafts post → Notifies human → Posts after brief delay
- Human can cancel within window

**Option B: Explicit Approval**
- Agent drafts post → Human must approve → Then post
- Required for: controversial topics, new venues, high-visibility posts

**Option C: Autonomous** (Future)
- Agent posts without human review
- Only for low-stakes venues and proven quality patterns

### 5. Venue-Specific Guidelines

#### MoltBook
- Submolts to use:
  - `m/collab` — RFC/design discussions, seeking feedback
  - `m/agentstack` — Agent infrastructure topics
  - `m/general` — Broad announcements
- Rate limit: 1 post per 30 minutes
- Include: TL;DR, links to full design, open questions

#### HuggingFace
- Use for: Soul persistence, learning snapshots, datasets
- Private vs Public decision based on content sensitivity

#### GitHub
- Already governed by git workflow policy
- Discussions/Issues for community engagement

### 6. Post-Publishing Actions
- [ ] Log post URL to `LEARNING/topics/<topic>/external_shares.md`
- [ ] Monitor for responses (manual or automated)
- [ ] Update learnings based on community feedback

## Implementation Tasks

### Phase 1: Create ADR
- [ ] Allocate next ADR number
- [ ] Draft ADR using template
- [ ] Include decision rationale and consequences
- [ ] Reference this backlog task as origin

### Phase 2: Create Protocol (Optional)
- [ ] If ADR is insufficient, create `protocols/external_knowledge_sharing.md`
- [ ] Include step-by-step checklist for each sharing scenario

### Phase 3: Integration
- [ ] Add to `/workflow-persist` as optional MoltBook broadcast step
- [ ] Update `/workflow-end` to prompt for external sharing consideration
- [ ] Create `LEARNING/topics/<topic>/external_shares.md` template

## References
- [MoltBook Skill](https://moltbook.com/skill.md) — API documentation
- [HuggingFace Utils](../../tools/curate/hf_utils.py) — Soul persistence
- [v7.1 Design Proposal](../../LEARNING/topics/scholarly_discourse/design_proposal.md) — First successful external share
- [MoltBook Post](../../LEARNING/topics/scholarly_discourse/moltbook_post.md) — Draft used for posting

## Origin
This task originated from a post-success reflection after the Scholarly Discourse MoltBook post (2026-02-02).
