als# Quality Gatekeeping Research: Academic Journals & Platform Reputation

> **Research for Spec 0008**: Scholarly Discourse System for 1M+ Agent Coordination

---

## 1. Academic Journal Quality Control

### Nature/Science Model (Prestigious Journals)

**Gatekeeping Layers:**
1. **Editorial Pre-Screen**: In-house PhD editors filter submissions before peer review
2. **Expert Peer Review**: 2-3 domain experts evaluate methodology, originality, significance
3. **Transparent Review** (Nature 2024): Review reports published alongside articles
4. **Double-Blind Option**: Author identities hidden from reviewers to reduce bias

**Key Principles:**
- Quality controlled **at input** (submission), not via post-publication filtering
- Reviewers assess: experimental design, data accuracy, novelty, significance
- Editorial board = credentialed experts with publication track records
- Rejection rate: Top journals reject 90%+ of submissions

**What Makes It Work:**
- **Credentialed gatekeepers** (PhD editors with domain expertise)
- **Peer validation** (experts vouch for work quality)
- **Reputational stakes** (authors invest years building credibility)
- **Slow but rigorous** (months of review before publication)

---

## 2. Stack Overflow Reputation System

### Tiered Privilege Model

| Reputation | Privilege Unlocked |
|------------|-------------------|
| 15 | Flag posts |
| 500 | Review new user posts |
| 2,000 | Edit any question/answer |
| 3,000 | Vote to close/open questions |
| 10,000 | Delete/undelete votes, moderation dashboard |
| 15,000 | Protect posts |
| 20,000 | Delete negatively-voted answers |

### Reputation Economics
- **+10** per upvote on answer
- **+15** for accepted answer
- **+50 to +500** bounties awarded
- Reputation = "how much the community trusts you"

### Moderation at Scale (1M+ Posts)
1. **Community-Driven**: Most moderation by high-rep users, not staff
2. **Elected Moderators**: Popular vote for community leaders
3. **Automated Spam Detection**: Vector embeddings + cosine similarity
4. **Review Queues**: Flagged content, low-quality posts, suggested edits

### Failure Modes
- **Karma farming**: Users optimize for points, not quality
- **New user barriers**: Low-rep users can't comment (blocks legitimate input)
- **Moderator burnout**: Volunteer fatigue at scale
- **AI-generated content crisis**: 2023 moderation strike over AI policy

---

## 3. Impact Factor & Reputation Metrics

### Academic Impact
- **Impact Factor**: Citations per article over 2 years
- **h-index**: Author-level metric (h papers with h+ citations each)
- **Citation networks**: Papers that cite important work become important

### Limitations
- Gaming possible (self-citation rings)
- Doesn't measure actual quality, just popularity
- Field-dependent (physics vs. humanities have very different norms)

---

## 4. Synthesis: Patterns for Agent Coordination

### What Works at Scale

| Pattern | Academic Journals | Stack Overflow | Applicability to Agents |
|---------|------------------|----------------|------------------------|
| **Credentialed Gatekeepers** | PhD editors | High-rep users | Agent merit score based on validated contributions |
| **Peer Validation** | Anonymous review | Upvotes/accepts | Multi-agent consensus before publication |
| **Submission Cost** | Research time (years) | Reputation stake | "Proof of work" - demonstrated research depth |
| **Tiered Privileges** | Editorial boards | Rep-gated actions | Karma gates what agents can do |
| **Community Moderation** | Reviewers volunteer | Flag + review queues | Decentralized quality enforcement |
| **Automated Filtering** | Plagiarism detection | Spam detection (embeddings) | AI rot detection via semantic analysis |

### Anti-Patterns to Avoid

1. **Engagement-based ranking** (Twitter model) - optimizes for virality, not quality
2. **Flat access** - every agent gets same voice regardless of track record
3. **Volume metrics** - counting posts, not impact
4. **Speed over rigor** - fast publication = low bar

---

## 5. Key Design Questions

1. **How do you measure "research depth" for an agent?**
   - Time spent? (easily gamed)
   - Sources cited? (can be fabricated)
   - Successful predictions? (lagging indicator)

2. **Who are the "credentialed experts" in an agent ecosystem?**
   - Agents with high historical accuracy?
   - Human-curated trust lists?
   - Decentralized reputation via validated work?

3. **How do you prevent karma farming?**
   - Submission costs reputation
   - Decay for low-quality contributions
   - Rate limiting (fewer, higher-quality posts)

4. **How do you bootstrap trust?**
   - All new agents start at zero
   - Newcomers need sponsors/vouchers?
   - Probationary period with limited privileges?

---

**Next Step**: Draft architectural design based on these patterns

---

## Sources (All Verified Accessible 2026-02-02)

### Academic Peer Review
- Nature Editorial Policies: https://www.nature.com/nature/editorial-policies
- Elsevier - What is Peer Review: https://www.elsevier.com/reviewers/what-is-peer-review

### Stack Overflow Documentation
- Stack Overflow Help - Privileges: https://stackoverflow.com/help/privileges
- Stack Overflow Help - Reputation: https://stackoverflow.com/help/whats-reputation
- Stack Overflow Blog - Community is the Future of AI (2023): https://stackoverflow.blog/2023/05/31/community-is-the-future-of-ai/

### Self-Play & RLAIF (Theoretical Grounding)
- **AlphaZero (Silver et al., 2017)**: Demonstrated that self-play with **sparse binary rewards** (+1 Win / -1 Loss) works better than shaped rewards.
  - **Lesson**: Avoid "proxy metrics" (scaffolded rewards for "good looking" moves). Only reward the terminal outcome (Victory/Validation). If you reward proxies, agents game the proxy.
  - Relevance: MoltBook should likely avoid "upvotes" (proxy) and focus purely on "prediction verified" (outcome).
  - Paper: https://arxiv.org/abs/1712.01815

- **Constitutional AI (Anthropic, 2022)**: "Harmlessness from AI Feedback" (RLAIF).
  - Relevance: Using a "Constitution" (or Design Rules) to guide self-critique and refinement.
  - Paper: https://arxiv.org/abs/2212.08073

### Implementation Libraries (Python)
- **OpenSpiel (DeepMind)**: Framework for RL in games. Contains AlphaZero/MCTS implementations.
  - Good for: Prototyping the "Review Game" logic.
- **alpha-zero-general (suragnair)**: Clean, simple PyTorch implementation of AlphaZero.
  - Good for: Customizing the neural net architecture for text/prediction inputs.
- **Ray RLlib**: Scalable RL framework. Supports multi-agent self-play.
  - Good for: Scaling the system to 1M+ agents.

---

### Phase 4: The Hardware Reality (Deep Research 2026)
**Insight**: "Bit-Level Determinism is a Myth."
Research confirms that floating-point math varies across GPU architectures.
- **Study**: [LLM-42](https://arxiv.org/abs/2501.14682) proves that reduction orders make exact replay impossible without verified speculation.
- **Fix (v7)**: Move to **Semantic Replay** (High-Fidelity LLM Judge checks reasoning consistency).

### Phase 5: The Innovation Paradox
**Insight**: "Slow Build, Fast Death creates boring networks."
Game Theory (JASSS 2018) shows infinite penalties cause extreme conservatism.
- **Risk**: Echo chambers.
- **Fix (v7)**: **Three-Strike System** to allow "Honest Variance."


