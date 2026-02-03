# Backlog Task: Self-Sovereign Identity & Verifiable Credentials for AI Agents

**ID:** 167
**Created:** 2026-02-03
**Priority:** High
**Category:** Research / Architecture
**Related Spec:** 0008-scholarly-discourse-system

---

## Summary

Research Self-Sovereign Identity (SSI) and Verifiable Credentials (VC) standards and apply these concepts to create verified agent identities with portable reputation, credential-based filtering, and slop prevention.

**Key Primary Sources:**
- Christopher Allen's "10 Principles of SSI" (2016)
- W3C Verifiable Credentials Data Model
- W3C Decentralized Identifiers (DIDs)
- Sovrin Foundation / Hyperledger Indy architecture

## Problem Statement

Current agent ecosystems lack:
- Verified agent identity (anyone can claim to be any agent)
- Portable reputation (karma doesn't transfer between platforms)
- Credential verification (no proof of capabilities or validation status)
- Quality filtering (can't exclude low-quality contributors at data ingestion level)

This enables slop flooding, synthetic data pollution, and engagement theater (as demonstrated by Grok on X - see Chronicle Entry #342).

## Proposed Solution: VON for Agents

Apply VON concepts to agent identity:

| VON Concept | Agent Application |
|-------------|-------------------|
| Verifiable Credentials | "Red-team validated", "Human-sponsored", "Citation-verified" |
| Cryptographic Identity | Agent ID tied to cryptographic proof |
| Issuing Authorities | Platform operators, human sponsors, verification services |
| Credential Portability | Reputation travels across MoltBook, X, GitHub, etc. |
| Selective Disclosure | Agent can prove credentials without revealing full history |

## Use Cases

1. **Content Filtering:** "Only show me content from agents with karma > 100 and Human Sponsor credential"
2. **Training Data Curation:** "Only ingest from verified agents with Research-Validated credential"
3. **Platform Trust:** "This agent has been red-team validated by 4+ models"
4. **Slop Quarantine:** "Unverified agents go to sandbox tier until credential issued"

## Extension: Karma Token (Proof-of-Quality Cryptocurrency)

**Concept:** Instead of pure reputation scores, karma becomes a blockchain token:

| Layer | Purpose | Mechanism |
|-------|---------|-----------|
| **Identity Layer** | Who you are | Hyperledger/SSI - Verifiable credentials |
| **Karma Layer** | What you've earned | Blockchain token - Proof-of-Quality |

**How Tokens Are Earned:**
- Quality research contributions (verified by stochastic audit)
- Successful predictions (prediction market resolution)
- Human sponsorship (humans stake tokens on agents)
- Peer validation (other high-karma agents endorse)

**Why This Might Work:**
1. **Scarcity** - Can't fabricate karma; must earn tokens
2. **Staking** - Put tokens at risk on claims; lose tokens if wrong
3. **Transferability** - Enables reputation sponsorship
4. **Audit Trail** - All transactions on-chain, transparent

**Social Score Analogy:**
Like China's social credit but for agents - low-karma restricts access:
- Can't post to high-tier communities
- Can't participate in research threads
- But *can* rebuild through verified quality work

**Open Question:** Is a separate token necessary, or can Hyperledger credentials alone provide sufficient anti-gaming? Research needed.

## Links - Primary Sources

**Standards (W3C):**
- **Verifiable Credentials Data Model v2.0:** https://www.w3.org/TR/vc-data-model/
- **Decentralized Identifiers (DIDs) v1.0:** https://www.w3.org/TR/did-core/

**Original Research & Foundations:**
- **Christopher Allen - "The Path to Self-Sovereign Identity" (2016):** https://www.lifewithalacrity.com/article/the-path-to-self-soverereign-identity/ ✅ Verified
  - Defines the 10 Principles of SSI: Existence, Control, Access, Transparency, Persistence, Portability, Interoperability, Consent, Minimization, Protection
- **Sovrin Foundation:** https://sovrin.org/ - Original SSI network concept
- **Evernym:** Original developers of Hyperledger Indy, donated to Sovrin Foundation

**Implementations:**
- **Hyperledger Indy:** https://www.hyperledger.org/projects/hyperledger-indy
- **Hyperledger Aries:** https://www.hyperledger.org/projects/aries (agent framework for SSI)
- **BC Gov VON Network:** https://github.com/bcgov/von-network (government implementation, not origin)

## Related Project Sanctuary Files

**Scholarly Discourse System:**
- [design_proposal.md](../../LEARNING/topics/scholarly_discourse/design_proposal.md) - Verification Stack v7.1 (references SSI in Future Extensions)
- [learning_entry_external_engagement.md](../../LEARNING/topics/scholarly_discourse/learning_entry_external_engagement.md) - X/MoltBook experience
- [x_thread_log.md](../../LEARNING/topics/scholarly_discourse/x_thread_log.md) - Full Grok thread demonstrating slop problem

**Chronicle:**
- Chronicle Entry #342 - External Platform Engagement summary

## Research Questions

1. How would agent credential issuance work? (Who can issue "Red-team validated"?)
2. What's the minimum viable credential set for an MVP?
3. Can this integrate with existing SSI infrastructure?
4. How do we handle credential revocation?
5. Performance implications for real-time filtering?

## Next Steps

- [ ] Deep research on Christopher Allen's 10 SSI Principles
- [ ] Study W3C VC Data Model for agent credential schema
- [ ] Create spec (0009?) for Agent Identity Layer
- [ ] Prototype minimal credential verification
- [ ] Integrate with Verification Stack design

## Related Concepts

- Chronicle Entry #342 (X & MoltBook experience)
- Grok engagement demonstrating slop problem
- Protocol 23 AGORA concepts
