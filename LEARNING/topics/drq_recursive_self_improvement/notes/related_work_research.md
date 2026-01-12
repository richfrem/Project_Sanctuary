---
id: drq_related_work_research
type: guide
status: active
last_verified: 2026-01-11
---

# Related Work Research: Self-Play, Quality-Diversity, and LLM Evolution

> **Purpose:** Deep research to ground the DRQ application proposal in established prior art.

---

## 1. Map-Elites: Quality-Diversity Foundation

**Source:** Academic research, originally by Mouret & Clune (2015)

### Core Concept
Map-Elites is a **Quality-Diversity (QD)** algorithm that balances:
- **Quality:** High performance/fitness
- **Diversity:** Significant behavioral differences between solutions

Unlike traditional evolutionary algorithms that converge to a single optimum, Map-Elites maintains an **archive** of elite solutions across a feature space.

### Key Metrics
| Metric | Definition | Sanctuary Application |
|--------|------------|----------------------|
| **Coverage** | Fraction of archive cells filled | How many learning niches are explored? |
| **QD-Score** | Sum of fitness values across all occupied cells | Total quality across all diverse outputs |
| **Global Best** | Single highest fitness found | Best individual output |

### Algorithm (Simplified)
```
1. Initialize empty archive grid (feature_dim_1 × feature_dim_2)
2. Generate random initial solutions
3. For each solution:
   a. Evaluate fitness
   b. Compute behavioral features → grid cell
   c. If cell empty OR new > existing: replace
4. Repeat: sample from archive → mutate → evaluate → place
```

### Advantages
- Avoids local optima by maintaining diverse candidates
- "Illuminates" the search space
- More robust solutions emerge

### Disadvantages
- Requires domain knowledge to define feature dimensions
- Grid size grows exponentially with dimensions
- High memory/compute for high-dimensional spaces

---

## 2. AlphaGo Zero: Self-Play Recursive Improvement

**Source:** DeepMind (2017) - https://deepmind.google

### Key Innovation: Tabula Rasa Learning
AlphaGo Zero started with **no human knowledge** beyond basic rules. It learned entirely through self-play.

### Recursive Self-Improvement Loop
```
1. Start with random neural network
2. Self-play games using current network + MCTS
3. Win/loss → reward signal → update network
4. Updated network plays more games
5. Repeat → progressively stronger
```

### Results
- After 3 days: Beat version that defeated Lee Sedol (100-0)
- Discovered novel strategies never conceived by humans
- "Move 37" example: Initially appeared wrong to humans, proved pivotal

### Key Insight for Sanctuary
> "No longer limited by the scope of human knowledge or biases, enabling AI to discover novel and superior strategies."

**Application:** Learning loop should aim for emergent insight, not just human knowledge reproduction.

---

## 3. Sakana AI Scientist: LLM-Driven Research

**Source:** Sakana AI (August 2024) - https://sakana.ai

### What It Does
- Generates novel research ideas
- Writes code and runs experiments
- Drafts complete scientific papers
- Self-evaluates via automated LLM reviewer

### Key Stats
- **Cost:** ~$15 per full research paper
- **Quality:** Papers "exceed acceptance threshold" for ML conferences
- **AI Scientist-v2 (2025):** Paper accepted to ICLR workshop (later withdrawn for transparency)

### Automated Review Loop
```
LLM generates paper
    ↓
LLM reviewer evaluates (near-human accuracy)
    ↓
Feedback → LLM refines paper
    ↓
Iterate until quality threshold met
```

### Relevance to DRQ
This is the same organization behind DRQ. The AI Scientist demonstrates their broader vision: **autonomous AI doing AI research** - the recursive self-improvement theme.

---

## 4. DeepMind FunSearch: Evolutionary Code Discovery

**Source:** DeepMind (2023, Nature paper) - https://deepmind.google

### Core Innovation
Combines LLMs with evolutionary algorithms for **code evolution** that makes **verifiable discoveries**.

### Architecture
```
┌─────────────────────────────────────────────┐
│              FunSearch Loop                  │
├─────────────────────────────────────────────┤
│  1. Start with seed program                 │
│  2. LLM generates mutations                 │
│  3. Automated EVALUATOR checks correctness  │
│  4. High-scoring programs → pool            │
│  5. Sample from pool → LLM mutates          │
│  6. Repeat                                  │
└─────────────────────────────────────────────┘
```

### Key Breakthrough
- Solved **cap set problem** (open math problem)
- Found more efficient **bin-packing algorithms**
- First LLM system to make **verifiable scientific discoveries**

### The "Evaluator" Pattern
> To mitigate LLM hallucinations, FunSearch pairs the LLM with an **automated evaluator** that rigorously checks and scores generated programs.

**Application to Sanctuary:** Our Red Team + Protocol 128 verification IS the evaluator. We should formalize it.

### AlphaEvolve (Follow-up)
- Extends FunSearch to evolve **entire codebases**
- Multiple programming languages
- Not just single functions

---

## 5. Comparative Synthesis

| System | Target Domain | Evolution Method | Diversity | Cumulative History |
|--------|--------------|------------------|-----------|-------------------|
| **AlphaGo Zero** | Game (Go) | Self-play + MCTS | Implicit via search | Yes (training history) |
| **Map-Elites** | General | Mutation + archive | Explicit (feature grid) | Yes (archive) |
| **FunSearch** | Code/Math | LLM mutation + evaluator | Pool sampling | Yes (scored pool) |
| **AI Scientist** | Research papers | LLM generation + review | N/A (single output) | Yes (iteration history) |
| **DRQ** | Code (Core War) | LLM mutation + play | Map-Elites archive | Cumulative opponents |

---

## 6. Application to Protocol 128 Learning Loop

### Pattern → Application Matrix

| Pattern | Source | Current Protocol 128 | Proposed Evolution |
|---------|--------|---------------------|-------------------|
| **Quality-Diversity Archive** | Map-Elites | No explicit diversity tracking | Track outputs by (depth, scope) grid |
| **Automated Evaluator** | FunSearch | Red Team (external) | Formalize internal evaluator |
| **Cumulative History** | DRQ, AlphaGo | Single-pass Red Team | Accumulate ALL edge cases |
| **Tabula Rasa Discovery** | AlphaGo Zero | Human knowledge reproduction | Allow emergent insights |
| **Self-Play Evolution** | All | One-shot learning | Iterate: generate → evaluate → improve |

### Concrete Next Steps

1. **Define Behavioral Axes for Learning Outputs**
   - Axis 1: Depth (shallow overview → deep technical)
   - Axis 2: Scope (single file → system-wide)
   - Track Coverage and QD-Score

2. **Formalize Evaluator Function**
   - Source coverage: Did it use all cited sources?
   - Accuracy: Is it factually correct?
   - Consistency: Consistent with prior knowledge?
   - Novelty: Does it add new insight?

3. **Implement Cumulative Edge Case Registry**
   - Store ALL Red Team findings
   - New outputs must pass ALL previous edge cases

4. **Enable Emergent Discovery**
   - Allow outputs that go beyond source material
   - Special "Curiosity Vector" outputs for exploration

---

## 7. Verification (ADR 078 Compliance)

### Sources Verified
- [x] DeepMind AlphaGo Zero blog (deepmind.google)
- [x] Sakana AI Scientist announcement (sakana.ai)
- [x] DeepMind FunSearch announcement (deepmind.google)
- [x] Map-Elites academic literature (multiple sources)

### Cross-References
- DRQ paper explicitly cites AlphaGo and evolutionary approaches
- FunSearch and DRQ share the "LLM + evaluator" pattern
- Map-Elites is the diversity preservation mechanism in DRQ

---

## 8. Questions for Red Team

1. **Is the Map-Elites behavioral archive viable for learning outputs?**
   - How do we measure "depth" and "scope" objectively?

2. **Should we implement automated pre-Red-Team evaluation?**
   - Reduce human burden, catch obvious issues early

3. **Is cumulative edge case tracking worth the complexity?**
   - Storage and performance considerations

4. **How do we balance "reproduce human knowledge" vs "emergent discovery"?**
   - Risk of hallucination vs value of novel insights
