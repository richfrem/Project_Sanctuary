---
id: drq_learning_loop_synthesis
type: insight
status: active
last_verified: 2026-01-11
---

# Technical Synthesis: DRQ Patterns → Learning Loop Evolution

> **Core Insight:** DRQ treats code evolution as a *game* with measurable fitness. We can treat learning loop outputs the same way.

## Pattern Extraction from DRQ Codebase

### 1. Prompt Architecture (Minimalism)

**DRQ Prompts:**
```
# new_prompt_0.txt (230 chars)
Create a new valid Core War program in redcode. Be creative. 
Write only the new program (with comments explaining what it does) 
and nothing else. ONLY DEFINE LABELS ON THE SAME LINE AS AN INSTRUCTION. 
Wrap program around ``` tags.

# mutate_prompt_0.txt (310 chars)
Mutate (change) the following Core War program in a way that is 
likely to improve its performance (survive and kill other programs). 
Write only the new updated program (with comments explaining what it does) 
and nothing else. ONLY DEFINE LABELS ON THE SAME LINE AS AN INSTRUCTION. 
Wrap program around ``` tags.
```

**System Prompt:** 15KB of domain specification (Redcode reference + examples)

**Pattern:** Tiny action prompts + comprehensive domain context = emergent complexity

**Application to Sanctuary:**
- **Split prompts:** Large `sanctuary-guardian-prompt.md` → small action prompts + domain reference
- **Action prompts:** "Generate a chronicle entry for this session" (~50 words)
- **Domain context:** ADR summaries, protocol specs (loaded once)

---

### 2. Map-Elites Diversity Preservation

**DRQ Code (`drq.py` lines 60-88):**
```python
class MapElites:
    def __init__(self):
        self.archive = {}  # bc -> phenotype (behavioral characteristic -> solution)
    
    def place(self, phenotype):
        # Only replace if new solution is BETTER in that behavioral niche
        place = (phenotype.bc not in self.archive) or 
                (phenotype.fitness > self.archive[phenotype.bc].fitness)
        if place:
            self.archive[phenotype.bc] = phenotype
```

**Behavioral Axes (`bc_axes`):**
- `tsp` = total_spawned_procs (how aggressively it replicates)
- `mc` = memory_coverage (how much of the arena it controls)

**Pattern:** Keep BEST solution for each *behavioral niche*, not just overall best

**Application to Sanctuary:**
- **Learning Archive:** Track outputs by behavioral characteristic
  - Axis 1: `depth` (shallow overview → deep technical)
  - Axis 2: `scope` (single file → system-wide)
- **Chronicle Diversity:** Don't overwrite entries that explore different niches
- **ADR Diversity:** Track ADRs by domain (security, performance, architecture)

---

### 3. Cumulative Adversarial History

**DRQ Code (`drq.py` lines 164-188):**
```python
def process_warrior(self, i_round, gpt_warrior):
    # Get ALL previous champions, not just latest
    prev_champs = [self.all_rounds_map_elites[i].get_best() for i in range(i_round)]
    
    # Evaluate against initial opponents + ALL previous champions
    opps = self.init_opps + prev_champs
    outputs = run_multiple_rounds(self.simargs, [gpt_warrior, *opps], ...)
```

**Pattern:** Each round must beat ALL previous winners, not just the latest

**Application to Sanctuary:**
- **Red Team Cumulative History:** Each audit includes edge cases from ALL previous audits
- **Learning Validation:** New knowledge must be consistent with ALL previous validated knowledge
- **Regression Prevention:** Don't just test new code, test against historical failure cases

---

### 4. Fitness Threshold Gating

**DRQ Code (`drq.py` lines 234-240):**
```python
best_fitness = me.get_best().fitness if len(me.archive) > 0 else -np.inf
should_skip = best_fitness > self.args.fitness_threshold  # 0.8 default

if not should_skip:
    if i_iter == 0:
        self.init_round(i_round)
    self.step(i_round)
```

**Pattern:** Only move to next round when fitness exceeds threshold

**Application to Sanctuary:**
- **Don't proceed until quality gate passes**
- **Explicit quality metrics for each phase:**
  - Discover: Source verification score > 0.9
  - Synthesize: Coverage of source material > 0.8
  - Validate: Red Team approval + no critical issues

---

### 5. Simple Task × Many Iterations = Emergence

**DRQ Algorithm Summary:**
```
for round in range(N_ROUNDS):  # 20 rounds
    for iter in range(N_ITERS):  # 250 iterations per round
        if random() < 0.1:
            warrior = llm.new()  # 10% generate fresh
        else:
            warrior = llm.mutate(archive.sample())  # 90% mutate existing
        
        score = evaluate(warrior, all_previous_champions)
        archive.place(warrior)  # Only keeps if better for its niche
```

**Total iterations:** 20 × 250 = 5,000 simple LLM calls
**Result:** Superhuman Core War strategies

---

## Proposed Learning Loop Evolution: Protocol 128 v4.0

### Current Loop (Sequential Human-Gated):
```
Scout → Synthesize → [HUMAN GATE] → Audit → [HUMAN GATE] → Seal
```

### Proposed Loop (DRQ-Inspired):
```
                    ┌─────────────────────────────────────┐
                    │         ADVERSARIAL ARENA           │
                    │  (Cumulative validation history)    │
                    └─────────────────────────────────────┘
                                    ▲
                                    │ evaluate
                                    │
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  SCOUT  │───▶│SYNTHESIZE│───▶│  MUTATE │───▶│ ARCHIVE │
│ (init)  │    │(generate)│    │(improve)│    │(Map-Elites)│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                    │               ▲              │
                    │               │              │
                    └───────────────┘              │
                      90% mutate from              │
                      archived outputs             │
                                                   ▼
                                          ┌─────────────┐
                                          │   SEAL      │
                                          │ (if fitness │
                                          │  > threshold)│
                                          └─────────────┘
```

### Concrete Implementation Steps

#### Step 1: Prompt Simplification
```
# Current prompt: sanctuary-guardian-prompt.md (~30KB)
# Proposed split:

domain_context.md:     ~15KB (ADR summaries, protocol specs, identity)
action_learn.md:       ~300 chars ("Synthesize key insights from [sources]...")
action_audit.md:       ~300 chars ("Review [output] for accuracy and gaps...")
action_chronicle.md:   ~300 chars ("Create chronicle entry for [topic]...")
```

#### Step 2: Behavioral Archive
```python
class LearningArchive:
    def __init__(self):
        self.archive = {}  # (depth, scope) -> LearningOutput
    
    def place(self, output):
        bc = (
            self.measure_depth(output),   # 0-5: shallow to deep
            self.measure_scope(output)    # 0-5: narrow to broad
        )
        if bc not in self.archive or output.quality > self.archive[bc].quality:
            self.archive[bc] = output
```

#### Step 3: Cumulative Validation
```python
def validate_output(new_output, archive):
    # Must pass ALL previous edge cases
    all_edge_cases = load_cumulative_edge_cases()
    for edge_case in all_edge_cases:
        if not new_output.handles(edge_case):
            return False, edge_case
    return True, None
```

#### Step 4: Fitness Metrics
```python
FITNESS_THRESHOLD = 0.8

def calculate_fitness(output):
    return (
        source_coverage(output) * 0.3 +      # Did it use all sources?
        accuracy_score(output) * 0.3 +        # Is it factually correct?
        consistency_score(output) * 0.2 +     # Consistent with prior knowledge?
        novelty_score(output) * 0.2           # Does it add new insight?
    )
```

---

## Implementation Priority

1. **Immediate (This Session):**
   - [ ] Create simplified action prompts
   - [ ] Define behavioral characterization axes

2. **Next Session:**
   - [ ] Implement LearningArchive class
   - [ ] Create cumulative edge case registry

3. **Future:**
   - [ ] Automated fitness scoring
   - [ ] Self-play prompt evolution

---

## Key Takeaways

| DRQ Principle | Learning Loop Application |
|---------------|---------------------------|
| 230-char prompts | Split guardian prompt into action + context |
| Map-Elites | Archive diverse outputs by (depth, scope) |
| Cumulative opponents | Accumulate all edge cases from all audits |
| Fitness threshold | Don't seal until quality > 0.8 |
| 90% mutate / 10% new | Mostly refine existing knowledge |
