# MISSION-076 discovery note (associative exploration)

Prompt: `prompts/associative-mission-discovery.md`. Richard supplied five stimulus topics and said they were meant "to stimulate creativity", not as constraints. Selection method: **deliberate diverse selection** (not random sampling; no seed). Domain descriptions come from learned knowledge; details marked (?) are uncertain and were not verified because the mission does not rely on them.

## Domains

| Domain | Concrete object | Mechanism / constraint | Surprise or tension |
|---|---|---|---|
| Ant trails (stigmergy) | Foraging trails of ants such as *Lasius niger* | Deposit pheromone where you walk; follow stronger scent; scent evaporates (timescale minutes to hours, species dependent (?)) | A colony with no plan builds a globally efficient trail, yet a trail can also die out if deposit is too weak against evaporation |
| Scribal copying and marginalia | Glossed manuscripts copied by hand, each copy keeping some of its predecessor's marginal notes | A copyist keeps, drops or adds notes; a visible note prompts further notes nearby; a note can point elsewhere ("see f. 12v") (?) | A tradition of annotation can vanish in a few copying generations even though every individual copy was made faithfully enough |
| Coral reefs | A reef patch hit by bleaching or acidification | Local recovery by regrowth and by larval settlement; larvae can arrive from reefs tens to hundreds of kilometres away (?) | Reefs recover from distant sources, so a locally dying patch is not decided by local rates alone |
| Critical phenomena (statistical physics) | The contact process / directed percolation on a line | A population with an absorbing "all dead" state has a sharp threshold; at threshold survival decays as a power law, P(t) ~ t^-0.1595 | One universality class covers epidemics, forest fires, catalysis, so the model is almost independent of what the "individuals" are |

Non-computing domains: ants, manuscripts, reefs (three). Physics is the formal domain.

## Crossings explored (8)

1. **Trail = annotation tradition (mechanism transfer).** Pheromone deposit, reinforcement and evaporation map to "a note prompts neighbouring notes, notes are dropped in copying". Both reduce to a branching process with an absorbing dead state. Descendant: the extinction threshold is a sharp, measurable thing.
2. **Reverse a constraint: remove evaporation.** Ant trails with no decay saturate; manuscript traditions with perfect copying keep every note. Boring; dropped.
3. **Change representation: time as copying generation.** A page is a row of positions, each generation is the next copy. This is a spatial design (1 space + 1 time dimension) and leads to directed percolation (DP) directly.
4. **Deceptive similarity (ants).** Real ant recruitment is nonlinear (more scent attracts disproportionately, a quorum effect), so a trail needs a critical mass. Marginalia need one reader. The analogy fails exactly here, which generates a question for later (Allee-effect variant, first-order transition). Kept as an inbox candidate, not done.
5. **Reefs add long-range recruitment.** Larval dispersal and cross-references ("see also f. 12v") both put a descendant far from the parent. Combined question: how much does a *few* long-range links help a local tradition near its threshold?
6. **Combine contrasts: local vs distant rescue.** Compare the same added recruitment placed next door versus anywhere on the page. Predicted to differ in how rescue time scales with the amount added.
7. **Awkward idea given a fair run: coral calcification as dependency-graph drift (user topic 5).** Developed to the point of seeing it needs a coherence metric I would have to invent, with no sourced mapping from pH to version constraints; I dropped it for this mission rather than manufacture a model.
8. **Joinery as fault-tolerant protocol (user topic 2).** Interesting, but the first step (abstract joint geometries into recovery rules) is a design exercise whose output is hard to falsify cheaply. Not selected.

## Candidate briefs

**A. Rescue time of a near-critical annotation tradition by cross-references (selected).**
- Question: if a tradition sits exactly at its extinction threshold, how does the time at which it is "rescued" (survival probability doubles relative to no rescue) scale with the amount of extra recruitment, when the extra recruitment is long-range (cross-reference or larval dispersal) versus nearest-neighbour (matched substitute)?
- Attraction: three sources (ants, manuscripts, reefs) share one abstract mechanism and I can compute the answer. Uncertain: whether the heuristic scaling holds at accessible times. Association changes the question and the design (matched local substitute) but not the mathematics.
- Next step: C simulation of bond directed percolation with optional long-range links, independent Python check, frozen exponents.
- Risk: finite-time corrections; known adjacent literature (contact process on small-world networks is mean-field); contribution is a rescue-time scaling, not new universality.

**B. Dependency graph under version drift as calcification (not selected).** Needs an invented coherence measure; no falsifiable step affordable now.

**C. Birdsong syntax with model checking (not selected).** Needs published song sequences; cannot verify sources offline in this session.

## Selection decision

A chosen: affordable (CPU only), falsifiable with a matched control, learning value (exponent of a rescue time I did not know), productive tension (local vs distant). The association is **generative** for the question and the control design; it is not claimed to be evidence for the mechanism in manuscripts, ants or reefs. No claim that this process improves creativity or novelty.

## Prior-work check

- Repo: only MISSION-074 shares terms (percolation). No mission on contact process, stigmergy or marginalia.
- Web (one search, snippets only, not full papers): 1D DP exponent delta = 0.1595; the contact process on Watts-Strogatz small-world networks is mean-field for any rewiring probability p > 0 (arXiv 1307.6186). So "shortcuts change the universality class" is **known** and not claimed. The rescue-time scaling with added long-range versus local recruitment was not found in the sources inspected; that is "a distinct framing relative to sources inspected", not established novelty.
