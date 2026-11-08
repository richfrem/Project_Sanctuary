# MISSION-067 discovery note (associative exploration, pre-registration)

Selection method: **deliberate diverse selection** (not random; no sampler used). No domain ledger exists; provenance is kept here.

## Domains
1. **English change ringing** (history/physical practice). Phenomenon: church bells rung in permuted orders ("rows"). Constraint: a heavy bell swings through a full circle each stroke, so between consecutive rows each bell can move at most one place. A "change" is therefore a set of disjoint swaps of adjacent bells. Tension: ringers prize *extents* (every row exactly once), a deterministic, exhaustive walk, never randomness.
2. **Card shuffling / Markov chain mixing** (mathematics). Mechanism: repeated random moves drive a deck toward uniform; mixing time depends on what moves are allowed. Known (from memory, unverified here): random adjacent transpositions mix slowly, on the order of n^3 log n steps.
3. **Braided rivers** (geomorphology). Channels split and rejoin; sediment moves only locally, but the braid pattern rearranges globally. Uncertain: how fast local exchange produces global reordering.
4. **Honeybee waggle-dance recruitment** (biology). Many local, parallel signals at once rather than one global broadcast. Uncertain mapping to anything permutational.

## Crossings (short)
- Ringing x shuffling: *a change is a parallel adjacent-swap shuffle*. If a band rang changes at random instead of by method, how many changes until the order is "well mixed"? **(developed)**
- Ringing x rivers: bells drift one place per row like sediment one bar per flood; descendant: "hunting" bells (moving steadily one place per row) resemble a channel migrating. Kept as provenance only.
- Shuffling x bees: parallel local moves vs one global move. Descendant, combined with ringing: does parallelism *save work* (total swaps), or only *time* (number of changes)? **(this became H2)**
- Rivers x shuffling (awkward idea, developed fairly): braided channels as a sorting network. Shared vocabulary ("exchange") but no clear stochastic model; dismissed for now, with no claim made.
- Deceptive similarity: ringing forbids repeating a row, but random shuffles repeat freely. So the random-change chain is *not* ringing; it only borrows the ringing move set. Recorded as a limitation.

## Candidate briefs
1. **Random-change shuffle mixing** (ringing + shuffling + bee parallelism). Exactly computable on S_n for n <= 8 with stdlib. Interest: does the physical "one place per row" rule still allow fast mixing once many swaps happen in parallel? Risk: per-step speedup may be trivial (it does more swaps). That is why H2 counts swaps. **Selected.**
2. Do classic methods (Plain Bob) produce row sequences that pass randomness tests? Rejected: an extent is deterministic and the test choice dominates the answer.
3. Braided-river sorting network. Rejected: no defensible model in this session.

Prior-work check: no Sanctuary mission or dossier mentions change ringing or shuffling (grep). Broader originality is not claimed. Random adjacent transposition is classical, and the parallel-matching variant may well be studied in the literature (not checked). The contribution is a new-here exact computation.
