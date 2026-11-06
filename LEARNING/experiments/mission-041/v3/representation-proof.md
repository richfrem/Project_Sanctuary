# Migration set representation and density measure

The migration contract takes `(version, set_of_keys)`, represented by a JSON
array. `domain('migration')` enumerates every version and every subset of nine
keys once: 32 * 512 = 16,384 semantic inputs. Uniform density means uniform over
these semantic inputs, not uniform over permutations of JSON representations.

The property pool reverses key-array order to test representation independence.
Such arrays represent the same legal set. In `components.migration` and EVERY
migration candidate, the first operation is `data = {k:1 for k in keys}`. All
subsequent observations are membership, pop, assignment and sorted output keys;
none observes insertion order or original array length. Therefore permutations
(and repetitions) of the input key array leave reference AND every candidate's
output unchanged. Canonical subset enumeration exhausts all semantic inputs and
extends to every representation by this source-level argument. No migration
candidate was excluded as equivalent (all four have exhaustive witnesses).

Phase-1 tests initially compared exact JSON representations for domain membership
and Python tuples against deserialized lists for suite reproduction. Those test
assertions were corrected to compare semantic sets / JSON values. The sealed
specification, generators and suites were not changed. Initial failing test output
is retained; no arm was run against any held-out candidate during these checks.
