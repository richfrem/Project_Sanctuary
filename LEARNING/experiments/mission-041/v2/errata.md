# V2 errata (independent review: changes requested)

V2 reproduced, but its near-ceiling equal-budget example detection made the
benchmark non-diagnostic for the proposed property-testing advantage. Its
“held-out” operators were procedurally withheld by the same author, not blind.

Of 120 recipes, 69 of the 73 reported “equivalents” were no-op input transforms.
There were 51 real applicable candidates, of which 4 were true equivalents and
47 non-equivalents. These input transforms were not source-level implementation
faults. The 12/12 existing-tests figure covers only 6 components; 14 components
had no tests. It must not be read as coverage of all 20 components.

V1 and V2 source files, artifacts and numerical results remain intact. V3 is a
separate source-mutation experiment; no results are pooled across versions.
