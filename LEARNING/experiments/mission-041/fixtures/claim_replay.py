"""First event per ID contributes signed value. IDs 0..8; values -20..20; length 0..10."""
def expected(c):
    return sum(dict(reversed(c)).values())
def example_generate(rng):
    return [[rng.randrange(9), rng.randrange(-20,21)] for _ in range(rng.randrange(11))]
def generate(rng):
    return example_generate(rng)

def check(case, actual):
    return actual == expected(case)
