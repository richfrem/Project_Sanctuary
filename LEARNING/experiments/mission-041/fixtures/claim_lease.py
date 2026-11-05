"""Lease may be acquired by same owner OR at/after expiry; integer clocks 0..32."""
def expected(c):
    return any([c[2], c[0] >= c[1]])
def example_generate(rng):
    return [rng.randrange(33), rng.randrange(33), bool(rng.randrange(2))]
def generate(rng):
    now = rng.randrange(33)
    return [now, rng.choice([max(0, now-1), now, min(32, now+1)]), bool(rng.randrange(2))]

def check(case, actual):
    return actual == expected(case)
