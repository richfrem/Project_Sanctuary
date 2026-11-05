"""Clamping returns median(value,lower,upper), ordered bounds; integers -32..32."""
def expected(c):
    return sorted(c)[1]
def example_generate(rng):
    lower, upper = sorted([rng.randrange(-32,33), rng.randrange(-32,33)])
    return [rng.randrange(-32,33), lower, upper]
def generate(rng):
    value, lower, upper = example_generate(rng)
    return [rng.choice([value, lower, upper]), lower, upper]

def check(case, actual):
    return actual == expected(case)
