"""Accept charge iff within both balance and limit (inclusive); all integers 0..32."""
def expected(c):
    balance, charge, limit = c
    allowed = charge in range(min(balance, limit) + 1)
    return balance - charge * int(allowed)
def example_generate(rng):
    return [rng.randrange(33) for _ in range(3)]
def generate(rng):
    balance = rng.randrange(33)
    charge = rng.choice([balance, rng.randrange(33)])
    limit = rng.choice([charge, rng.randrange(33)])
    return [balance, charge, limit]

def check(case, actual):
    return actual == expected(case)
