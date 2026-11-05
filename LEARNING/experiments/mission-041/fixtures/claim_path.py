"""Normalize absolute root-clamped token paths; a,b,c,.,..; length 0..8."""
import posixpath
def expected(c):
    return posixpath.normpath('/' + '/'.join(c))
def example_generate(rng):
    return [rng.choice(['a', 'b', 'c', '.', '..']) for _ in range(rng.randrange(9))]
def generate(rng):
    return example_generate(rng)

def check(case, actual):
    return actual == expected(case)
