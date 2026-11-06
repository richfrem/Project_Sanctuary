"""Frozen claim wrapper for lease."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import specifications as spec
NAME = 'lease'
def generate(rng):
    return spec.property_generate(NAME, rng)
def example_generate(rng):
    return spec.example_generate(NAME, rng)
def expected(c):
    return spec.expected(NAME, c)
def check(c, actual):
    return actual == spec.expected(NAME, c)
