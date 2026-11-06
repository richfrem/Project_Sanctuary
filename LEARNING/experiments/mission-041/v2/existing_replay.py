import sys
from pathlib import Path
import importlib.util
sys.path.insert(0, str(Path(__file__).resolve().parent))
import specifications
claim_path = Path(__file__).resolve().parent.parent / 'fixtures' / 'claim_replay.py'
spec = importlib.util.spec_from_file_location('old_claim', claim_path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
CASES = iter(specifications.domain('replay'))
def generate(rng):
    return next(CASES)
def example_generate(rng):
    return next(CASES)
def expected(c):
    return old.expected(c)
def check(c, actual):
    return old.check(c, actual)
