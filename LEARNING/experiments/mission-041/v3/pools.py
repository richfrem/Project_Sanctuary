"""Claim-derived probes and conventional examples; no mutant dependencies.

Each property probe costs two candidate calls. No adaptive selection or retries.
Existing is a NEW conventional pre-mutant suite, never labeled historical.
"""
import random
from specifications import NAMES, TOKENS, BASENAMES, ENUMS, domain

SEED = 41053
BUDGET = 16
# Literal expected values, reviewed independently of components.
EXISTING = {
 'lease': [([0, 0, False], True), ([10, 11, False], False), ([11, 10, False], True), ([0, 127, True], True), ([127, 127, False], True)],
 'renew': [([0, 10, 5], 10), ([5, 10, 10], 15), ([10, 10, 20], 10), ([11, 10, 20], 10), ([0, 31, 31], 31)],
 'release': [([0, 0, 0], 0), ([2, 2, 3], 2), ([2, 3, 3], 3), ([31, 31, 1], 0), ([0, 31, 31], 31)],
 'fence': [([0, 0], 0), ([5, 6], 6), ([6, 5], 6), ([255, 255], 255), ([0, 255], 255)],
 'path': [([], '/'), (['..'], '/'), (['a', '.', 'b'], '/a/b'), (['a', '..', 'b'], '/b'), (['..', '..', 'a'], '/a')],
 'relative': [([], '.'), (['..'], '..'), (['a', '..'], '.'), (['..', '..', 'a'], '../../a'), (['a', '.', 'b'], 'a/b')],
 'extension': [([], ''), (['.env'], ''), (['a.'], ''), (['a.py'], 'py'), (['a.b.c'], 'c'), (['a.py', 'b.PY'], 'PY')],
 'path_key': [([], ''), (['.'], ''), (['a', '.', 'b'], 'a/b'), (['..', 'a'], '../a'), (['a', 'a'], 'a/a')],
 'replay': [([], 0), ([[0, 1]], 1), ([[0, 1], [0, -1]], 1), ([[0, -1], [1, 1]], 0), ([[3, 0], [3, 1]], 0)],
 'last_replay': [([], []), ([[0, 1]], [[0, 1]]), ([[0, 1], [0, -1]], [[0, -1]]), ([[1, 1], [0, -1]], [[0, -1], [1, 1]]), ([[3, 0]], [[3, 0]])],
 'revision': [([], 0), ([0], 0), ([1, 2, 3], 3), ([15, 1, 2], 15), ([4, 4], 4)],
 'tombstone': [([], []), ([[0, 'delete']], []), ([[0, 'put'], [0, 'delete']], []), ([[0, 'delete'], [0, 'put']], [0]), ([[7, 'put'], [1, 'put']], [1, 7])],
 'accounting': [([0, 0, 0], 0), ([10, 4, 5], 6), ([10, 11, 31], 10), ([10, 4, 3], 10), ([31, 31, 31], 0)],
 'clamp': [([0, -2, 2], 0), ([-3, -2, 2], -2), ([3, -2, 2], 2), ([2, 2, 2], 2), ([-32, -32, 32], -32)],
 'reserve': [([10, []], 10), ([10, [3, 4]], 3), ([7, [7]], 0), ([5, [3, 4]], 5), ([0, [0, 0]], 0)],
 'refund': [([0, 0, 0], 0), ([4, 3, 10], 7), ([4, 6, 10], 10), ([4, 7, 10], 10), ([31, 0, 31], 31)],
 'schema': [(['new', 'open', 0], 'active'), (['active', 'close', 0], 'closed'), (['closed', 'reset', 0], 'new'), (['new', 'close', 0], 'new'), (['new', 'open', 1], 'new'), (['closed', 'reset', 255], 'closed')],
 'migration': [([0, []], []), ([0, ['A']], ['a']), ([1, ['A']], ['A']), ([0, ['A', 'a', 'B']], ['B', 'a']), ([31, ['H']], ['H'])],
 'required': [([], ['A', 'B', 'C']), (['A', 'B', 'C'], []), (['A', 'A'], ['B', 'C']), (['D'], ['A', 'B', 'C']), (['C', 'B'], ['A'])],
 'enum_map': [([], []), (list('AaBb'), [0, 1, 2, 3]), (['C'], [-1]), (['a', 'A'], [1, 0]), (['N', 'b'], [-1, 3])],
}


def property_probe(name, r, i):
    """Construct one domain-specific two-call law, without reading candidates."""
    if name == 'lease':
        t = r.randrange(127)
        a = [t, t + r.choice([0, 1]), bool(i % 2)]
        return a, [t+1, a[1], a[2]], 'nondecreasing'
    if name == 'renew':
        t = r.randrange(31); d = r.randrange(31)
        a = [t, min(31, t+r.choice([0, 1, 2])), d]
        return a, [t, a[1], d+1], 'nondecreasing'
    if name == 'release':
        owner = r.randrange(32); depth = r.choice([0, 1, 2, 31])
        return [owner, (owner+1) % 32, depth], [owner, owner, depth], 'nonincreasing'
    if name == 'fence':
        t = r.randrange(255)
        return [t, t], [t, t+1], 'nondecreasing'
    if name in {'path', 'relative', 'path_key'}:
        prefix = [r.choice(TOKENS) for _ in range(r.randrange(3))]
        a = prefix + r.choice([['..'], ['a'], ['.']])
        return a, a+['.'], 'equal'
    if name == 'extension':
        b = BASENAMES[i % len(BASENAMES)]
        return [b], ['file', b], 'equal'
    if name in {'replay', 'last_replay'}:
        k = r.randrange(4); v = r.choice([-1, 0, 1])
        a = [[k, v], [(k+1) % 4, -v], [k, -v]]
        # Duplicating an identical final event preserves both replay contracts.
        return a, a+[a[-1]], 'equal'
    if name == 'revision':
        a = [r.randrange(16) for _ in range(3)]
        return a, a+[0], 'equal'
    if name == 'tombstone':
        k = r.randrange(8)
        a = [[k, 'put'], [k, r.choice(['put', 'delete'])]]
        return a, a+[a[-1]], 'equal'
    if name == 'accounting':
        b = r.randrange(32); a = r.choice([0, b, min(31, b+1)])
        limit = r.randrange(31)
        return [b, a, limit], [b, a, limit+1], 'nonincreasing'
    if name == 'clamp':
        lo = r.randrange(-31, 32); hi = r.randrange(lo, 33)
        v = r.choice([lo-1, lo, min(31, hi)])
        return [v, lo, hi], [v+1, lo, hi], 'nondecreasing'
    if name == 'reserve':
        cap = r.randrange(32); a = [r.randrange(8), r.randrange(8)]
        return [cap, a], [cap, a+[0]], 'equal'
    if name == 'refund':
        cap = r.randrange(32); b = r.randrange(cap+1); a = r.randrange(31)
        return [b, a, cap], [b, a+1, cap], 'nondecreasing'
    if name == 'schema':
        state, event = [('new', 'open'), ('active', 'close'), ('closed', 'reset'), ('new', 'close')][i % 4]
        # Unsupported versions all preserve state.
        return [state, event, r.randrange(1, 255)], [state, event, 255], 'equal'
    if name == 'migration':
        keys = r.choice([[], ['A'], ['A', 'a'], ['A', 'B', 'H']])
        version = i % 2
        return [version, keys], [version, list(reversed(keys))], 'equal'
    if name == 'required':
        a = r.sample(list('ABC'), i % 4)
        return a, a+['N'], 'equal'
    if name == 'enum_map':
        a = r.sample(list('AaBb'), 3)
        return a, list(reversed(a)), 'reverse'
    raise KeyError(name)


def relation(law, a, b):
    return {'equal': lambda: a == b, 'nondecreasing': lambda: a <= b,
            'nonincreasing': lambda: a >= b, 'reverse': lambda: a == list(reversed(b))}[law]()


def build():
    result = {}
    for offset, name in enumerate(NAMES):
        r = random.Random(SEED+offset)
        probes = [property_probe(name, r, i) for i in range(BUDGET//2)]
        # Uniform over the COMPLETE declared finite domain, plus four ordinary
        # engineer examples (empty/zero/equality/typical bounds, no witnesses).
        cases = list(domain(name))
        r = random.Random(SEED+offset)
        examples = [c for c, _ in EXISTING[name][:4]] + [cases[r.randrange(len(cases))] for _ in range(BUDGET-4)]
        result[name] = {'property': probes, 'example': examples,
                        'existing': EXISTING[name], 'existing_provenance': 'authored_v3_before_mutants',
                        'property_calls': BUDGET, 'example_calls': BUDGET}
    return result
