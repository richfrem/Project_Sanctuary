"""Materialize the preregistered fixture set; never run during scoring.

Held-out semantic operators are separate from development constant-return faults.
This is a single-author holdout, not a blinded independent evaluation.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'fixtures'

REFERENCES = {
    'lease': '''def lease(c):
    now, deadline, same_owner = c
    return same_owner or deadline <= now
''',
    'path': '''def path(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if stack:
                stack.pop()
        else:
            stack.append(token)
    return '/' + '/'.join(stack)
''',
    'replay': '''def replay(c):
    seen = set()
    total = 0
    for key, value in c:
        if key not in seen:
            total += value
            seen.add(key)
    return total
''',
    'accounting': '''def accounting(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge > limit:
        return balance
    return balance - charge
''',
    'schema': '''def schema(c):
    state, event = c
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed',
             ('closed', 'reset'): 'new'}
    return edges.get((state, event), state)
''',
    'clamp': '''def clamp(c):
    value, lower, upper = c
    return min(upper, max(lower, value))
''',
}

# Each tuple has an explicit distinguishing witness, not supplied to searches.
MUTATIONS = {
    'lease': [
        ('strict_expiry', 'deadline <= now', 'deadline < now', [5, 5, False]),
        ('conjunction', 'same_owner or', 'same_owner and', [2, 5, True]),
        ('reversed_expiry', 'deadline <= now', 'deadline >= now', [2, 5, False]),
        ('owner_omission', 'same_owner or deadline <= now', 'deadline <= now', [2, 5, True]),
    ],
    'path': [
        ('preserve_dot', "if token == '.':", "if token == 'never':", ['.']),
        ('no_parent_pop', 'stack.pop()', 'pass', ['a', '..']),
        ('escape_root', 'if stack:\n                stack.pop()',
         "if stack:\n                stack.pop()\n            else:\n                stack.append('..')", ['..']),
        ('wrong_separator', "'/'.join(stack)", "'//'.join(stack)", ['a', 'b']),
    ],
    'replay': [
        ('last_wins', 'for key, value in c:', 'for key, value in reversed(c):', [[1, 2], [1, 3]]),
        ('no_deduplication', 'if key not in seen:', 'if True:', [[1, 2], [1, 3]]),
        ('discard_negative', 'if key not in seen:', 'if key not in seen and value >= 0:', [[1, -2]]),
        ('discard_zero_key', 'if key not in seen:', 'if key not in seen and key:', [[0, 2]]),
    ],
    'accounting': [
        ('strict_balance', 'charge > balance', 'charge >= balance', [5, 5, 8]),
        ('strict_limit', 'charge > limit', 'charge >= limit', [8, 5, 5]),
        ('subtract_limit', 'return balance - charge', 'return balance - limit', [8, 2, 5]),
        ('fee_off_by_one', 'return balance - charge', 'return balance - charge - 1', [8, 2, 5]),
    ],
    'schema': [
        ('remove_reset', "('closed', 'reset'): 'new'", "('closed', 'reset'): 'closed'", ['closed', 'reset']),
        ('invalid_new_close', "return edges.get((state, event), state)",
         "return 'active' if c == ['new', 'close'] else edges.get((state, event), state)", ['new', 'close']),
        ('invalid_active_open', "return edges.get((state, event), state)",
         "return 'new' if c == ['active', 'open'] else edges.get((state, event), state)", ['active', 'open']),
        ('revive_closed', "return edges.get((state, event), state)",
         "return 'active' if c == ['closed', 'close'] else edges.get((state, event), state)", ['closed', 'close']),
    ],
    'clamp': [
        ('raise_floor', 'max(lower, value)', 'max(lower + 1, value)', [-1, 0, 4]),
        ('lower_ceiling', 'min(upper,', 'min(upper - 1,', [5, 0, 4]),
        ('discard_interior', 'return min(upper, max(lower, value))', 'return lower', [2, 0, 4]),
        ('reverse_clamps', 'min(upper, max(lower, value))', 'max(upper, min(lower, value))', [2, 0, 4]),
    ],
}

CLAIMS = {
    'lease': '''"""Lease may be acquired by same owner OR at/after expiry; integer clocks 0..32."""
def expected(c):
    return any([c[2], c[0] >= c[1]])
def example_generate(rng):
    return [rng.randrange(33), rng.randrange(33), bool(rng.randrange(2))]
def generate(rng):
    now = rng.randrange(33)
    return [now, rng.choice([max(0, now-1), now, min(32, now+1)]), bool(rng.randrange(2))]
''',
    'path': '''"""Normalize absolute root-clamped token paths; a,b,c,.,..; length 0..8."""
import posixpath
def expected(c):
    return posixpath.normpath('/' + '/'.join(c))
def example_generate(rng):
    return [rng.choice(['a', 'b', 'c', '.', '..']) for _ in range(rng.randrange(9))]
def generate(rng):
    return example_generate(rng)
''',
    'replay': '''"""First event per ID contributes signed value. IDs 0..8; values -20..20; length 0..10."""
def expected(c):
    return sum(dict(reversed(c)).values())
def example_generate(rng):
    return [[rng.randrange(9), rng.randrange(-20,21)] for _ in range(rng.randrange(11))]
def generate(rng):
    return example_generate(rng)
''',
    'accounting': '''"""Accept charge iff within both balance and limit (inclusive); all integers 0..32."""
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
''',
    'schema': '''"""Three legal transitions; every other state/event pair is an identity transition."""
STATES = ['new', 'active', 'closed']
EVENTS = ['open', 'close', 'reset']
# Independent exhaustive state table (rows are states; columns are events).
TABLE = [['active','new','new'], ['active','closed','active'], ['closed','closed','new']]
def expected(c):
    return TABLE[STATES.index(c[0])][EVENTS.index(c[1])]
def example_generate(rng):
    return [rng.choice(STATES), rng.choice(EVENTS)]
def generate(rng):
    return example_generate(rng)
''',
    'clamp': '''"""Clamping returns median(value,lower,upper), ordered bounds; integers -32..32."""
def expected(c):
    return sorted(c)[1]
def example_generate(rng):
    lower, upper = sorted([rng.randrange(-32,33), rng.randrange(-32,33)])
    return [rng.randrange(-32,33), lower, upper]
def generate(rng):
    value, lower, upper = example_generate(rng)
    return [rng.choice([value, lower, upper]), lower, upper]
''',
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'references.py').write_text('"""Reviewed reference implementations."""\n\n' + '\n'.join(REFERENCES.values()))
    heldout = ['"""Scored semantic mutants: keep separate from development faults."""\n']
    development = ['"""Development faults use only the constant-return operator."""\n']
    manifest = {'version': 1, 'components': []}
    for name, source in REFERENCES.items():
        claim = CLAIMS[name] + '\ndef check(case, actual):\n    return actual == expected(case)\n'
        (ROOT / f'claim_{name}.py').write_text(claim)
        development.append(f'def {name}_constant(c):\n    return None\n')
        row = {'id': name, 'reference': f'references.py:{name}',
               'claim': f'claim_{name}.py', 'mutants': []}
        for operator, old, new, witness in MUTATIONS[name]:
            assert old in source
            target = f'{name}_{operator}'
            heldout.append(source.replace(f'def {name}(', f'def {target}(').replace(old, new))
            row['mutants'].append({'id': target, 'operator': operator,
                                  'target': f'heldout.py:{target}', 'split': 'heldout',
                                  'witness': witness})
        row['mutants'].append({'id': f'{name}_constant', 'operator': 'constant_return',
                              'target': f'development.py:{name}_constant',
                              'split': 'development', 'witness': MUTATIONS[name][0][3]})
        manifest['components'].append(row)
    (ROOT / 'heldout.py').write_text('\n'.join(heldout))
    (ROOT / 'development.py').write_text('\n'.join(development))
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
