"""Materialize source-level faults, never transform target inputs.

Recipes were authored only after pool-seal.json. Do not change pools in response
to curation. Every old/new edit must change source and match exactly once.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def recipes():
    d = {}
    def add(name, old, news, operator='boundary_shift'):
        d.setdefault(name, []).extend((old, new, operator) for new in news)
    add('lease', 'deadline <= now', ['deadline < now', 'deadline + 2 <= now', 'deadline - 1 <= now', 'deadline - 2 <= now'])
    add('renew', 'now >= expiry', ['now > expiry', 'now >= expiry - 1'])
    add('renew', 'candidate > expiry', ['candidate > expiry + 1', 'candidate > expiry + 2'])
    add('release', 'depth > 0', ['depth > 1', 'depth >= 0'])
    add('release', 'return depth - 1', ['return 1 - depth'], 'wrong_branch_value')
    add('release', 'owner == requester and depth > 0', ['owner == requester and depth > 0 and depth != 31'], 'branch_deletion')
    add('fence', 'proposed > current', [f'proposed > current + {k}' for k in range(1, 5)])
    add('path', 'if stack:', ['if len(stack) > 1:', 'if stack and len(stack) != 2:'])
    add('path', 'stack.pop()', ['stack.pop(0)'], 'wrong_state_update')
    add('path', 'stack.pop()', ['pass'], 'branch_deletion')
    add('relative', "token == '..' and out and out[-1] != '..'", ["token == '..' and len(out) > 1 and out[-1] != '..'", "token == '..' and out and out[-1] not in {'..', 'a'}"])
    add('relative', 'out.pop()', ['out.pop(0)'], 'wrong_state_update')
    add('relative', "if token == '.':\n            continue", "if token == '.':\n            out.append(token)".split('\0'), 'wrong_state_update')
    add('extension', "c[-1].split('.')", ["c[-1].split('.', 1)"], 'wrong_state_update')
    add('extension', "tokens[0] == ''", ["tokens[0] in ('', 'a')"], 'branch_deletion')
    add('extension', 'return tokens[-1]', ['return tokens[-1].lower()'], 'wrong_branch_value')
    add('extension', 'len(tokens) == 2', ['len(tokens) >= 2'])
    add('path_key', "if token != '.':", [f"if token not in {{'.', '{key}'}}:" for key in 'abcd'], 'branch_deletion')
    add('replay', 'seen.add(key)', [f'\n            if key != {key}:\n                seen.add(key)' for key in range(4)], 'wrong_state_update')
    add('last_replay', 'state[key] = value', [f'\n        if key != {key} or key not in state:\n            state[key] = value' for key in range(4)], 'wrong_state_update')
    add('revision', 'if incoming > current:', [f'if incoming > current and incoming != {k}:' for k in [15, 14, 13, 12]], 'branch_deletion')
    add('tombstone', 'live.discard(key)', [f'\n            if key != {key}:\n                live.discard(key)' for key in range(4)], 'wrong_state_update')
    add('accounting', 'charge > balance', ['charge >= balance', 'charge > balance + 1'])
    add('accounting', 'charge > limit', ['charge >= limit', 'charge > limit + 1'])
    add('clamp', 'value < lower', ['value < lower - 1', 'value < lower - 2'])
    add('clamp', 'value > upper', ['value > upper + 1', 'value > upper + 2'])
    add('reserve', 'used > capacity', ['used >= capacity', 'used > capacity + 1', 'used > capacity + 2', 'used > capacity - 2'])
    add('refund', 'result > capacity', [f'result > capacity + {k}' for k in range(1, 5)])
    add('schema', 'version != 0', ['version > 1'])
    for old, new in [("('new', 'open'): 'active'", "('new', 'open'): 'closed'"),
                     ("('active', 'close'): 'closed'", "('active', 'close'): 'new'"),
                     ("('closed', 'reset'): 'new'", "('closed', 'reset'): 'active'")]:
        add('schema', old, [new], 'wrong_branch_value')
    add('migration', 'version == 0', ['version <= 1', 'version <= 2'])
    add('migration', "data['a'] = data.pop('A')", ["data['b'] = data.pop('A')"], 'wrong_state_update')
    add('migration', "'A' in data", ["'A' in data and 'a' not in data"], 'branch_deletion')
    add('required', 'if key not in c:', [f"if key not in c or (key == '{k}' and c.count(key) > 1):" for k in 'ABC'], 'wrong_branch_value')
    add('required', 'if key not in c:', ["if key not in c or (key == 'A' and c.count(key) > 2):"], 'wrong_branch_value')
    for k in range(4):
        add('enum_map', f'out.append({k})', [f'out.append({(k+1)%4})'], 'wrong_branch_value')
    # Genuine semantic equivalents: changed predicates execute a different branch
    # at equality, but both branches yield the identical specified value.
    equivalents = {
        'fence': ('proposed > current', 'proposed >= current', 'boundary_shift'),
        'renew': ('candidate > expiry', 'candidate >= expiry', 'boundary_shift'),
        'clamp': ('value < lower', 'value <= lower', 'boundary_shift'),
        'refund': ('result > capacity', 'result >= capacity', 'boundary_shift'),
        'revision': ('incoming > current', 'incoming >= current', 'boundary_shift'),
    }
    for name, recipe in equivalents.items():
        d[name].append(recipe)
    return d


def main():
    if (ROOT/'mutants.json').exists():
        raise SystemExit('Refuse to overwrite catalog')
    seal = json.loads((ROOT/'pool-seal.json').read_text())
    for n, digest in seal['sha256'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == digest, n
    source = (ROOT/'components.py').read_text()
    parsed = ast.parse(source)
    sources = {node.name: ast.get_source_segment(source, node)+'\n' for node in parsed.body if isinstance(node, ast.FunctionDef)}
    catalog = []
    output = ROOT/'mutants'
    output.mkdir(exist_ok=True)
    for name, edits in recipes().items():
        for index, (old, new, op) in enumerate(edits):
            s = sources[name]
            assert old != new and s.count(old) == 1, (name, old)
            changed = s.replace(old, new)
            assert ast.dump(ast.parse(s)) != ast.dump(ast.parse(changed))
            filename = f'mutants/{name}_{index:02d}.py'
            (ROOT/filename).write_text(changed)
            catalog.append(dict(id=f'{name}_{index:02d}', component=name,
                                operator=op, split='heldout', file=filename,
                                old=old, new=new))
    (ROOT/'mutants.json').write_text(json.dumps(catalog, indent=2)+'\n')

if __name__ == '__main__':
    main()
