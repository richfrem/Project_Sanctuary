"""Normative finite contracts, independent of candidate implementations.

V3 deliberately enlarges the domains. These bounds define the entire benchmark,
not a sample of an unbounded production API. No candidate/mutation imports.
"""
import itertools
import posixpath

NAMES = ['lease', 'renew', 'release', 'fence', 'path', 'relative', 'extension',
         'path_key', 'replay', 'last_replay', 'revision', 'tombstone',
         'accounting', 'clamp', 'reserve', 'refund', 'schema', 'migration',
         'required', 'enum_map']
TOKENS = ['.', '..'] + list('abcdefghijklmn')
KEYS = list('ABCDEFGH')
BASENAMES = ['a.py', 'b.PY', '.env', 'a.', 'a.b.c', 'file', 'a.txt', '.a.b',
             'b.md', 'c.json', 'd.csv', 'e.xml', 'f.log', 'g.ini', 'h.cfg', 'i.sh']
ENUMS = list('AaBbCDEFGHIJKLMN')
CONTRACTS = {
 'lease': 'Acquire iff same owner or expiry <= now; clocks 0..127, owner Boolean.',
 'renew': 'Authorized renewal before expiry returns max(expiry, now+duration); otherwise expiry; 0..31 each.',
 'release': 'Reentrant release: matching owner decrements positive depth, others preserve it. Owner/requester/depth 0..31.',
 'fence': 'Return max(current, proposed), fencing tokens 0..255.',
 'path': 'Root-clamped absolute POSIX normalization; TOKENS sequences length 0..4.',
 'relative': 'Relative POSIX normalization, unmatched parents preserved; TOKENS length 0..4.',
 'extension': 'Suffix of last BASENAMES entry, excluding a sole leading dot; length 0..4.',
 'path_key': 'Case-sensitive slash key dropping dot tokens only; TOKENS length 0..4.',
 'replay': 'First event per ID contributes value; IDs 0..3, values -1,0,1; length 0..4.',
 'last_replay': 'Last event per ID wins; same event domain as replay; sorted key/value pairs.',
 'revision': 'Maximum of initial zero and incoming revisions 0..15; length 0..4.',
 'tombstone': 'Last put/delete per ID determines membership; IDs 0..7, ops put/delete; length 0..4.',
 'accounting': 'Charge iff charge <= balance and limit; else preserve balance; all 0..31.',
 'clamp': 'Median(value, lower, upper), with ordered bounds; integers -32..32.',
 'reserve': 'Subtract sum(requests) iff affordable, otherwise preserve capacity; capacity 0..31, requests 0..7 length 0..3.',
 'refund': 'min(capacity, balance+amount); capacity 0..31, balance 0..capacity, amount 0..31.',
 'schema': 'new/open->active, active/close->closed, closed/reset->new only at supported version 0; other state/event/version preserve state; version 0..255.',
 'migration': 'At version 0 rename A to a (set of keys, values all 1), other versions identity; versions 0..31, subsets of KEYS plus a.',
 'required': 'Sorted missing required fields A,B,C; ignore duplicates and unknown keys; ENUMS sequences length 0..4.',
 'enum_map': 'Map A,a,B,b to 0,1,2,3 respectively, all others -1; ENUMS length 0..4.'}


def sequences(alphabet, maximum):
    for length in range(maximum + 1):
        yield from (list(c) for c in itertools.product(alphabet, repeat=length))


def domain(name):
    if name == 'lease':
        yield from (list(c) for c in itertools.product(range(128), range(128), [False, True]))
    elif name in {'renew', 'release', 'accounting'}:
        yield from (list(c) for c in itertools.product(range(32), repeat=3))
    elif name == 'fence':
        yield from (list(c) for c in itertools.product(range(256), repeat=2))
    elif name in {'path', 'relative', 'path_key'}:
        yield from sequences(TOKENS, 4)
    elif name == 'extension':
        yield from sequences(BASENAMES, 4)
    elif name in {'replay', 'last_replay'}:
        yield from sequences(list(itertools.product(range(4), [-1, 0, 1])), 4)
    elif name == 'revision':
        yield from sequences(range(16), 4)
    elif name == 'tombstone':
        yield from sequences(list(itertools.product(range(8), ['put', 'delete'])), 4)
    elif name == 'clamp':
        yield from ([v, lo, hi] for v in range(-32, 33) for lo in range(-32, 33) for hi in range(lo, 33))
    elif name == 'reserve':
        yield from ([cap, req] for cap in range(32) for req in sequences(range(8), 3))
    elif name == 'refund':
        yield from ([b, a, cap] for cap in range(32) for b in range(cap+1) for a in range(32))
    elif name == 'schema':
        yield from (list(c) for c in itertools.product(['new', 'active', 'closed'], ['open', 'close', 'reset'], range(256)))
    elif name == 'migration':
        alphabet = KEYS + ['a']
        yield from ([v, [k for i, k in enumerate(alphabet) if mask & (1 << i)]] for v in range(32) for mask in range(512))
    elif name in {'required', 'enum_map'}:
        yield from sequences(ENUMS, 4)
    else:
        raise KeyError(name)


def expected(name, c):
    if name == 'lease':
        return any([c[2], c[0] >= c[1]])
    if name == 'renew':
        return max(c[1], c[0]+c[2]) if c[0] in range(c[1]) else c[1]
    if name == 'release':
        return c[2] - int(c[0] == c[1] and c[2] > 0)
    if name == 'fence':
        return sorted(c)[-1]
    if name == 'path':
        return posixpath.normpath('/' + '/'.join(c))
    if name == 'relative':
        return posixpath.normpath('/'.join(c))
    if name == 'extension':
        b = c[-1] if c else ''
        p = b.rfind('.')
        return b[p+1:] if p > 0 else ''
    if name == 'path_key':
        return '/'.join(x for x in c if x != '.')
    if name == 'replay':
        return sum(dict(reversed(c)).values())
    if name == 'last_replay':
        return sorted([k, v] for k, v in dict(c).items())
    if name == 'revision':
        return max([0] + c)
    if name == 'tombstone':
        return sorted(k for k, op in dict(c).items() if op == 'put')
    if name == 'accounting':
        b, a, limit = c
        return b - a * int(a in range(min(b, limit)+1))
    if name == 'clamp':
        return sorted(c)[1]
    if name == 'reserve':
        cap, req = c
        return cap - sum(req) * int(sum(req) in range(cap+1))
    if name == 'refund':
        return min(c[2], c[0]+c[1])
    if name == 'schema':
        return {('new', 'open', 0): 'active', ('active', 'close', 0): 'closed',
                ('closed', 'reset', 0): 'new'}.get(tuple(c), c[0])
    if name == 'migration':
        return sorted({'a' if k == 'A' and c[0] == 0 else k for k in c[1]})
    if name == 'required':
        return sorted(set('ABC') - set(c))
    if name == 'enum_map':
        return [dict(zip('AaBb', range(4))).get(x, -1) for x in c]
    raise KeyError(name)
