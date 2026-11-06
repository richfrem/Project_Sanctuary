"""Independent finite specifications; no candidate or mutant imports.

The six v1 contracts retain their domains. New contracts have finite enumerated
state models. This module is frozen before curation and measured searches.
"""
import itertools
import posixpath
import random

NAMES = ['lease', 'renew', 'release', 'fence', 'path', 'relative', 'extension',
         'path_key', 'replay', 'last_replay', 'revision', 'tombstone',
         'accounting', 'clamp', 'reserve', 'refund', 'schema', 'migration',
         'required', 'enum_map']
FAMILIES = {n: f for f, names in zip(
    ['lease', 'path', 'replay', 'accounting', 'schema'],
    [NAMES[0:4], NAMES[4:8], NAMES[8:12], NAMES[12:16], NAMES[16:20]]) for n in names}
OLD = {'lease', 'path', 'replay', 'accounting', 'schema', 'clamp'}
STATES = ['new', 'active', 'closed']
EVENTS = ['open', 'close', 'reset']
TABLE = [['active', 'new', 'new'], ['active', 'closed', 'active'], ['closed', 'closed', 'new']]
CONTRACTS = {
    'lease': 'Acquire iff same owner or deadline <= now. Integer clocks 0..32, Boolean same owner.',
    'renew': 'At now strictly before expiry, same-owner renewal returns max(expiry, now+duration); otherwise expiry. now/expiry/duration 0..8, Boolean owner.',
    'release': 'Return free iff already free OR requester matches owner; owner/requester in A,a,B; Boolean free.',
    'fence': 'Accept strictly greater proposed fencing token; return max(current, proposed). Tokens 0..8.',
    'path': 'Absolute root-clamped POSIX path from a,b,c,.,.. tokens, length 0..8.',
    'relative': 'Relative POSIX normalization: preserve unmatched parents. Tokens A,a,.,..; length 0..4.',
    'extension': 'Last dot suffix of final basename, excluding leading dot and absent dot; case preserved. Enumerated basename alphabet.',
    'path_key': 'Slash-joined case-sensitive tokens, dropping dot tokens only; A,a,B,.; length 0..4.',
    'replay': 'First event per ID contributes value, IDs 0..8, values -20..20, length 0..10.',
    'last_replay': 'Latest value per ID; return sorted key/value pairs. IDs A,a; values -1,0,1; length 0..4.',
    'revision': 'Accept only events with strictly rising positive revision; return last accepted revision. Revisions 0..4, length 0..4.',
    'tombstone': 'Sequential put/delete events update live-ID set; return sorted live IDs. IDs A,a; put/delete; length 0..4.',
    'accounting': 'Charge iff charge <= both balance and limit, inclusive. All integers 0..32.',
    'clamp': 'Return median(value, lower, upper); ordered lower <= upper; each integer -32..32.',
    'reserve': 'Reserve sum of requests iff sum <= capacity, otherwise capacity unchanged. Capacity 0..8; requests 0..4, length 0..4.',
    'refund': 'Return min(capacity, balance+refund). balance 0..capacity, capacity/refund 0..8.',
    'schema': 'State table: new/open->active; active/close->closed; closed/reset->new; others identity. Enumerated 3x3.',
    'migration': 'Version 0 renames A to a if A exists, dropping A; version 1 preserves fields. keys A,a,B subset, values 1. Output sorted keys.',
    'required': 'Return sorted missing fields from required A,a,B, ignoring duplicate provided fields. Provided A,a,B,X sequences length 0..4.',
    'enum_map': 'Case-sensitive enum conversion: A->0, a->1, B->2, b->3; unknown->-1. Map sequence length 0..4.'}


def expected(name, c):
    """Oracles expressed separately from imperative reference implementations."""
    if name == 'lease':
        return any([c[2], c[0] >= c[1]])
    if name == 'renew':
        now, expiry, duration, owner = c
        return sorted([expiry, now + duration])[-1] if owner and now in range(expiry) else expiry
    if name == 'release':
        owner, requester, free = c
        return free or len({owner, requester}) == 1
    if name == 'fence':
        return sorted(c)[-1]
    if name == 'path':
        return posixpath.normpath('/' + '/'.join(c))
    if name == 'relative':
        return posixpath.normpath('/'.join(c))
    if name == 'extension':
        basename = c[-1] if c else ''
        dot = basename.rfind('.')
        return basename[dot+1:] if 0 < dot < len(basename)-1 else ''
    if name == 'path_key':
        return '/'.join(filter(lambda x: x != '.', c))
    if name == 'replay':
        return sum(dict(reversed(c)).values())
    if name == 'last_replay':
        return sorted([key, value] for key, value in dict(c).items())
    if name == 'revision':
        return max([0] + c)
    if name == 'tombstone':
        final = dict(c)
        return sorted(k for k, operation in final.items() if operation == 'put')
    if name == 'accounting':
        balance, charge, limit = c
        return balance - charge * int(charge in range(min(balance, limit)+1))
    if name == 'clamp':
        return sorted(c)[1]
    if name == 'reserve':
        capacity, requests = c
        return capacity - sum(requests) * int(sum(requests) in range(capacity+1))
    if name == 'refund':
        balance, amount, capacity = c
        return sorted([balance+amount, capacity])[0]
    if name == 'schema':
        return TABLE[STATES.index(c[0])][EVENTS.index(c[1])]
    if name == 'migration':
        version, keys = c
        return sorted({'a' if k == 'A' and version == 0 else k for k in keys})
    if name == 'required':
        return sorted(set(['A','a','B']).difference(c))
    if name == 'enum_map':
        return [dict(zip(['A','a','B','b'], range(4))).get(x, -1) for x in c]
    raise KeyError(name)


def sequences(alphabet, maximum):
    for n in range(maximum+1):
        for case in itertools.product(alphabet, repeat=n):
            yield [list(x) if isinstance(x, tuple) else x for x in case]


def domain(name):
    """Exhaust exact finite domains, except explicit reduced+sampled v1 replay."""
    if name == 'lease':
        yield from (list(x) for x in itertools.product(range(33), range(33), [False,True]))
    elif name == 'renew':
        yield from (list(x) for x in itertools.product(range(9), range(9), range(9), [False,True]))
    elif name == 'release':
        yield from (list(x) for x in itertools.product(['A','a','B'], ['A','a','B'], [False,True]))
    elif name == 'fence':
        yield from (list(x) for x in itertools.product(range(9), repeat=2))
    elif name == 'path':
        yield from sequences(['a','b','c','.','..'], 8)
    elif name == 'relative':
        yield from sequences(['A','a','.','..'], 4)
    elif name == 'extension':
        yield from sequences(['A.py','a.PY','.env','A','a.','a.b.c'], 3)
    elif name == 'path_key':
        yield from sequences(['A','a','B','.'], 4)
    elif name == 'replay':
        yield from sequences(list(itertools.product([0,1], [-1,0,1])), 6)
        rng = random.Random(41000)
        for _ in range(10000):
            yield example_generate(name, rng)
    elif name == 'last_replay':
        yield from sequences(list(itertools.product(['A','a'], [-1,0,1])), 4)
    elif name == 'revision':
        yield from sequences(range(5), 4)
    elif name == 'tombstone':
        yield from sequences(list(itertools.product(['A','a'], ['put','delete'])), 4)
    elif name == 'accounting':
        yield from (list(x) for x in itertools.product(range(33), repeat=3))
    elif name == 'clamp':
        yield from ([v,l,u] for v in range(-32,33) for l in range(-32,33) for u in range(l,33))
    elif name == 'reserve':
        yield from ([b, seq] for b in range(9) for seq in sequences(range(5),4))
    elif name == 'refund':
        yield from ([b,a,cap] for cap in range(9) for b in range(cap+1) for a in range(9))
    elif name == 'schema':
        yield from (list(x) for x in itertools.product(STATES, EVENTS))
    elif name == 'migration':
        yield from ([v, list(keys)] for v in range(2) for n in range(4)
                    for keys in itertools.combinations(['A','a','B'], n))
    elif name == 'required':
        yield from sequences(['A','a','B','X'],4)
    elif name == 'enum_map':
        yield from sequences(['A','a','B','b','X'],4)


def example_generate(name, r):
    """Ordinary independent randomized examples, no mutant/curation access."""
    if name in ['lease', 'accounting']:
        return [r.randrange(33), r.randrange(33), bool(r.randrange(2))] if name == 'lease' else [r.randrange(33) for _ in range(3)]
    if name == 'renew':
        return [r.randrange(9), r.randrange(9), r.randrange(9), bool(r.randrange(2))]
    if name == 'release':
        return [r.choice(['A','a','B']), r.choice(['A','a','B']), bool(r.randrange(2))]
    if name == 'fence':
        return [r.randrange(9), r.randrange(9)]
    alphabets = {'path':(['a','b','c','.','..'],8), 'relative':(['A','a','.','..'],4),
        'extension':(['A.py','a.PY','.env','A','a.','a.b.c'],3), 'path_key':(['A','a','B','.'],4),
        'revision':(list(range(5)),4), 'required':(['A','a','B','X'],4), 'enum_map':(['A','a','B','b','X'],4)}
    if name in alphabets:
        alphabet, length = alphabets[name]
        return [r.choice(alphabet) for _ in range(r.randrange(length+1))]
    if name == 'replay':
        return [[r.randrange(9), r.randrange(-20,21)] for _ in range(r.randrange(11))]
    if name == 'last_replay':
        return [[r.choice(['A','a']),r.choice([-1,0,1])] for _ in range(r.randrange(5))]
    if name == 'tombstone':
        return [[r.choice(['A','a']),r.choice(['put','delete'])] for _ in range(r.randrange(5))]
    if name == 'clamp':
        lower, upper = sorted([r.randrange(-32,33),r.randrange(-32,33)])
        return [r.randrange(-32,33),lower,upper]
    if name == 'reserve':
        return [r.randrange(9),[r.randrange(5) for _ in range(r.randrange(5))]]
    if name == 'refund':
        capacity = r.randrange(9)
        return [r.randrange(capacity+1),r.randrange(9),capacity]
    if name == 'schema':
        return [r.choice(STATES),r.choice(EVENTS)]
    if name == 'migration':
        return [r.randrange(2),[k for k in ['A','a','B'] if r.randrange(2)]]
    raise KeyError(name)


def property_generate(name, r):
    """Twenty claim-specific constructors; never call example_generate."""
    if name == 'lease':
        now = r.randrange(33)
        return [now, r.choice([max(0,now-1),now,min(32,now+1)]), bool(r.randrange(2))]
    if name == 'renew':
        now = r.randrange(9)
        return [now,r.choice([now,max(0,now-1),min(8,now+1)]),r.choice([0,1,8]),r.choice([True,False])]
    if name == 'release':
        owner = r.choice(['A','a','B'])
        return [owner,r.choice([owner,'A','a']),r.choice([False,False,True])]
    if name == 'fence':
        current = r.randrange(9)
        return [current,r.choice([current,max(0,current-1),min(8,current+1)])]
    if name == 'path':
        prefix = [r.choice(['a','b','c']) for _ in range(r.randrange(4))]
        return r.choice([['..']+prefix, prefix+['.'],prefix+['..'],prefix+['..','..','a']])
    if name == 'relative':
        return r.choice([['..','..','A'],['A','..','a'],['.','a'],[],['A','.','a','..']])
    if name == 'extension':
        return r.choice([['A.py'],['a.PY'],['.env'],['A'],['a.'],['a.b.c'],['A','a.PY']])
    if name == 'path_key':
        return r.choice([['A','a'],['a','A'],['.','A','.'],[],['B','a','A','.']])
    if name == 'replay':
        key = r.randrange(9)
        value = r.randrange(-20,21)
        return r.choice([[[key,value],[key,-value]],[[0,0],[key,value],[key,1]],[[key,value]]])
    if name == 'last_replay':
        return r.choice([[['A',1],['A',-1]],[['A',1],['a',-1]],[],[['a',0],['A',1],['a',-1]]])
    if name == 'revision':
        return r.choice([[0],[4,1,4],[1,2,3,4],[4,3,2,1],[],[0,0,0,0]])
    if name == 'tombstone':
        return r.choice([[['A','put'],['A','delete']],[['A','put'],['a','delete']],
            [['A','delete'],['A','put']],[],[['a','put'],['A','put'],['a','delete']]])
    if name == 'accounting':
        b = r.randrange(33)
        q = r.choice([0,b,min(32,b+1)])
        return [b,q,r.choice([q,max(0,q-1),32])]
    if name == 'clamp':
        lower = r.randrange(-32,33)
        upper = r.randrange(lower,33)
        return [r.choice([lower,upper,max(-32,lower-1),min(32,upper+1)]),lower,upper]
    if name == 'reserve':
        b = r.randrange(9)
        return [b,r.choice([[0,0,0,0],[4,4],[min(4,b)],[4,4,4,4],[]])]
    if name == 'refund':
        capacity = r.randrange(9)
        balance = r.choice([0,capacity])
        return [balance,r.choice([0,capacity,8]),capacity]
    if name == 'schema':
        return r.choice([['new','open'],['active','close'],['closed','reset'],
                         ['new','close'],['active','open'],['closed','close']])
    if name == 'migration':
        return [r.choice([0,1]),r.choice([[],['A'],['a'],['A','a'],['A','a','B']])]
    if name == 'required':
        return r.choice([[],['A','a','B'],['A','A'],['a','B','X'],['X','X','X','X']])
    if name == 'enum_map':
        return r.choice([[],['A','a','B','b'],['X'],['a','A'],['b','B','X']])
    raise KeyError(name)
