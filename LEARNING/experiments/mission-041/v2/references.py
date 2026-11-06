"""Imperative candidates, independent of specifications.py and its tables."""
def lease(c):
    now, deadline, same_owner = c
    return same_owner or deadline <= now


def renew(c):
    now, expiry, duration, owner = c
    if not owner or now >= expiry:
        return expiry
    candidate = now + duration
    if candidate > expiry:
        return candidate
    return expiry


def release(c):
    owner, requester, free = c
    if free:
        return True
    return owner == requester


def fence(c):
    current, proposed = c
    if proposed > current:
        return proposed
    return current


def path(c):
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


def relative(c):
    out = []
    for token in c:
        if token == '.':
            continue
        if token == '..' and out and out[-1] != '..':
            out.pop()
        else:
            out.append(token)
    return '/'.join(out) or '.'


def extension(c):
    if not c:
        return ''
    tokens = c[-1].split('.')
    if len(tokens) < 2 or (len(tokens) == 2 and tokens[0] == ''):
        return ''
    return tokens[-1]


def path_key(c):
    out = []
    for token in c:
        if token != '.':
            out.append(token)
    return '/'.join(out)


def replay(c):
    seen = set()
    total = 0
    for key, value in c:
        if key not in seen:
            total += value
            seen.add(key)
    return total


def last_replay(c):
    state = {}
    for key, value in c:
        state[key] = value
    return [[key,state[key]] for key in sorted(state)]


def revision(c):
    current = 0
    for incoming in c:
        if incoming > current:
            current = incoming
    return current


def tombstone(c):
    live = set()
    for key, operation in c:
        if operation == 'put':
            live.add(key)
        else:
            live.discard(key)
    return sorted(live)


def accounting(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge > limit:
        return balance
    return balance - charge


def clamp(c):
    value, lower, upper = c
    return min(upper, max(lower, value))


def reserve(c):
    capacity, requests = c
    used = 0
    for amount in requests:
        used += amount
    if used > capacity:
        return capacity
    return capacity - used


def refund(c):
    balance, amount, capacity = c
    result = balance + amount
    if result > capacity:
        result = capacity
    return result


def schema(c):
    state, event = c
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed', ('closed', 'reset'): 'new'}
    return edges.get((state, event), state)


def migration(c):
    version, keys = c
    data = {k:1 for k in keys}
    if version == 0 and 'A' in data:
        data['a'] = data.pop('A')
    return sorted(data)


def required(c):
    missing = []
    for key in ['A','a','B']:
        if key not in c:
            missing.append(key)
    return sorted(missing)


def enum_map(c):
    out = []
    for token in c:
        if token == 'A':
            out.append(0)
        elif token == 'a':
            out.append(1)
        elif token == 'B':
            out.append(2)
        elif token == 'b':
            out.append(3)
        else:
            out.append(-1)
    return out
