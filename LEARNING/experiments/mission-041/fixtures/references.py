"""Reviewed reference implementations."""

def lease(c):
    now, deadline, same_owner = c
    return same_owner or deadline <= now

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

def replay(c):
    seen = set()
    total = 0
    for key, value in c:
        if key not in seen:
            total += value
            seen.add(key)
    return total

def accounting(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge > limit:
        return balance
    return balance - charge

def schema(c):
    state, event = c
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed',
             ('closed', 'reset'): 'new'}
    return edges.get((state, event), state)

def clamp(c):
    value, lower, upper = c
    return min(upper, max(lower, value))
