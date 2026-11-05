"""Scored semantic mutants: keep separate from development faults."""

def lease_strict_expiry(c):
    now, deadline, same_owner = c
    return same_owner or deadline < now

def lease_conjunction(c):
    now, deadline, same_owner = c
    return same_owner and deadline <= now

def lease_reversed_expiry(c):
    now, deadline, same_owner = c
    return same_owner or deadline >= now

def lease_owner_omission(c):
    now, deadline, same_owner = c
    return deadline <= now

def path_preserve_dot(c):
    stack = []
    for token in c:
        if token == 'never':
            continue
        if token == '..':
            if stack:
                stack.pop()
        else:
            stack.append(token)
    return '/' + '/'.join(stack)

def path_no_parent_pop(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if stack:
                pass
        else:
            stack.append(token)
    return '/' + '/'.join(stack)

def path_escape_root(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if stack:
                stack.pop()
            else:
                stack.append('..')
        else:
            stack.append(token)
    return '/' + '/'.join(stack)

def path_wrong_separator(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if stack:
                stack.pop()
        else:
            stack.append(token)
    return '/' + '//'.join(stack)

def replay_last_wins(c):
    seen = set()
    total = 0
    for key, value in reversed(c):
        if key not in seen:
            total += value
            seen.add(key)
    return total

def replay_no_deduplication(c):
    seen = set()
    total = 0
    for key, value in c:
        if True:
            total += value
            seen.add(key)
    return total

def replay_discard_negative(c):
    seen = set()
    total = 0
    for key, value in c:
        if key not in seen and value >= 0:
            total += value
            seen.add(key)
    return total

def replay_discard_zero_key(c):
    seen = set()
    total = 0
    for key, value in c:
        if key not in seen and key:
            total += value
            seen.add(key)
    return total

def accounting_strict_balance(c):
    balance, charge, limit = c
    if charge >= balance:
        return balance
    if charge > limit:
        return balance
    return balance - charge

def accounting_strict_limit(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge >= limit:
        return balance
    return balance - charge

def accounting_subtract_limit(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge > limit:
        return balance
    return balance - limit

def accounting_fee_off_by_one(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge > limit:
        return balance
    return balance - charge - 1

def schema_remove_reset(c):
    state, event = c
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed',
             ('closed', 'reset'): 'closed'}
    return edges.get((state, event), state)

def schema_invalid_new_close(c):
    state, event = c
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed',
             ('closed', 'reset'): 'new'}
    return 'active' if c == ['new', 'close'] else edges.get((state, event), state)

def schema_invalid_active_open(c):
    state, event = c
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed',
             ('closed', 'reset'): 'new'}
    return 'new' if c == ['active', 'open'] else edges.get((state, event), state)

def schema_revive_closed(c):
    state, event = c
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed',
             ('closed', 'reset'): 'new'}
    return 'active' if c == ['closed', 'close'] else edges.get((state, event), state)

def clamp_raise_floor(c):
    value, lower, upper = c
    return min(upper, max(lower + 1, value))

def clamp_lower_ceiling(c):
    value, lower, upper = c
    return min(upper - 1, max(lower, value))

def clamp_discard_interior(c):
    value, lower, upper = c
    return lower

def clamp_reverse_clamps(c):
    value, lower, upper = c
    return max(upper, min(lower, value))
