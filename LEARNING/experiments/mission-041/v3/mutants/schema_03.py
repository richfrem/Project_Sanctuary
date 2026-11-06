def schema(c):
    state, event, version = c
    if version != 0:
        return state
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed', ('closed', 'reset'): 'active'}
    return edges.get((state, event), state)
