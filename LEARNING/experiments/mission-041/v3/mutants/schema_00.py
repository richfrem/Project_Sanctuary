def schema(c):
    state, event, version = c
    if version > 1:
        return state
    edges = {('new', 'open'): 'active', ('active', 'close'): 'closed', ('closed', 'reset'): 'new'}
    return edges.get((state, event), state)
