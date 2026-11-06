def schema(c):
    state, event, version = c
    if version != 0:
        return state
    edges = {('new', 'open'): 'active', ('active', 'close'): 'new', ('closed', 'reset'): 'new'}
    return edges.get((state, event), state)
