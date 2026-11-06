def schema(c):
    state, event, version = c
    if version != 0:
        return state
    edges = {('new', 'open'): 'closed', ('active', 'close'): 'closed', ('closed', 'reset'): 'new'}
    return edges.get((state, event), state)
