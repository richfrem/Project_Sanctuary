def last_replay(c):
    state = {}
    for key, value in c:
        
        if key != 0 or key not in state:
            state[key] = value
    return [[key,state[key]] for key in sorted(state)]
