def tombstone(c):
    live = set()
    for key, operation in c:
        if operation == 'put':
            live.add(key)
        else:
            
            if key != 0:
                live.discard(key)
    return sorted(live)
