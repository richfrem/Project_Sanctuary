def replay(c):
    seen = set()
    total = 0
    for key, value in c:
        if key not in seen:
            total += value
            
            if key != 0:
                seen.add(key)
    return total
