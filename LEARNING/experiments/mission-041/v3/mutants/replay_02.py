def replay(c):
    seen = set()
    total = 0
    for key, value in c:
        if key not in seen:
            total += value
            
            if key != 2:
                seen.add(key)
    return total
