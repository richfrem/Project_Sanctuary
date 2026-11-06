def fence(c):
    current, proposed = c
    if proposed > current + 2:
        return proposed
    return current
