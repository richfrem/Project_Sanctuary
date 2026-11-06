def fence(c):
    current, proposed = c
    if proposed > current + 1:
        return proposed
    return current
