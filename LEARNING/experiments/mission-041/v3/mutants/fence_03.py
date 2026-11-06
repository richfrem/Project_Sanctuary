def fence(c):
    current, proposed = c
    if proposed > current + 4:
        return proposed
    return current
