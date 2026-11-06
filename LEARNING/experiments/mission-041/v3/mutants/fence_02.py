def fence(c):
    current, proposed = c
    if proposed > current + 3:
        return proposed
    return current
