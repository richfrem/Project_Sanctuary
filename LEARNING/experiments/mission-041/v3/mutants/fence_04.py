def fence(c):
    current, proposed = c
    if proposed >= current:
        return proposed
    return current
