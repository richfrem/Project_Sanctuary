def clamp(c):
    value, lower, upper = c
    if value <= lower:
        return lower
    if value > upper:
        return upper
    return value
