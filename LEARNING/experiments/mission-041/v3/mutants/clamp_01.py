def clamp(c):
    value, lower, upper = c
    if value < lower - 2:
        return lower
    if value > upper:
        return upper
    return value
