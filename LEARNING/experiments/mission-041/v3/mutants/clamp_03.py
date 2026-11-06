def clamp(c):
    value, lower, upper = c
    if value < lower:
        return lower
    if value > upper + 2:
        return upper
    return value
