def clamp(c):
    value, lower, upper = c
    if value < lower:
        return lower
    if value > upper + 1:
        return upper
    return value
