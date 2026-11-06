def reserve(c):
    capacity, requests = c
    used = 0
    for amount in requests:
        used += amount
    if used > capacity + 2:
        return capacity
    return capacity - used
