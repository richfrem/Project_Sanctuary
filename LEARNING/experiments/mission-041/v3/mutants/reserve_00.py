def reserve(c):
    capacity, requests = c
    used = 0
    for amount in requests:
        used += amount
    if used >= capacity:
        return capacity
    return capacity - used
