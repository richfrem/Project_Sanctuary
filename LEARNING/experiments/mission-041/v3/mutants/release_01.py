def release(c):
    owner, requester, depth = c
    if owner == requester and depth >= 0:
        return depth - 1
    return depth
