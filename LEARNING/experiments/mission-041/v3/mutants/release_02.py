def release(c):
    owner, requester, depth = c
    if owner == requester and depth > 0:
        return 1 - depth
    return depth
