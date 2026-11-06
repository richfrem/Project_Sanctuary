def release(c):
    owner, requester, depth = c
    if owner == requester and depth > 0 and depth != 31:
        return depth - 1
    return depth
