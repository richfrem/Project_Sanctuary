def lease(c):
    now, deadline, same_owner = c
    return same_owner or deadline - 2 <= now
