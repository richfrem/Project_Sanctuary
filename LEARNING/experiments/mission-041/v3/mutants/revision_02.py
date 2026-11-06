def revision(c):
    current = 0
    for incoming in c:
        if incoming > current and incoming != 13:
            current = incoming
    return current
