def revision(c):
    current = 0
    for incoming in c:
        if incoming >= current:
            current = incoming
    return current
