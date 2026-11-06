def renew(c):
    now, expiry, duration = c
    if now > expiry:
        return expiry
    candidate = now + duration
    if candidate > expiry:
        return candidate
    return expiry
