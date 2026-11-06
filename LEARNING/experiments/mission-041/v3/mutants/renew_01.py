def renew(c):
    now, expiry, duration = c
    if now >= expiry - 1:
        return expiry
    candidate = now + duration
    if candidate > expiry:
        return candidate
    return expiry
