def renew(c):
    now, expiry, duration = c
    if now >= expiry:
        return expiry
    candidate = now + duration
    if candidate > expiry + 1:
        return candidate
    return expiry
