def required(c):
    missing = []
    for key in ['A','B','C']:
        if key not in c or (key == 'A' and c.count(key) > 1):
            missing.append(key)
    return sorted(missing)
