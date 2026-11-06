def migration(c):
    version, keys = c
    data = {k:1 for k in keys}
    if version <= 1 and 'A' in data:
        data['a'] = data.pop('A')
    return sorted(data)
