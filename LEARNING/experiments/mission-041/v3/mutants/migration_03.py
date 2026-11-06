def migration(c):
    version, keys = c
    data = {k:1 for k in keys}
    if version == 0 and 'A' in data and 'a' not in data:
        data['a'] = data.pop('A')
    return sorted(data)
