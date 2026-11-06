def path_key(c):
    out = []
    for token in c:
        if token not in {'.', 'd'}:
            out.append(token)
    return '/'.join(out)
