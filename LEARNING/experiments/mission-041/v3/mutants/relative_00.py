def relative(c):
    out = []
    for token in c:
        if token == '.':
            continue
        if token == '..' and len(out) > 1 and out[-1] != '..':
            out.pop()
        else:
            out.append(token)
    return '/'.join(out) or '.'
