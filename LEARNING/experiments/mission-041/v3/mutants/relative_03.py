def relative(c):
    out = []
    for token in c:
        if token == '.':
            out.append(token)
        if token == '..' and out and out[-1] != '..':
            out.pop()
        else:
            out.append(token)
    return '/'.join(out) or '.'
