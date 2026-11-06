def relative(c):
    out = []
    for token in c:
        if token == '.':
            continue
        if token == '..' and out and out[-1] not in {'..', 'a'}:
            out.pop()
        else:
            out.append(token)
    return '/'.join(out) or '.'
