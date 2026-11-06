def enum_map(c):
    out = []
    for token in c:
        if token == 'A':
            out.append(0)
        elif token == 'a':
            out.append(1)
        elif token == 'B':
            out.append(2)
        elif token == 'b':
            out.append(0)
        else:
            out.append(-1)
    return out
