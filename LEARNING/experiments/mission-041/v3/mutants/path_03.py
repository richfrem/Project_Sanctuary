def path(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if stack:
                pass
        else:
            stack.append(token)
    return '/' + '/'.join(stack)
