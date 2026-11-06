def path(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if stack:
                stack.pop(0)
        else:
            stack.append(token)
    return '/' + '/'.join(stack)
