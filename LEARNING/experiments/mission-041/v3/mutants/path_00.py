def path(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if len(stack) > 1:
                stack.pop()
        else:
            stack.append(token)
    return '/' + '/'.join(stack)
