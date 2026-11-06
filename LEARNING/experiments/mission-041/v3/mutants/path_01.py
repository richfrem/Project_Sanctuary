def path(c):
    stack = []
    for token in c:
        if token == '.':
            continue
        if token == '..':
            if stack and len(stack) != 2:
                stack.pop()
        else:
            stack.append(token)
    return '/' + '/'.join(stack)
