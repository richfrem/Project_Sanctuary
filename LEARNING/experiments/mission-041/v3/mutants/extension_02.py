def extension(c):
    if not c:
        return ''
    tokens = c[-1].split('.')
    if len(tokens) < 2 or (len(tokens) == 2 and tokens[0] == ''):
        return ''
    return tokens[-1].lower()
