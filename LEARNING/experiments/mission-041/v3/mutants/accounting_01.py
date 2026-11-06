def accounting(c):
    balance, charge, limit = c
    if charge > balance + 1:
        return balance
    if charge > limit:
        return balance
    return balance - charge
