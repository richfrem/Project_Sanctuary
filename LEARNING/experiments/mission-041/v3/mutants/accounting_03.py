def accounting(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge > limit + 1:
        return balance
    return balance - charge
