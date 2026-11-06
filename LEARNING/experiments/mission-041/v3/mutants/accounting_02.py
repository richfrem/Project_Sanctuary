def accounting(c):
    balance, charge, limit = c
    if charge > balance:
        return balance
    if charge >= limit:
        return balance
    return balance - charge
