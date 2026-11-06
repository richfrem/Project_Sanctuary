def refund(c):
    balance, amount, capacity = c
    result = balance + amount
    if result > capacity + 4:
        result = capacity
    return result
