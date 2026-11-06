def refund(c):
    balance, amount, capacity = c
    result = balance + amount
    if result > capacity + 1:
        result = capacity
    return result
