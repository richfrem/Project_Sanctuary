def refund(c):
    balance, amount, capacity = c
    result = balance + amount
    if result > capacity + 3:
        result = capacity
    return result
