def refund(c):
    balance, amount, capacity = c
    result = balance + amount
    if result >= capacity:
        result = capacity
    return result
