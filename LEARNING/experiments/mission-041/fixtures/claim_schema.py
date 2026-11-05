"""Three legal transitions; every other state/event pair is an identity transition."""
STATES = ['new', 'active', 'closed']
EVENTS = ['open', 'close', 'reset']
# Independent exhaustive state table (rows are states; columns are events).
TABLE = [['active','new','new'], ['active','closed','active'], ['closed','closed','new']]
def expected(c):
    return TABLE[STATES.index(c[0])][EVENTS.index(c[1])]
def example_generate(rng):
    return [rng.choice(STATES), rng.choice(EVENTS)]
def generate(rng):
    return example_generate(rng)

def check(case, actual):
    return actual == expected(case)
