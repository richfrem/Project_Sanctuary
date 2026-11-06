"""Held-out operator families. Not imported by specifications or generators.

These apply lossy preprocessing at a component boundary, then call the unchanged
candidate. Development uses only constant-return. Operators, not just cases, are
excluded from development. Single-author procedural holdout, not blinded.
"""
import copy
import references

OPERATORS = ['unsigned_width_2', 'identifier_casefold', 'stable_sequence_sort',
             'sequence_tail_loss', 'signedness_erasure', 'identifier_prefix_encoding']
SEQUENCES = {'path','relative','extension','path_key','replay','last_replay',
             'revision','tombstone','required','enum_map'}


def recurse(case, transform):
    if isinstance(case, list):
        return [recurse(x, transform) for x in case]
    return transform(case)


def transform(name, operator, case):
    c = copy.deepcopy(case)
    if operator == 'unsigned_width_2':
        c = recurse(c, lambda x: x % 4 if type(x) is int else x)
    elif operator == 'identifier_casefold':
        c = recurse(c, lambda x: x.casefold() if type(x) is str else x)
    elif operator == 'signedness_erasure':
        c = recurse(c, lambda x: abs(x) if type(x) is int else x)
    elif operator == 'identifier_prefix_encoding':
        c = recurse(c, lambda x: x[:1] if type(x) is str else x)
    elif operator in ['stable_sequence_sort', 'sequence_tail_loss']:
        def apply(seq):
            return sorted(seq) if operator == 'stable_sequence_sort' else seq[:-1]
        if name in SEQUENCES:
            c = apply(c)
        elif name in ['reserve','migration']:
            c[1] = apply(c[1])
    else:
        raise KeyError(operator)
    # Keep the lossy representation in a valid ordered/subordinate domain.
    if name == 'clamp':
        c[1], c[2] = sorted(c[1:])
    if name == 'refund':
        c[0] = min(c[0], c[2])
    return c


def evaluate(name, operator, case):
    return getattr(references, name)(transform(name, operator, case))
