def list_to_tuple(lst):
    return tuple(tuple(x for x in row) for row in lst)


def list_to_tuple_with_value(lst, row_num, col_num, val):
    return tuple(tuple(x if i != row_num or j != col_num else val for j, x in enumerate(row))
                 for i, row in enumerate(lst))


def hamming_distance(s1, s2):
    f1 = flatten_tuples(s1)
    f2 = flatten_tuples(s2)

    dist = 0
    for x1, x2 in zip(f1, f2):
        if x1 != x2:
            dist += 1
    return dist


def flatten_tuples(t):
    return sum(t, ())
