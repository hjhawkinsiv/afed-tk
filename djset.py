from numba import njit, types as numba_types
from numba.typed import Dict

@njit
def empty_disjoint_set(value_type):
    return Dict.empty(key_type=value_type, value_type=value_type)


@njit
def disjoint_set_find(value, disjoint_set):
    if value not in disjoint_set:
        disjoint_set[value] = value
        return value

    if disjoint_set[value] != value:
        disjoint_set[value] = disjoint_set_find(disjoint_set[value], disjoint_set)

    return disjoint_set[value]


@njit
def disjoint_set_add(value, disjoint_set):
    if value not in disjoint_set:
        disjoint_set[value] = value
    elif disjoint_set[value] != value:
        disjoint_set[value] = disjoint_set_find(disjoint_set[value], disjoint_set)


@njit
def disjoint_set_union(x, y, disjoint_set, ranks):
    px = disjoint_set_find(x, disjoint_set)
    py = disjoint_set_find(y, disjoint_set)

    if px == py:
        return

    rankx = ranks.get(px, 0)
    ranky = ranks.get(py, 0)

    ranks[px] = rankx
    ranks[py] = ranky

    if rankx < ranky:
        disjoint_set[px] = py
    else:
        disjoint_set[py] = px

        if rankx == ranky:
            ranks[px] += 1
