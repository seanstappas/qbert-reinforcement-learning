import numpy as np

INITIAL_PARAMETERS1 = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

INITIAL_PARAMETERS2 = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]


def numpy_equality():
    print(np.array_equal((0, 0, 0), [0, 0, 0]))
    print(np.array_equal((0, 0, 0), np.array([0, 0, 0])))


def numpy_multiply():
    a = np.array(INITIAL_PARAMETERS1, dtype=np.uint8)
    b = np.array(INITIAL_PARAMETERS2, dtype=np.uint8)
    print(a * b)


def test_dicts_and_lists():
    d = {}
    lst = (1, 2, 3)
    d[lst] = 'a'
    print(d[(1, 2, 3)])

    lst = (3, 4), (1, 2, 3)
    d[lst] = 'b'
    print(d[(3, 4), (1, 2, 3)])


if __name__ == '__main__':
    test_dicts_and_lists()
