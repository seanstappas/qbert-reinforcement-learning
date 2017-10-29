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


if __name__ == '__main__':
    numpy_multiply()
