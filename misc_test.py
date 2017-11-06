import numpy as np

from csv_utils import save_to_csv, read_from_csv
from pickler import save_to_pickle, load_from_pickle
from plotter import plot_scores

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


def test_return_none(param):
    if param is 0:
        return 'ZERO'
    if param is 1:
        return 'ONE'
    if param is 2:
        return None


def test_pickle():
    q = {(1, 2): 5, (5, 6): 10, 4: float('-inf')}
    print(q)
    save_to_pickle(q, 'test')
    q2 = load_from_pickle('test')
    print(q2)


def test_plot():
    plot_scores([1, 10, 6, 3], 'test_plot')


def test_csv():
    lst = [4, 0, 19, 300, 20, 1]
    filename = 'test'
    save_to_csv(lst, filename)
    lst2 = read_from_csv(filename)
    print(lst2)


if __name__ == '__main__':
    test_csv()
