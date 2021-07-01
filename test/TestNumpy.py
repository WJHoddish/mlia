import unittest

import numpy as np


def check_mat(x):
    x[x < 0.0] = 0.0

    return x


class TestNumpy(unittest.TestCase):

    def test(self):
        a = np.random.uniform(-5, 5, (2, 3))
        print(a)

        b = (a > 0).astype(int)
        print(b)

    def test2(self):
        a = np.arange(12).reshape((3, 4))
        print(id(a), a)

        b = np.arange(12).reshape((3, 4))
        print(id(b), b)

    def test_maximum(self):
        a = np.random.uniform(-1, 1, (3, 3))
        print(id(a))

        b = a
        print(id(b))

        c = a.copy()
        print(id(c))

        d = np.maximum(a, 0.0)
        print(id(d))

    def test_func(self):
        a = np.random.uniform(-1, 1, (3, 3))
        print(a, id(a))

        b = check_mat(a)
        print(b, id(b))

    def test_reshape(self):
        a = np.random.uniform(-1, 1, (3, 3))
        print(a, id(a))

        b = a.reshape((3, 3))
        print(b, id(b))

    def test_add_vector(self):
        a = np.arange(12).reshape((4, 3))

        b = np.ones(4)
        for i in range(4):
            b[i] = i + 1

        print(a)
        print(b)
        print(a + b)
