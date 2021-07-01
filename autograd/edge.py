import math
from abc import abstractmethod, ABC

from node import Node


class Edge:

    def __call__(self):
        """
        Generate a new node, which represents the result.

        :return:
        """

        raise NotImplementedError

    @abstractmethod
    def compute(self, x):
        raise NotImplementedError

    @abstractmethod
    def gradient(self, x, eta):
        raise NotImplementedError


class Add(Edge, ABC):

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, x):
        return x[0] + x[1]

    def gradient(self, x, eta):
        return [eta, eta]


class Sub(Edge, ABC):

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, x):
        return x[0] - x[1]

    def gradient(self, x, eta):
        return [eta, -eta]


class Mul(Edge, ABC):
    """
    Multiply.
    """

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, x):
        return x[0] * x[1]

    def gradient(self, x, eta):
        return [eta * x[1], eta * x[0]]


class Log(Edge, ABC):
    """
    Log(e)().
    """

    def __call__(self, a):
        return Node(self, [a])

    def compute(self, x):
        return math.log(x[0])

    def gradient(self, x, eta):
        return [eta * 1.0 / x[0]]


class Sin(Edge, ABC):

    def __call__(self, a):
        return Node(self, [a])

    def compute(self, x):
        return math.sin(x[0])

    def gradient(self, x, eta):
        return [eta * math.cos(x[0])]


class Identity(Edge, ABC):

    def __call__(self, a):
        return Node(self, [a])

    def compute(self, x):
        return x[0]

    def gradient(self, x, eta):
        return [eta]
