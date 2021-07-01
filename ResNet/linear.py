import numpy as np


class Linear:

    def __init__(self, ch1, ch2):
        """

        :param ch1:
        :param ch2:
        """

        # in/output channel
        self.ch1 = ch1
        self.ch2 = ch2

        lim = np.sqrt(6.0 / (ch1 + ch2))  # xavier

        # kernel
        self.k = np.random.uniform(
            low=-lim,
            high=lim,
            size=(self.ch2, self.ch1)
        )

        # bias
        self.b = np.zeros(self.ch2)  # (ch2,)

        self.s = None
        self.x = None
        self.y = None
        self.g = None

    def forward(self, x):
        """

        :param x: (ba, ch1, 1, 1)
        :return:
        """

        self.s = x.shape
        self.x = x.reshape(x.shape[0], -1).copy()  # (ba, ch1)
        self.y = np.dot(self.x, self.k.T) + self.b.T  # y = kx + b

        return self.y

    def backward(self, g, lr):
        # dL/dk = dL/dy * dy/dk
        dL_dk = np.dot(g.T, self.x).squeeze()
        dL_db = np.sum(g.T, axis=0).reshape(self.b.shape)  # dy/db = 1

        self.g = np.dot(g, self.k).reshape(self.s)  # dL/dx

        self.k -= lr * dL_dk
        self.b -= lr * dL_db
