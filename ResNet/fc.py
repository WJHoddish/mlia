import numpy as np


class FC:
    """
    Kernel-based fully-connected linear layer + activation (sigmoid).
    """

    def __init__(self, ch1, ch2):
        # kernel (xavier)
        self.k = np.random.uniform(
            low=-np.sqrt(6.0 / (ch1 + ch2)),
            high=np.sqrt(6.0 / (ch1 + ch2)),
            size=(ch2, ch1)
        )

        # bias
        self.b = np.zeros(ch2)  # (ch2,)

        # vars
        self.s = None
        self.x = None
        self.y = None

    def forward(self, x):
        """

        :param x:
        :return:
        """

        s = x.shape

        # (ba, ch1)
        x = x.reshape(x.shape[0], -1)

        # (ba, ch1) * (ch1, ch2) + bias for each row
        y = np.dot(x, self.k.T) + self.b.T

        # sigmoid
        y = 1.0 / (1.0 + np.exp(- y))

        self.s, self.x, self.y = s, x, y

        return y

    def backward(self, eta, lr):
        """

        :param eta: dL/da
        :param lr:
        :return:
        """

        # multiply element by element
        da_dy = self.y * (1 - self.y)

        # dL/dy = dL/da * da/dy
        dL_dy = eta * da_dy

        # dL/dk = dL/dy * dy/dk, dy/dk = x
        dL_dk = np.dot(dL_dy.T, self.x).squeeze()

        # dy/db = 1
        dL_db = np.sum(dL_dy.T, axis=0).reshape(self.b.shape)

        # update model
        self.k -= lr * dL_dk
        self.b -= lr * dL_db

        return np.dot(dL_dy, self.k).reshape(self.s)
