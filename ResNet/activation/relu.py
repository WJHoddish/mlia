import numpy as np


class ReLU:
    """
    Rectified Linear Unit (ReLU).
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        phase1: make local x, y;
        phase2: save data into object.

        :param x:
        :return:
        """

        y = np.maximum(0.0, x)

        self.x = x

        return y

    def backward(self, eta):
        """

        :param eta: dL/da
        :return:
        """

        eta[self.x < 0.0] = 0.0

        return eta
