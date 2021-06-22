import numpy as np


class ReLU:
    """
    ReLU, the activation function.
    """

    def __init__(self):
        self.x = None
        self.y = None

        self.eta = None  # dL/dx

    def forward(self, x):
        self.x = x.copy
        self.y = np.maximum(0, x)

        return self.y

    def backward(self, eta):
        self.eta = eta.copy()
        self.eta[self.x < 0.0] = 0.0

        return self.eta
