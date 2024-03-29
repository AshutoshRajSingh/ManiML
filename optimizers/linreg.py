import numpy as np
from .__baseoptimizer import WBGradientDescentOptimizer, SGDMixin


class BatchGradientDescentOptimizer(WBGradientDescentOptimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__(lr)
    
    def compute_loss(self, x, y):
        return np.mean((x.dot(self._weights) + self._bias - y) ** 2)

    def compute_gradients(self, x, y): 
        m, _ = x.shape

        dw = 2 / m * np.sum((x.dot(self._weights) +
                            self._bias - y) * x, axis=0)
        dw = dw[:, np.newaxis]

        db = 2 / m * np.sum(x.dot(self._weights) + self._bias - y, axis=0)

        return dw, db


class StochasticGradientDescentOptimizer(SGDMixin, BatchGradientDescentOptimizer):
    def __init__(self, lr) -> None:
        super().__init__(lr)
