import util
import numpy as np
from __baseoptimizer import WBGradientDescentOptimizer, SGDMixin

class BatchGradientDescentOptimizer(WBGradientDescentOptimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__(lr)
    
    def compute_gradients(self, x, y):
        m, n = x.shape
        dw = (1 / m * np.sum((util.sigmoid(x.dot(self._weights) + self._bias) - y) * x, axis=0)).reshape([n, 1])
        db = (1 / m * np.sum((util.sigmoid(x.dot(self._weights) + self._bias) - y), axis=0))

        return dw, db

class StochasticGradientDescentOptimizer(SGDMixin, BatchGradientDescentOptimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__(lr)
