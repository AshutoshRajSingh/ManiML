import numpy as np
from optimizer import WBGradientDescentOptimizer

class BatchGradientDescentOptimizer(WBGradientDescentOptimizer):
    def __init__(self, lr) -> None:
        super().__init__(lr)
    def compute_gradients(self, x, y, epoch):
        m, _ = x.shape

        dw = 2 / m * np.sum((x.dot(self._weights) + self._bias - y) * x, axis=0)
        dw = dw[:, np.newaxis]
    
        db = 2 / m * np.sum(x.dot(self._weights) + self._bias - y, axis=0)

        return dw, db
