import numpy as np
from abc import ABC

class WBGradientDescentOptimizer(ABC):
    '''
    Base class for implementing algorithms that optimize any loss
    whose gradient is given as grad = f(xw + b)
    '''
    def __init__(self, lr) -> None:
        self._lr = lr
        self._weights = np.empty([])
        self._bias = np.random.rand(1)
    
    def _initialize_weight_dims(self, x):
        _, n = x.shape
        self._weights = np.random.rand(n, 1)
    
    def _validate_input_dims(self, x, y):
        if len(x.shape) != 2:
            raise ValueError(f'x must be 2 dimensional, got shape {x.shape} instead')

        m, _ = x.shape

        if y.shape != (m, 1):
            raise ValueError(f'y must be of shape [m, 1], got shape {y.shape} instead')

    def compute_gradients(self, x, y, epoch):
        raise NotImplementedError()

    def _apply_gradients(self, dw, db):
        self._weights -= dw * self._lr
        self._bias -= db * self._lr

    def _perform_training_step(self, x, y, epoch):
        dw, db = self.compute_gradients(x, y, epoch)
        self._apply_gradients(dw, db)

    def fit_remembering_weights(self, x, y, epochs=1000):
        self._validate_input_dims(x, y)
        self._initialize_weight_dims(x)

        weights, biases = list(), list()

        for epoch in range(epochs):
            self._perform_training_step(x, y, epoch)
            weights.append(self._weights.ravel())
            biases.append(self._bias.ravel())
        
        return weights, biases
