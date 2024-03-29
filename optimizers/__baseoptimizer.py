import numpy as np
from abc import ABC
from copy import deepcopy


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
            raise ValueError(
                f'x must be 2 dimensional, got shape {x.shape} instead')

        m, _ = x.shape

        if y.shape != (m, 1):
            raise ValueError(
                f'y must be of shape [m, 1], got shape {y.shape} instead')

    def compute_gradients(self, x, y):
        '''
        This method must be overridden and return a tuple of the form (dw, db)
        where dw is the gradient wrt weights and db is gradient wrt bias
        '''
        raise NotImplementedError()
    
    def compute_loss(self, x, y):
        '''
        This method should be overridden and return a single loss value if needed
        '''
        raise NotImplementedError()

    def _apply_gradients(self, dw, db):
        self._weights -= dw * self._lr
        self._bias -= db * self._lr

    def _perform_training_step(self, x, y):
        dw, db = self.compute_gradients(x, y)
        self._apply_gradients(dw, db)

    def perform_training_step(self, x, y, epoch):
        '''
        Epoch available here primarily to implement a learning schedule,
        just override this method and implement any lr manipulation
        '''
        self._perform_training_step(x, y)

    def fit_remembering_weights(self, x, y, epochs=1000):
        self._validate_input_dims(x, y)
        self._initialize_weight_dims(x)

        weights, biases = list(), list()

        for epoch in range(epochs):
            self.perform_training_step(x, y, epoch)
            weights.append(deepcopy(self._weights))
            biases.append(deepcopy(self._bias))

        return weights, biases

class SGDMixin(WBGradientDescentOptimizer):
    def perform_training_step(self, x, y, epoch):
        m, _ = x.shape

        random_idx = np.random.randint(0, m)

        x_random = x[random_idx][np.newaxis, :]
        y_random = y[random_idx][np.newaxis, :]

        super().perform_training_step(x_random, y_random, epoch)