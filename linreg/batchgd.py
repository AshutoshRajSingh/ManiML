import numpy as np

class BatchGradientDescentOptimizer:
    def __init__(self, lr=0.01) -> None:
        self._lr = lr
        self._theta = np.empty([])

    def _compute_mse_gradient(self, x, y, theta):
        m, _ = x.shape
        return 2 / m * x.T.dot(x.dot(theta) - y)
    
    def _apply_gradient(self, gradient):
        self._theta = self._theta - gradient * self._lr
    
    def _initialize_theta_dims(self, n):
        self._theta = np.random.rand(n, 1)

    def _perform_training_step(self, x, y):
        grad = self._compute_mse_gradient(x, y, self._theta)
        self._apply_gradient(grad)
    
    def _initialize_optimizer(self, x, y):
        try:
            _, n = x.shape
            self._initialize_theta_dims(n)
        except ValueError:
            raise ValueError('x must be 2-dimensional')
        
        if y.shape[-1] != 1 or len(y.shape) != 2:
            raise ValueError(f'y must be of shape [None, 1], got {y.shape} instead')
            
    def fit_remembering_weights(self, x, y, epochs=100):
        self._initialize_optimizer(x, y)

        thetas = list()

        for _ in range(epochs):
            self._perform_training_step(x, y)
            thetas.append(self._theta)
        
        return thetas
