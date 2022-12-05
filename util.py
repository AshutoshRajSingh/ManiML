import numpy as np

def generate_dummy_linear_data(n=100, w=3, b=4):
    x = np.random.rand(n, 1)
    y = b + w * x + np.random.rand(n, 1)

    return x, y

def mse_loss(x, y, weights, bias):
    return np.mean((x.dot(weights) + bias - y) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))