import numpy as np
from sklearn.datasets import load_iris

def generate_dummy_linear_data(n=100, w=3, b=4):
    x = np.random.rand(n, 1)
    y = b + w * x + np.random.rand(n, 1)

    return x, y

def obtain_classification_data(norm=False):
    iris = load_iris()
    petal_dims_setosa = iris.data[:, 2:4][iris.target == 0]
    petal_dims_virginica = iris.data[:, 2:4][iris.target == 2]

    petal_dims = np.concatenate([petal_dims_setosa, petal_dims_virginica], axis=0)

    petal_labels = np.concatenate([iris.target[iris.target == 0], iris.target[iris.target == 2]])

    petal_data = np.concatenate([petal_dims, petal_labels.reshape(100, 1)], axis=1)
    petal_data[50:, 2] = 1

    np.random.shuffle(petal_data)

    petals_x = petal_data[:, :-1]
    petals_y = petal_data[:, -1:]

    petals_x_means = np.mean(petals_x, axis=0)
    petals_x_stds = np.std(petals_x, axis=0)

    if norm:
        petals_x = (petals_x - petals_x_means) / petals_x_stds

    return petals_x, petals_y

def mse_loss(x, y, weights, bias):
    return np.mean((x.dot(weights) + bias - y) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_loss(x, y, weights, bias):
    return -np.mean(y * np.log(sigmoid(x.dot(weights) + bias)) + (1 - y) * np.log(1 - sigmoid(x.dot(weights) + bias)))