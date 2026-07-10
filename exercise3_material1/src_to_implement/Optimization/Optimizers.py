import numpy as np


class Optimizer:
    """Base optimizer — provides regularizer slot for all subclasses.

    Any concrete optimizer inherits this and gets add_regularizer()
    for free. The regularizer gradient is added in calculate_update.
    """

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weights = weight_tensor
        if self.regularizer is not None:
            weights = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weights - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        weights = weight_tensor
        if self.regularizer is not None:
            weights = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weights + self.v


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.eps = 1e-8
        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        weights = weight_tensor
        if self.regularizer is not None:
            weights = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        self.k += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor
        v_hat = self.v / (1 - self.mu ** self.k)
        r_hat = self.r / (1 - self.rho ** self.k)
        return weights - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)
