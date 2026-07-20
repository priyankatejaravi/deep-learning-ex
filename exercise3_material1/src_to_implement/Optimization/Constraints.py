import numpy as np


class L1_Regularizer:
    """L2 (Frobenius) regularization also known as weight decay.

    Forward: adds alpha * sum(w^2) to the loss
    Backward: gradient contribution is 2 * alpha * w
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * np.sum(weights ** 2)
