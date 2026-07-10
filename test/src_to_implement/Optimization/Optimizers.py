# Parameter update rules
import numpy as np


# Base class giving all optimizers regularizer support
class Optimizer:
    """Base optimizer — provides regularizer slot for all subclasses.

    Any concrete optimizer inherits this and gets add_regularizer()
    for free. The regularizer gradient is added in calculate_update.
    """

    # Function entry point
    def __init__(self):
        # Regularizer starts as None — means no constraint
        self.regularizer = None

    # Function entry point
    def add_regularizer(self, regularizer):
        # Attach a Constraints object (L1 or L2) to this optimizer
        self.regularizer = regularizer


# Main component implementation
class Sgd(Optimizer):
    """Vanilla stochastic gradient descent (no momentum)."""

    # Function entry point
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    # Function entry point
    def calculate_update(self, weight_tensor, gradient_tensor):
        # If a regularizer is set, its gradient is added to the actual gradient
        if self.regularizer is not None:
            gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
        # Update: w = w - learning_rate * gradient
        return weight_tensor - self.learning_rate * gradient_tensor


# Main component implementation
class SgdWithMomentum(Optimizer):
    """SGD with momentum. Keeps a velocity `v` to smooth updates."""

    # constructor receives learning_rate and momentum_rate
    # Function entry point
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0  # velocity initialized with 0

    # Function entry point
    def calculate_update(self, weight_tensor, gradient_tensor):
        # Add regularizer gradient if one is registered
        if self.regularizer is not None:
            gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
        # v^(k) = mu * v^(k-1) - eta * gradient
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        # w^(k+1) = w^(k) + v^(k)
        return weight_tensor + self.v


# Main component implementation
class Adam(Optimizer):
    """ADAM optimizer implementation with bias-corrected first and second moments."""

    # constructor receives learning_rate, mu (beta1) and rho (beta2)
    # Function entry point
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.eps = 1e-8
        self.v = 0  # first moment initialized with 0
        self.r = 0  # second moment initialized with 0
        self.k = 0  # k is an exponent used in bias correction

    # Function entry point
    def calculate_update(self, weight_tensor, gradient_tensor):
        # Add regularizer gradient if one is registered
        if self.regularizer is not None:
            gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
        self.k += 1
        g = gradient_tensor
        # v^(k) = mu * v^(k-1) + (1 - mu) * g
        self.v = self.mu * self.v + (1 - self.mu) * g
        # r^(k) = rho * r^(k-1) + (1 - rho) * g ⊙ g
        self.r = self.rho * self.r + (1 - self.rho) * g * g
        # bias correction
        v_hat = self.v / (1 - self.mu ** self.k)
        r_hat = self.r / (1 - self.rho ** self.k)
        # w^(k+1) = w^(k) - eta * v_hat / (sqrt(r_hat) + eps)
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)