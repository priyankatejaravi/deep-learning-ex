# Regularization constraints: L1 and L2 penalty on weights
import numpy as np


# Main component implementation
class L2_Regularizer:
    """L2 (Frobenius) regularization also known as weight decay.

    Forward: adds alpha * sum(w^2) to the loss
    Backward: gradient contribution is 2 * alpha * w
    """

    # Function entry point
    def __init__(self, alpha):
        # alpha controls how strongly we penalize large weights
        self.alpha = alpha

    # Function entry point
    def calculate_gradient(self, weights):
        # Gradient of alpha * ||w||^2 w.r.t. w is 2 * alpha * w
        return 2 * self.alpha * weights

    # Function entry point
    def norm(self, weights):
        # The squared Frobenius norm — no sqrt so gradient is clean
        return self.alpha * np.sum(weights ** 2)


# Main component implementation
class L1_Regularizer:
    """L1 regularization — promotes sparsity in the weights.

    Forward: adds alpha * sum(|w|) to the loss
    Backward: sub-gradient is alpha * sign(w)
    """

    # Function entry point
    def __init__(self, alpha):
        # alpha controls the strength of sparsity pressure
        self.alpha = alpha

    # Function entry point
    def calculate_gradient(self, weights):
        # Sub-gradient of alpha * ||w||_1 is alpha * sign(w)
        return self.alpha * np.sign(weights)

    # Function entry point
    def norm(self, weights):
        # Sum of absolute values of all weights
        return self.alpha * np.sum(np.abs(weights))
