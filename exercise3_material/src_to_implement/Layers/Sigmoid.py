# Sigmoid activation function
import numpy as np
from Layers.Base import BaseLayer


# Main component implementation
class Sigmoid(BaseLayer):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x)).

    Gradient: f'(x) = sigmoid(x) * (1 - sigmoid(x)) = activation * (1 - activation)
    We store the activation (output) instead of the input because the
    gradient formula only needs the output — no need to keep the input.
    """

    # Function entry point
    def __init__(self):
        super().__init__()
        # Store sigmoid output for efficient backward computation
        self.activations = None

    # Function entry point
    def forward(self, input_tensor):
        # Compute sigmoid and save the output
        self.activations = 1.0 / (1.0 + np.exp(-input_tensor))
        return self.activations

    # Function entry point
    def backward(self, error_tensor):
        # Gradient of sigmoid: sigma * (1 - sigma), then chain rule with error
        return error_tensor * self.activations * (1 - self.activations)
