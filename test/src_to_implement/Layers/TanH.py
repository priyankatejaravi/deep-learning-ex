# Hyperbolic tangent activation function
import numpy as np
from Layers.Base import BaseLayer


# Main component implementation
class TanH(BaseLayer):
    """Hyperbolic tangent activation: f(x) = tanh(x).

    Gradient: f'(x) = 1 - tanh(x)^2 = 1 - activations^2
    We store the activation (output) instead of the input because the
    gradient formula only needs the output — no need to keep the input.
    """

    # Function entry point
    def __init__(self):
        super().__init__()
        # Store the tanh output for the backward pass
        self.activations = None

    # Function entry point
    def forward(self, input_tensor):
        # Compute tanh and store it — gradient will reuse this
        self.activations = np.tanh(input_tensor)
        return self.activations

    # Function entry point
    def backward(self, error_tensor):
        # Gradient of tanh: derivative is (1 - tanh(x)^2)
        # Multiply element-wise with incoming error (chain rule)
        return error_tensor * (1 - self.activations ** 2)
