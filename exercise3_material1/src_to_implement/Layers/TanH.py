import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):
    """Hyperbolic tangent activation: f(x) = tanh(x).

    Gradient: f'(x) = 1 - tanh(x)^2 = 1 - activations^2
    We store the activation (output) instead of the input because the
    gradient formula only needs the output — no need to keep the input.
    """
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activation ** 2)
