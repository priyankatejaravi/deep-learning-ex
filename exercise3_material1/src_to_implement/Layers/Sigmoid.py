import numpy as np
from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x)).

    Gradient: f'(x) = sigmoid(x) * (1 - sigmoid(x)) = activation * (1 - activation)
    We store the activation (output) instead of the input because the
    gradient formula only needs the output — no need to keep the input.
    """
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * self.activation * (1 - self.activation)
