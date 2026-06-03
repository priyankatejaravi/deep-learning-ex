import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    """ReLU activation: f(x) = max(0, x)"""
    
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # ReLU: keep positive values, set negative to 0
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        # Gradient: 1 where input > 0, else 0
        return error_tensor * (self.input_tensor > 0)
