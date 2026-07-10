import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        shifted = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output_tensor = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        sum_term = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)
        return self.output_tensor * (error_tensor - sum_term)
