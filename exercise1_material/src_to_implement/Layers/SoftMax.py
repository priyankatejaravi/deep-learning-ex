import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        # shift by row max for numerical stability, then normalise
        shifted = input_tensor - input_tensor.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.output_tensor = exp/exp.sum(axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        # softmax backward: y * (error - sum(error * y))
        y = self.output_tensor
        return y * (error_tensor - (error_tensor * y).sum(axis=1, keepdims=True))
