import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    """SoftMax activation - converts logits to probabilities."""
    
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        # Subtract max for numerical stability
        input_shifted = input_tensor - input_tensor.max(axis=1, keepdims=True)
        
        # Compute softmax
        exp_values = np.exp(input_shifted)
        self.output_tensor = exp_values / exp_values.sum(axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        # Gradient: output * (error - sum(output * error))
        output = self.output_tensor
        sum_term = (output * error_tensor).sum(axis=1, keepdims=True)
        return output * (error_tensor - sum_term)
