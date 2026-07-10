# Utility layer to bridge convolution output and dense layers
from Layers.Base import BaseLayer


# Main component implementation
class Flatten(BaseLayer):
    """Layer that flattens the spatial dimensions into a vector while preserving batch size.

    Stores the original input shape for correct reshaping in the backward pass.
    """
    # constructor receiving no arguments
    # Function entry point
    def __init__(self):
        super().__init__()
        self.input_shape = None  # stored for backward pass

    # Function entry point
    def forward(self, input_tensor):
        # save shape for backward, keep batch dimension and flatten the rest
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    # Function entry point
    def backward(self, error_tensor):
        # reshape error back to original input shape
        return error_tensor.reshape(self.input_shape)