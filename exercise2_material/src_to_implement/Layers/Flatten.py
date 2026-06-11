from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    # constructor receiving no arguments
    def __init__(self):
        super().__init__()
        self.input_shape = None  # stored for backward pass

    def forward(self, input_tensor):
        # save shape for backward, keep batch dimension and flatten the rest
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        # reshape error back to original input shape
        return error_tensor.reshape(self.input_shape)
