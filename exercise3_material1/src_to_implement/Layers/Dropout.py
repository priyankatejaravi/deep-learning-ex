import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    """Inverted Dropout layer.

    During training, each neuron output is kept with probability `probability`
    and scaled by 1/probability so the expected value stays the same.
    During testing, the layer does nothing (pass-through).
    """
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor

        self.mask = np.random.rand(*input_tensor.shape) < self.probability
        return input_tensor * self.mask / self.probability

    def backward(self, error_tensor):
        return error_tensor * self.mask / self.probability
