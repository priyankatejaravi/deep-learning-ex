# Dropout regularization layer (inverted dropout variant)
import numpy as np
from Layers.Base import BaseLayer


# Main component implementation
class Dropout(BaseLayer):
    """Inverted Dropout layer.

    During training, each neuron output is kept with probability `probability`
    and scaled by 1/probability so the expected value stays the same.
    During testing, the layer does nothing (pass-through).
    """

    # Function entry point
    def __init__(self, probability):
        super().__init__()
        # probability is the fraction of units to KEEP (not drop)
        self.probability = probability
        # mask is saved during forward so backward can reuse it
        self.mask = None

    # Function entry point
    def forward(self, input_tensor):
        if self.testing_phase:
            # At test time we just pass the input through — no masking
            return input_tensor

        # Sample a binary mask: each unit is kept with prob = probability
        self.mask = (np.random.rand(*input_tensor.shape) < self.probability)

        # Inverted dropout: scale up kept units by 1/p so the magnitude
        # stays the same on average (no change needed at test time)
        return (input_tensor * self.mask) / self.probability

    # Function entry point
    def backward(self, error_tensor):
        # Apply the same mask and scaling factor used in the forward pass
        return (error_tensor * self.mask) / self.probability
