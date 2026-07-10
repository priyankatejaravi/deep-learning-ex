# Base abstraction for all network layers

# Main component implementation
class BaseLayer:
    """Minimal base class for all layers.

    Attributes:
        trainable (bool): whether the layer has learnable parameters.
        weights: container for the layer parameters (if any).
        testing_phase (bool): True when network is in evaluation/test mode.
    """
    # Function entry point
    def __init__(self):
        self.trainable = False
        self.weights = None
        # Flag to switch layer behaviour between train and test time
        self.testing_phase = False

    # Function entry point
    def forward(self, input_tensor):
        """Forward pass through the layer. Should be overridden by subclasses."""
        pass

    # Function entry point
    def backward(self, error_tensor):
        """Backward pass through the layer. Should be overridden by subclasses."""
        pass