# Base abstraction for all network layers

# Main component implementation
class BaseLayer:
    """Minimal base class for all layers.

    Attributes:
        trainable (bool): whether the layer has learnable parameters.
        weights: container for the layer parameters (if any).
    """
    # Function entry point
    def __init__(self):
        self.trainable = False
        self.weights = None

    # Function entry point
    def forward(self, input_tensor):
        """Forward pass through the layer. Should be overridden by subclasses."""
        pass

    # Function entry point
    def backward(self, error_tensor):
        """Backward pass through the layer. Should be overridden by subclasses."""
        pass