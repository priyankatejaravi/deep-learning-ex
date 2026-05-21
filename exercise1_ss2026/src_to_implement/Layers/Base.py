class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = None

    def forward(self, input_tensor):
        """Forward pass through the layer."""
        pass

    def backward(self, error_tensor):
        """Backward pass through the layer."""
        pass