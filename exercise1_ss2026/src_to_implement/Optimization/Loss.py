import numpy as np

class CrossEntropyLoss:
    """Cross Entropy Loss for classification tasks."""
    
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, predictions, labels):
        # Store predictions for backward pass
        self.prediction_tensor = predictions
        
        # Epsilon for numerical stability (avoid log(0))
        epsilon = np.finfo(float).eps
        
        # Loss = -sum(labels * log(predictions))
        return -np.sum(labels * np.log(predictions + epsilon))

    def backward(self, labels):
        # Gradient = -labels / predictions
        epsilon = np.finfo(float).eps
        return -labels / (self.prediction_tensor + epsilon)