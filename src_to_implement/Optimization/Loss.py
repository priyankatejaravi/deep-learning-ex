import numpy as np

class CrossEntropyLoss:
    """Cross Entropy Loss for classification tasks."""
    
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # Store predictions for backward pass
        self.prediction_tensor = prediction_tensor

        # Epsilon for numerical stability (avoid log(0))
        epsilon = np.finfo(float).eps
        
        # Loss = -sum(labels * log(predictions))
        return -np.sum(label_tensor * np.log(prediction_tensor + epsilon))

    def backward(self, label_tensor):
        # Gradient = -labels / predictions
        epsilon = np.finfo(float).eps
        return -label_tensor / (self.prediction_tensor + epsilon)