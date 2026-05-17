import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # store prediction for use in backward
        self.prediction_tensor = prediction_tensor
        # loss accumulated over the batch, no loops
        ep = np.finfo(float).eps
        return -np.sum(label_tensor * np.log(prediction_tensor + ep))

    def backward(self, label_tensor):
        # backpropagation starts here, no incoming error tensor needed
        ep = np.finfo(float).eps
        return -(label_tensor / (self.prediction_tensor + ep))