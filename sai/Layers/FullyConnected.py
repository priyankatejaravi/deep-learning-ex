from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        # store weights privately and expose via properties
        self._weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.gradient_weights = None
        self.input_tensor = None
        self._optimizer = None

    def forward(self, input_tensor):
        # Add bias column to input
        batch_size = input_tensor.shape[0]
        bias = np.ones((batch_size, 1))
        self.input_tensor = np.concatenate([input_tensor, bias], axis=1)
        return self.input_tensor @ self.weights

    @property
    def weights(self):
        return self._weights

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    
    def backward(self, error_tensor):
        # Compute weight gradients
        self.gradient_weights = self.input_tensor.T @ error_tensor
        
        # Propagate error back (remove bias error)
        error_back = error_tensor @ self.weights.T
        error_back = error_back[:, :-1]
        
        # Update weights if optimizer exists
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        
        return error_back









