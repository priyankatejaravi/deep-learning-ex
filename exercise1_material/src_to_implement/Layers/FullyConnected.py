from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.trainable = True 
        self.weights = np.random.uniform(low=0,high=1,size=(input_size + 1,output_size))

        self.gradient_weights = None
        self.input_tensor = None
        self._optimizer = None

    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1))
        self.input_tensor = np.concatenate([input_tensor, ones], axis=1)
        return self.input_tensor @ self.weights

    def backward(self, error_tensor):
       # self.gradient_weights = np.matmul(self.input_tensor.T,error_tensor)
        self.gradient_weights = self.input_tensor.T @ error_tensor

        # pass error back but remove the bias column
        error_back = error_tensor @ self.weights.T
        error_back = error_back[:, :-1]

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_back
    

    









