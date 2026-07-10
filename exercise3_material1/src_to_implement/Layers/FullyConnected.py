import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        # last row of weights is the bias
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

        self.input_tensor = None
        self.gradient_weights = None
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        self.weights = np.concatenate([weights, bias], axis=0)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        bias_column = np.ones((batch_size, 1))
        self.input_tensor = np.concatenate([input_tensor, bias_column], axis=1)
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        error_previous = np.dot(error_tensor, self.weights.T)
        error_previous = error_previous[:, :-1]

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_previous
